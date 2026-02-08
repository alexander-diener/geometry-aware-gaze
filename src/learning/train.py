import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.geometry.camera import yaw_pitch_to_unit, angular_error_deg
from src.geometry.gaze_model import geometry_baseline
from src.learning.model import ResidualMLP


# -----------------------------
# Data generation (synthetic)
# -----------------------------
def make_synth(n: int, seed: int, domain: str):
    """
    Synthetic generator with explicit domain parameters to simulate domain shift.
    Domain A vs B differ in systematic bias and observation noise.
    """
    rng = np.random.default_rng(seed)

    head_yaw = rng.uniform(-0.6, 0.6, size=n)
    head_pitch = rng.uniform(-0.4, 0.4, size=n)

    eye_yaw = rng.uniform(-0.4, 0.4, size=n)
    eye_pitch = rng.uniform(-0.25, 0.25, size=n)

    if domain == "A":
        bias_yaw = 0.10 * np.sin(1.5 * head_yaw) + 0.04 * head_pitch
        bias_pitch = 0.06 * np.sin(1.2 * head_pitch) - 0.03 * head_yaw
        noise_yaw = 0.03
        noise_pitch = 0.02
    elif domain == "B":
        bias_yaw = 0.14 * np.sin(1.1 * head_yaw + 0.2) + 0.02 * head_pitch + 0.03
        bias_pitch = 0.09 * np.sin(1.7 * head_pitch - 0.1) - 0.02 * head_yaw - 0.02
        noise_yaw = 0.05
        noise_pitch = 0.035
    else:
        raise ValueError("domain must be 'A' or 'B'")

    # baseline using ideal angles (kept for reference only; we will evaluate baseline from x later)
    base_ideal = geometry_baseline(head_yaw, head_pitch, eye_yaw, eye_pitch)

    # ground truth gaze (ideal + systematic bias)
    true = yaw_pitch_to_unit(
        head_yaw + eye_yaw + bias_yaw,
        head_pitch + eye_pitch + bias_pitch
    )

    # observed (noisy) eye angles – what the model actually sees
    eye_yaw_obs = eye_yaw + rng.normal(0, noise_yaw, size=n)
    eye_pitch_obs = eye_pitch + rng.normal(0, noise_pitch, size=n)

    x = np.stack([head_yaw, head_pitch, eye_yaw_obs, eye_pitch_obs], axis=-1).astype(np.float32)
    y = np.stack([bias_yaw, bias_pitch], axis=-1).astype(np.float32)

    return x, y, base_ideal.astype(np.float32), true.astype(np.float32)


# -----------------------------
# Metrics
# -----------------------------
def mean_ang_err_deg(pred_vec: np.ndarray, true_vec: np.ndarray) -> float:
    return float(angular_error_deg(pred_vec, true_vec).mean())


# -----------------------------
# Training
# -----------------------------
def train_model(x_tr: np.ndarray, y_tr: np.ndarray, epochs: int = 6) -> ResidualMLP:
    ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)

    model = ResidualMLP(hidden=64)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.SmoothL1Loss()

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{epochs}")
        for xb, yb in pbar:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss))
    return model


# -----------------------------
# Evaluation (baseline + residual)
# IMPORTANT: baseline computed from OBSERVED inputs x
# so alpha=0 matches baseline exactly.
# -----------------------------
def eval_bundle(x: np.ndarray, true_vec: np.ndarray, model: ResidualMLP, alpha: float = 1.0):
    head_yaw, head_pitch, eye_yaw_obs, eye_pitch_obs = x.T

    # baseline from observed angles
    base_vec = yaw_pitch_to_unit(head_yaw + eye_yaw_obs, head_pitch + eye_pitch_obs)
    base_err = mean_ang_err_deg(base_vec, true_vec)

    model.eval()
    with torch.no_grad():
        res = model(torch.from_numpy(x)).numpy()

    pred_vec = yaw_pitch_to_unit(
        head_yaw + eye_yaw_obs + alpha * res[:, 0],
        head_pitch + eye_pitch_obs + alpha * res[:, 1],
    )
    pred_err = mean_ang_err_deg(pred_vec, true_vec)
    return base_err, pred_err


# -----------------------------
# OOD alpha (Mahalanobis distance in x-space)
# -----------------------------
def fit_mahalanobis(x_tr: np.ndarray, eps: float = 1e-6):
    mu = x_tr.mean(axis=0)
    xc = x_tr - mu
    cov = (xc.T @ xc) / (len(x_tr) - 1)
    cov += np.eye(cov.shape[0]) * eps
    inv = np.linalg.inv(cov)
    return mu, inv


def mahalanobis_dist(x: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray):
    xc = x - mu
    d2 = np.einsum("ni,ij,nj->n", xc, inv_cov, xc)
    return np.sqrt(np.maximum(d2, 0.0))


def dist_to_alpha(d: np.ndarray, q: float = 0.90, floor: float = 0.0):
    thr = np.quantile(d, q)
    alpha = np.exp(-np.maximum(d - thr, 0.0))
    if floor > 0:
        alpha = np.maximum(alpha, floor)
    return alpha


def eval_bundle_ood_alpha(x: np.ndarray, true_vec: np.ndarray, model: ResidualMLP,
                         mu: np.ndarray, inv_cov: np.ndarray, q: float = 0.90):
    head_yaw, head_pitch, eye_yaw_obs, eye_pitch_obs = x.T

    # baseline (observed)
    base_vec = yaw_pitch_to_unit(head_yaw + eye_yaw_obs, head_pitch + eye_pitch_obs)
    base_err = mean_ang_err_deg(base_vec, true_vec)

    # alpha from OOD distance
    d = mahalanobis_dist(x, mu, inv_cov)
    alpha = dist_to_alpha(d, q=q, floor=0.0)

    # residual prediction
    model.eval()
    with torch.no_grad():
        res = model(torch.from_numpy(x)).numpy()

    pred_vec = yaw_pitch_to_unit(
        head_yaw + eye_yaw_obs + alpha * res[:, 0],
        head_pitch + eye_pitch_obs + alpha * res[:, 1],
    )
    pred_err = mean_ang_err_deg(pred_vec, true_vec)
    return base_err, pred_err, alpha, d


# -----------------------------
# Plots
# -----------------------------
def plot_domain_shift(results: dict, out_path: str):
    import matplotlib.pyplot as plt
    labels = list(results.keys())
    base = [results[k][0] for k in labels]
    pred = [results[k][1] for k in labels]

    x = np.arange(len(labels))
    w = 0.35

    plt.figure()
    plt.bar(x - w/2, base, width=w, label="Geometry baseline")
    plt.bar(x + w/2, pred, width=w, label="Geometry + residual")
    plt.xticks(x, labels)
    plt.ylabel("Mean angular error (deg)")
    plt.title("Domain shift evaluation (synthetic)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_ab_damping(ab_sweep: dict, out_path: str):
    import matplotlib.pyplot as plt
    alphas = sorted(ab_sweep.keys())
    baseline = [ab_sweep[a][0] for a in alphas]
    pred = [ab_sweep[a][1] for a in alphas]

    plt.figure()
    plt.plot(alphas, baseline, marker="o", label="Geometry baseline (A→B)")
    plt.plot(alphas, pred, marker="o", label="Geometry + α·residual (A→B)")
    plt.xlabel("alpha")
    plt.ylabel("Mean angular error (deg)")
    plt.title("Damping residual under domain shift (A→B)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_ood_alpha(alpha: np.ndarray, d: np.ndarray, out_path: str):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(alpha, bins=60, alpha=0.85)
    plt.xlabel("alpha (OOD-based)")
    plt.ylabel("count")
    plt.title("OOD-based alpha distribution (A→B)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    plt.figure()
    plt.hist(d, bins=60, alpha=0.85)
    plt.xlabel("Mahalanobis distance")
    plt.ylabel("count")
    plt.title("OOD distance distribution (A→B)")
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_dist.png"), dpi=200)
    plt.close()


def plot_error_hist_AA(xA_te: np.ndarray, trueA_te: np.ndarray, modelA: ResidualMLP, out_path: str):
    import matplotlib.pyplot as plt

    head_yaw, head_pitch, eye_yaw_obs, eye_pitch_obs = xA_te.T
    base_vec = yaw_pitch_to_unit(head_yaw + eye_yaw_obs, head_pitch + eye_pitch_obs)

    modelA.eval()
    with torch.no_grad():
        res = modelA(torch.from_numpy(xA_te)).numpy()

    pred_vec = yaw_pitch_to_unit(
        head_yaw + eye_yaw_obs + res[:, 0],
        head_pitch + eye_pitch_obs + res[:, 1],
    )

    base_err = angular_error_deg(base_vec, trueA_te)
    pred_err = angular_error_deg(pred_vec, trueA_te)

    plt.figure()
    plt.hist(base_err, bins=60, alpha=0.7, label="Geometry baseline (A→A)")
    plt.hist(pred_err, bins=60, alpha=0.7, label="Geometry + residual (A→A)")
    plt.xlabel("Angular error (deg)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs("figures", exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    n_train, n_test = 60000, 12000

    xA_tr, yA_tr, _, trueA_tr = make_synth(n_train, seed=1, domain="A")
    xA_te, yA_te, _, trueA_te = make_synth(n_test, seed=2, domain="A")
    xB_te, yB_te, _, trueB_te = make_synth(n_test, seed=3, domain="B")

    # Train model on A
    modelA = train_model(xA_tr, yA_tr, epochs=6)

    # Optional: train on B (to show within-domain learning)
    xB_tr, yB_tr, _, trueB_tr = make_synth(n_train, seed=4, domain="B")
    modelB = train_model(xB_tr, yB_tr, epochs=6)

    # Domain shift results
    res = {}
    res["A→A"] = eval_bundle(xA_te, trueA_te, modelA, alpha=1.0)
    res["A→B"] = eval_bundle(xB_te, trueB_te, modelA, alpha=1.0)
    res["B→B"] = eval_bundle(xB_te, trueB_te, modelB, alpha=1.0)

    # A→B damping sweep
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    ab_sweep = {a: eval_bundle(xB_te, trueB_te, modelA, alpha=a) for a in alphas}

    # OOD-based alpha for A→B
    muA, invA = fit_mahalanobis(xA_tr)
    b_ood, p_ood, alphaB, dB = eval_bundle_ood_alpha(xB_te, trueB_te, modelA, muA, invA, q=0.90)
    res["A→B (OOD α)"] = (b_ood, p_ood)

    # Write results
    with open("figures/results.txt", "w", encoding="utf-8") as f:
        for k, (b, p) in res.items():
            f.write(f"{k}_baseline_deg={b:.6f}\n")
            f.write(f"{k}_geom_plus_residual_deg={p:.6f}\n")

        f.write("\n# A→B damping sweep (alpha)\n")
        for a in alphas:
            b, p = ab_sweep[a]
            f.write(f"A→B_alpha={a:.2f}_baseline_deg={b:.6f}\n")
            f.write(f"A→B_alpha={a:.2f}_geom_plus_residual_deg={p:.6f}\n")

        f.write("\n# OOD alpha stats (A→B)\n")
        f.write(f"A→B_ood_alpha_mean={alphaB.mean():.6f}\n")
        f.write(f"A→B_ood_alpha_p10={np.quantile(alphaB, 0.10):.6f}\n")
        f.write(f"A→B_ood_alpha_p50={np.quantile(alphaB, 0.50):.6f}\n")
        f.write(f"A→B_ood_alpha_p90={np.quantile(alphaB, 0.90):.6f}\n")

    # Plots
    plot_error_hist_AA(xA_te, trueA_te, modelA, "figures/error_hist.png")
    plot_domain_shift(res, "figures/domain_shift.png")
    plot_ab_damping(ab_sweep, "figures/ab_damping.png")
    plot_ood_alpha(alphaB, dB, "figures/ood_alpha_hist.png")

    print("Domain shift results:")
    for k, (b, p) in res.items():
        print(f"{k}: baseline={b:.3f} deg, geom+residual={p:.3f} deg")


if __name__ == "__main__":
    main()
