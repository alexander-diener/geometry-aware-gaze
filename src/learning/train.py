import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.geometry.camera import yaw_pitch_to_unit, angular_error_deg
from src.geometry.gaze_model import geometry_baseline
from src.learning.model import ResidualMLP


def make_synth(n: int, seed: int, domain: str):
    """
    Synthetic generator with explicit domain parameters to simulate domain shift.
    Domain A vs B differ in systematic bias and observation noise (proxy for camera/subject conditions).
    """
    rng = np.random.default_rng(seed)

    # Head pose in radians
    head_yaw = rng.uniform(-0.6, 0.6, size=n)
    head_pitch = rng.uniform(-0.4, 0.4, size=n)

    # Eye relative angles
    eye_yaw = rng.uniform(-0.4, 0.4, size=n)
    eye_pitch = rng.uniform(-0.25, 0.25, size=n)

    # Domain-specific bias + noise
    if domain == "A":
        # mild bias, moderate noise
        bias_yaw = 0.10 * np.sin(1.5 * head_yaw) + 0.04 * head_pitch
        bias_pitch = 0.06 * np.sin(1.2 * head_pitch) - 0.03 * head_yaw
        noise_yaw = 0.03
        noise_pitch = 0.02
    elif domain == "B":
        # different bias shape + higher noise
        bias_yaw = 0.14 * np.sin(1.1 * head_yaw + 0.2) + 0.02 * head_pitch + 0.03
        bias_pitch = 0.09 * np.sin(1.7 * head_pitch - 0.1) - 0.02 * head_yaw - 0.02
        noise_yaw = 0.05
        noise_pitch = 0.035
    else:
        raise ValueError("domain must be 'A' or 'B'")

    # Baseline (geometry-inspired)
    base = geometry_baseline(head_yaw, head_pitch, eye_yaw, eye_pitch)

    # "True" gaze = baseline + systematic bias (domain effect)
    true = yaw_pitch_to_unit(
        head_yaw + eye_yaw + bias_yaw,
        head_pitch + eye_pitch + bias_pitch
    )

    # Observed eye angles (noisy measurements)
    eye_yaw_obs = eye_yaw + rng.normal(0, noise_yaw, size=n)
    eye_pitch_obs = eye_pitch + rng.normal(0, noise_pitch, size=n)

    x = np.stack([head_yaw, head_pitch, eye_yaw_obs, eye_pitch_obs], axis=-1).astype(np.float32)

    # Learning target: residual correction (delta yaw/pitch)
    y = np.stack([bias_yaw, bias_pitch], axis=-1).astype(np.float32)

    return x, y, base.astype(np.float32), true.astype(np.float32)


def mean_ang_err_deg(pred_vec: np.ndarray, true_vec: np.ndarray) -> float:
    return float(angular_error_deg(pred_vec, true_vec).mean())


def eval_bundle(x: np.ndarray, base_vec: np.ndarray, true_vec: np.ndarray, model: ResidualMLP):
    """
    Returns mean error for:
    - geometry baseline
    - geometry + learned residual
    """
    base_err = mean_ang_err_deg(base_vec, true_vec)

    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x)
        res = model(xt).numpy()

    head_yaw, head_pitch, eye_yaw_obs, eye_pitch_obs = x.T
    pred_vec = yaw_pitch_to_unit(
        head_yaw + eye_yaw_obs + res[:, 0],
        head_pitch + eye_pitch_obs + res[:, 1],
    )
    pred_err = mean_ang_err_deg(pred_vec, true_vec)
    return base_err, pred_err


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


def plot_domain_shift(results, out_path: str):
    """
    results: dict with keys like:
      "A->A": (base_err, pred_err), etc.
    """
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    base = [results[k][0] for k in labels]
    pred = [results[k][1] for k in labels]

    x = np.arange(len(labels))
    w = 0.35

    plt.figure()
    plt.bar(x - w/2, base, width=w, label="Geometry baseline")
    plt.bar(x + w/2, pred, width=w, label="Geometry + learned residual")
    plt.xticks(x, labels)
    plt.ylabel("Mean angular error (deg)")
    plt.title("Domain shift evaluation (synthetic)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    os.makedirs("figures", exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    # Train on Domain A, evaluate on A and B
    n_train, n_test = 60000, 12000

    xA_tr, yA_tr, baseA_tr, trueA_tr = make_synth(n_train, seed=1, domain="A")
    xA_te, yA_te, baseA_te, trueA_te = make_synth(n_test, seed=2, domain="A")
    xB_te, yB_te, baseB_te, trueB_te = make_synth(n_test, seed=3, domain="B")

    modelA = train_model(xA_tr, yA_tr, epochs=6)

    res = {}
    res["A→A"] = eval_bundle(xA_te, baseA_te, trueA_te, modelA)
    res["A→B"] = eval_bundle(xB_te, baseB_te, trueB_te, modelA)

    # Optional: also train on B (to show within-domain)
    xB_tr, yB_tr, baseB_tr, trueB_tr = make_synth(n_train, seed=4, domain="B")
    modelB = train_model(xB_tr, yB_tr, epochs=6)
    res["B→B"] = eval_bundle(xB_te, baseB_te, trueB_te, modelB)

    # Write results
    with open("figures/results.txt", "w", encoding="utf-8") as f:
        for k, (b, p) in res.items():
            f.write(f"{k}_baseline_deg={b:.6f}\n")
            f.write(f"{k}_geom_plus_residual_deg={p:.6f}\n")

    # Keep the previous histogram for A→A (nice distribution view)
    base_err = angular_error_deg(baseA_te, trueA_te)
    modelA.eval()
    with torch.no_grad():
        xt = torch.from_numpy(xA_te)
        r = modelA(xt).numpy()
    head_yaw, head_pitch, eye_yaw_obs, eye_pitch_obs = xA_te.T
    predA = yaw_pitch_to_unit(head_yaw + eye_yaw_obs + r[:, 0], head_pitch + eye_pitch_obs + r[:, 1])
    pred_err = angular_error_deg(predA, trueA_te)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(base_err, bins=60, alpha=0.7, label="Geometry baseline (A→A)")
    plt.hist(pred_err, bins=60, alpha=0.7, label="Geometry + residual (A→A)")
    plt.xlabel("Angular error (deg)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/error_hist.png", dpi=200)
    plt.close()

    # New domain shift plot
    plot_domain_shift(res, "figures/domain_shift.png")

    print("Domain shift results:")
    for k, (b, p) in res.items():
        print(f"{k}: baseline={b:.3f} deg, geom+residual={p:.3f} deg")


if __name__ == "__main__":
    main()
