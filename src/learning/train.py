import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.geometry.camera import yaw_pitch_to_unit, angular_error_deg
from src.geometry.gaze_model import geometry_baseline
from src.learning.model import ResidualMLP

def make_synth(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)

    # Head pose in radians (small-ish range)
    head_yaw = rng.uniform(-0.6, 0.6, size=n)
    head_pitch = rng.uniform(-0.4, 0.4, size=n)

    # Eye relative angles (smaller)
    eye_yaw = rng.uniform(-0.4, 0.4, size=n)
    eye_pitch = rng.uniform(-0.25, 0.25, size=n)

    # "True" gaze: geometry baseline + non-linear bias (domain effect)
    base = geometry_baseline(head_yaw, head_pitch, eye_yaw, eye_pitch)

    # Create a synthetic "domain bias": e.g., camera/subject-specific offset depends on head yaw/pitch
    bias_yaw = 0.10 * np.sin(1.5 * head_yaw) + 0.04 * head_pitch
    bias_pitch = 0.06 * np.sin(1.2 * head_pitch) - 0.03 * head_yaw

    true = yaw_pitch_to_unit(head_yaw + eye_yaw + bias_yaw, head_pitch + eye_pitch + bias_pitch)

    # Add observation noise to "measured" eye angles (as if landmarks/noise)
    eye_yaw_obs = eye_yaw + rng.normal(0, 0.03, size=n)
    eye_pitch_obs = eye_pitch + rng.normal(0, 0.02, size=n)

    x = np.stack([head_yaw, head_pitch, eye_yaw_obs, eye_pitch_obs], axis=-1).astype(np.float32)

    # Regression target: residual in angle space to correct baseline
    # residual = (true angles) - (baseline angles)
    # Here we use the known synthetic bias as target, but in real data we'd learn it from supervision.
    y = np.stack([bias_yaw, bias_pitch], axis=-1).astype(np.float32)
    return x, y, base.astype(np.float32), true.astype(np.float32)

def eval_deg(base_vec, pred_vec, true_vec):
    b = angular_error_deg(base_vec, true_vec).mean()
    p = angular_error_deg(pred_vec, true_vec).mean()
    return float(b), float(p)

def main():
    os.makedirs("figures", exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    n_train, n_test = 50000, 10000
    x_tr, y_tr, base_tr, true_tr = make_synth(n_train, seed=1)
    x_te, y_te, base_te, true_te = make_synth(n_test, seed=2)

    ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)

    model = ResidualMLP(hidden=64)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.SmoothL1Loss()

    model.train()
    for epoch in range(5):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/5")
        for xb, yb in pbar:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss))

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(x_te)
        res = model(x).numpy()
        head_yaw, head_pitch, eye_yaw_obs, eye_pitch_obs = x_te.T
        pred_vec = yaw_pitch_to_unit(head_yaw + eye_yaw_obs + res[:, 0],
                                     head_pitch + eye_pitch_obs + res[:, 1])

    base_mean, pred_mean = eval_deg(base_te, pred_vec, true_te)
    print(f"Mean angular error (deg) - geometry baseline: {base_mean:.3f}")
    print(f"Mean angular error (deg) - geometry + learned residual: {pred_mean:.3f}")

    # Save a tiny text summary for reproducibility
    with open("figures/results.txt", "w", encoding="utf-8") as f:
        f.write(f"baseline_deg={base_mean:.6f}\n")
        f.write(f"geom_plus_residual_deg={pred_mean:.6f}\n")

    # Quick plot: distribution comparison
    import matplotlib.pyplot as plt
    base_err = angular_error_deg(base_te, true_te)
    pred_err = angular_error_deg(pred_vec, true_te)

    plt.figure()
    plt.hist(base_err, bins=60, alpha=0.7, label="Geometry baseline")
    plt.hist(pred_err, bins=60, alpha=0.7, label="Geometry + learned residual")
    plt.xlabel("Angular error (deg)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/error_hist.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
