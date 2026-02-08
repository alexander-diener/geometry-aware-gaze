import torch
import torch.nn as nn

class ResidualMLP(nn.Module):
    """
    Predicts small residual correction (delta_yaw, delta_pitch) on top of geometry baseline.
    Input features: [head_yaw, head_pitch, eye_yaw, eye_pitch]
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
