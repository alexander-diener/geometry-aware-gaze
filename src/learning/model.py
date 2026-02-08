import torch
import torch.nn as nn

class ResidualMLP(nn.Module):
    """
    Predicts small residual correction (delta_yaw, delta_pitch).
    Includes dropout to enable MC-dropout uncertainty at inference time.
    """
    def __init__(self, hidden: int = 64, p_drop: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
