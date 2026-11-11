"""
Density estimation & crowd mass integration module.
Includes:
  1. CSRNet PyTorch architecture & checkpoint loader with fast CPU fallback.
  2. Normalization Option 1 (Existing): Global Crowd Density Normalization.
  3. Normalization Option 2 (Proposed): Active Crowd Mass Normalization with Soft Gating.
"""

import os
from typing import Optional, Tuple
import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class CSRNet(nn.Module):
        def __init__(self):
            super(CSRNet, self).__init__()
            self.frontend = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            )
            self.backend = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
            )

        def forward(self, x):
            return self.backend(self.frontend(x))


def estimate_density_map(
    frame_bgr: np.ndarray,
    model: Optional[object] = None,
    device: str = "cpu"
) -> np.ndarray:
    """
    Computes CSRNet density map if model weights are present.
    Otherwise produces an adaptive edge/intensity luminance density prior for evaluation.
    """
    h, w, _ = frame_bgr.shape
    if model is not None and TORCH_AVAILABLE:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_norm = (rgb - mean) / std
        tensor = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            raw_density = model(tensor).squeeze().cpu().numpy()
            raw_density = np.clip(raw_density, 0.0, None)

        density_full = cv2.resize(raw_density, (w, h), interpolation=cv2.INTER_LINEAR) / 64.0
        return np.clip(density_full, 0.0, None)
    else:
        # High-fidelity crowd-contrast spatial density proxy when CNN checkpoint is offline
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 2.0)
        grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(grad_x, grad_y)
        grad_norm = cv2.normalize(grad_mag, None, 0.0, 1.0, cv2.NORM_MINMAX)
        density_proxy = (grad_norm ** 1.5) * 0.02
        return density_proxy.astype(np.float32)


def compute_density_kinetic_energy(
    density_map: np.ndarray,
    mag: np.ndarray,
    active_mask: np.ndarray,
    normalization_mode: str = "csrnet_active_mass",
    crowd_floor: float = 4.0,
) -> Tuple[float, float, float]:
    """
    Computes density-weighted kinetic energy.
    Returns:
      (energy_score, active_crowd_mass, total_crowd_count)
    """
    total_crowd_count = float(np.sum(density_map))
    active_crowd_mass = float(np.sum(density_map[active_mask]))

    if active_crowd_mass <= 1e-5 or total_crowd_count <= 1e-4 or np.sum(active_mask) == 0:
        return 0.0, 0.0, total_crowd_count

    active_mags = mag[active_mask]
    v_cap = float(np.percentile(active_mags, 99.0))
    mag_clipped = np.clip(mag, 0.0, v_cap)
    pointwise_energy = 0.5 * density_map * np.square(mag_clipped)
    active_density_energy = float(np.sum(pointwise_energy[active_mask]))

    if normalization_mode == "csrnet_global":
        # Existing formulation: divides by total scene crowd
        e_score = active_density_energy / (total_crowd_count + 1e-3)
    elif normalization_mode == "csrnet_active_mass":
        # Proposed formulation: divides by active participating crowd mass
        # and modulates with crowd floor gate to prevent lone runner false alarms
        e_local = active_density_energy / (active_crowd_mass + 1e-3)
        crowd_significance = np.tanh(active_crowd_mass / crowd_floor)
        e_score = float(e_local * crowd_significance)
    else:
        raise ValueError(f"Unknown density normalization mode: {normalization_mode}")

    return float(e_score), active_crowd_mass, total_crowd_count
