# data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from continuum_kinematics_torch import continuum_fk_points as fk_points
from model import AngleSpec, decode_phi


@dataclass
class ObstacleSpec:
    max_obs: int = 4
    # spheres only: (cx,cy,cz,r)
    r_min: float = 0.05
    r_max: float = 0.20
    # workspace bounds for centers
    c_min: float = -0.8
    c_max: float = 0.8


@dataclass
class GenSpec:
    horizon: int = 20
    # number of points sampled along the arm for collision checking
    points_per_segment: int = 10
    # NOTE: legacy fields from the old "repair" optimizer approach.
    # They are unused in the random-walk data generator (kept for CLI/backward-compat).
    repair_steps: int = 120
    repair_lr: float = 0.05
    # weights
    w_goal_terminal: float = 10.0
    w_goal_path: float = 0.25
    w_coll: float = 25.0
    w_smooth: float = 0.5
    w_jerk: float = 0.25
    # collision softness
    coll_sigma: float = 0.02
    arm_radius: float = 0.02


def collision_loss_spheres(points: torch.Tensor, spheres: torch.Tensor, arm_radius: float, sigma: float) -> torch.Tensor:
    """
    points:  (B,T,P,3)
    spheres: (B,O,4) [cx,cy,cz,r]
    returns scalar loss per batch: (B,)
    """
    B, T, P, _ = points.shape
    spheres.shape[1]

    c = spheres[..., :3]  # (B,O,3)
    r = spheres[..., 3:4] # (B,O,1)

    # distances: (B,T,P,O)
    diff = points.unsqueeze(3) - c.unsqueeze(1).unsqueeze(1)  # (B,T,P,O,3)
    dist = torch.linalg.norm(diff, dim=-1)                    # (B,T,P,O)
    clearance = dist - (r.squeeze(-1).unsqueeze(1).unsqueeze(1) + arm_radius)  # (B,T,P,O)

    # Penalize negative clearance smoothly
    # softplus(-clearance/sigma) is ~0 when clearance positive, grows when negative
    pen = F.softplus(-clearance / sigma)  # (B,T,P,O)
    return pen.mean(dim=(1, 2, 3))        # (B,)


def make_obstacles_adversarial(
    points_nominal: torch.Tensor,
    tip_nominal: torch.Tensor,
    obs_spec: ObstacleSpec,
    collide_prob: float = 0.7,
) -> torch.Tensor:
    """
    Place spheres near the nominal path so obstacles matter sometimes.
    points_nominal: (B,T,P,3)
    tip_nominal:    (B,T,3)
    returns spheres: (B,max_obs,4)
    """
    B, T, P, _ = points_nominal.shape
    device = points_nominal.device
    dtype = points_nominal.dtype

    spheres = torch.zeros((B, obs_spec.max_obs, 4), device=device, dtype=dtype)

    # pick anchor points along the arm or tip trajectory to place obstacles nearby
    # choose either random arm sample or along the straight tip-to-goal line
    for b in range(B):
        for j in range(obs_spec.max_obs):
            r = torch.empty((), device=device, dtype=dtype).uniform_(obs_spec.r_min, obs_spec.r_max)

            if torch.rand((), device=device) < collide_prob:
                # near a random point on the nominal arm sweep
                tb = torch.randint(low=0, high=T, size=(1,), device=device).item()
                pb = torch.randint(low=0, high=P, size=(1,), device=device).item()
                anchor = points_nominal[b, tb, pb]
                offset = torch.randn((3,), device=device, dtype=dtype) * (r * 0.75)
                c = anchor + offset
            else:
                # random in workspace
                c = torch.empty((3,), device=device, dtype=dtype).uniform_(obs_spec.c_min, obs_spec.c_max)

            spheres[b, j, :3] = c
            spheres[b, j, 3] = r

    return spheres


def sample_random_start_goal_latent(B: int, angle_spec: AngleSpec, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      x0_latent: (B,6) [theta1,u1, theta2,u2, theta3,u3]
      goal_xyz:  (B,3)
    """
    # theta free / periodic
    theta = torch.rand((B, 3), device=device) * (2 * torch.pi)

    # u_phi unconstrained; random around 0 tends to center of sigmoid range
    u_phi = torch.randn((B, 3), device=device) * 0.5

    x0 = torch.zeros((B, 6), device=device)
    x0[:, 0] = theta[:, 0]
    x0[:, 1] = u_phi[:, 0]
    x0[:, 2] = theta[:, 1]
    x0[:, 3] = u_phi[:, 1]
    x0[:, 4] = theta[:, 2]
    x0[:, 5] = u_phi[:, 2]

    # goal in workspace (toy)
    goal = torch.empty((B, 3), device=device).uniform_(-0.7, 0.7)
    goal[:, 2] = torch.empty((B,), device=device).uniform_(0.0, 0.8)  # positive z a bit
    return x0, goal


def nominal_deltas(B: int, H: int, angle_spec: AngleSpec, device: torch.device) -> torch.Tensor:
    """
    Produce smooth-ish random deltas as a starting guess:
    deltas: (B,H,6)
    """
    d = torch.randn((B, H, 6), device=device) * 0.04

    # smooth in time (simple box filter)
    kernel = torch.ones((1, 1, 5), device=device) / 5.0
    d_ch = d.permute(0, 2, 1).reshape(B * 6, 1, H)
    d_sm = F.conv1d(F.pad(d_ch, (2, 2), mode="replicate"), kernel)
    d = d_sm.reshape(B, 6, H).permute(0, 2, 1)

    # apply rate limits
    dtheta_max = angle_spec.dtheta_max
    du_max = angle_spec.du_phi_max
    for idx in (0, 2, 4):
        d[..., idx] = torch.clamp(d[..., idx], -dtheta_max, dtheta_max)
    for idx in (1, 3, 5):
        d[..., idx] = torch.clamp(d[..., idx], -du_max, du_max)
    return d


def integrate_latents(x0_latent: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """
    x0_latent: (B,6), deltas:(B,H,6) => x_latent:(B,H,6)
    Safe autograd version (no in-place writes).
    """
    return x0_latent.unsqueeze(1) + torch.cumsum(deltas, dim=1)


def decode_latent_to_phys(x_latent: torch.Tensor, angle_spec: AngleSpec) -> torch.Tensor:
    """
    x_latent: (B,H,6) [theta1,u1,...]
    returns x_phys: (B,H,6) [theta1,phi1,...]
    """
    phi_min, phi_max = angle_spec.phi_min, angle_spec.phi_max
    x = x_latent.clone()
    x[..., 1] = decode_phi(x[..., 1], phi_min, phi_max)
    x[..., 3] = decode_phi(x[..., 3], phi_min, phi_max)
    x[..., 5] = decode_phi(x[..., 5], phi_min, phi_max)
    return x




# ----------------- New helpers for random-walk data generation -----------------

def min_clearance_to_spheres(points: torch.Tensor, spheres: torch.Tensor, arm_radius: float) -> torch.Tensor:
    """Compute minimum clearance between a swept set of points and a batch of spheres.

    points:  (B,T,P,3)
    spheres: (B,O,4) [cx,cy,cz,r]

    Returns:
        min_clearance: (B,O) minimum over (T,P) of dist - (r + arm_radius)
    """
    c = spheres[..., :3]  # (B,O,3)
    r = spheres[..., 3]   # (B,O)

    diff = points.unsqueeze(3) - c.unsqueeze(1).unsqueeze(1)    # (B,T,P,O,3)
    dist = torch.linalg.norm(diff, dim=-1)                      # (B,T,P,O)
    clearance = dist - (r.unsqueeze(1).unsqueeze(1) + arm_radius)
    return clearance.amin(dim=(1, 2))                            # (B,O)


def sample_nonintersecting_spheres(
    points: torch.Tensor,
    obs_spec: ObstacleSpec,
    arm_radius: float,
    margin: float = 0.01,
    max_tries: int = 200,
) -> torch.Tensor:
    """Sample spheres that do NOT intersect the arm sweep.

    Strategy: rejection sample each sphere center/radius until min clearance > margin.

    points: (B,T,P,3)
    Returns: spheres (B,max_obs,4)
    """
    B, T, P, _ = points.shape
    device = points.device
    dtype = points.dtype

    spheres = torch.zeros((B, obs_spec.max_obs, 4), device=device, dtype=dtype)

    for b in range(B):
        placed = 0
        tries = 0
        while placed < obs_spec.max_obs and tries < max_tries:
            tries += 1

            r = torch.empty((), device=device, dtype=dtype).uniform_(obs_spec.r_min, obs_spec.r_max)
            c = torch.empty((3,), device=device, dtype=dtype).uniform_(obs_spec.c_min, obs_spec.c_max)

            cand = torch.zeros((1, 1, 4), device=device, dtype=dtype)
            cand[0, 0, :3] = c
            cand[0, 0, 3] = r

            # (1,1) min clearance vs this candidate
            mc = min_clearance_to_spheres(points[b : b + 1], cand, arm_radius=arm_radius)[0, 0]
            if mc > margin:
                spheres[b, placed, :3] = c
                spheres[b, placed, 3] = r
                placed += 1

        # If we fail to place all spheres, remaining ones stay zeros.
        # That's OK; collision loss treats r=0 spheres as near-no-op.

    return spheres


def random_walk_deltas(
    B: int,
    H: int,
    angle_spec: AngleSpec,
    device: torch.device,
    step_sigma: float = 0.04,
    smooth_kernel: int = 5,
) -> torch.Tensor:
    """Fast random-walk delta generator (no inner optimization).

    Produces deltas: (B,H,6) with simple temporal smoothing + rate limits.
    """
    d = torch.randn((B, H, 6), device=device) * step_sigma

    # Smooth in time (box filter) to avoid jitter
    if smooth_kernel and smooth_kernel > 1:
        k = smooth_kernel
        pad = k // 2
        kernel = torch.ones((1, 1, k), device=device) / float(k)
        d_ch = d.permute(0, 2, 1).reshape(B * 6, 1, H)
        d_sm = F.conv1d(F.pad(d_ch, (pad, pad), mode="replicate"), kernel)
        d = d_sm.reshape(B, 6, H).permute(0, 2, 1)

    # Apply rate limits (out-of-place, MPS friendly)
    theta_idx = torch.tensor([0, 2, 4], device=device)
    uphi_idx = torch.tensor([1, 3, 5], device=device)

    d_theta = torch.clamp(d.index_select(-1, theta_idx), -angle_spec.dtheta_max, angle_spec.dtheta_max)
    d_uphi = torch.clamp(d.index_select(-1, uphi_idx), -angle_spec.du_phi_max, angle_spec.du_phi_max)

    d_out = torch.empty_like(d)
    d_out[..., 0] = d_theta[..., 0]
    d_out[..., 2] = d_theta[..., 1]
    d_out[..., 4] = d_theta[..., 2]
    d_out[..., 1] = d_uphi[..., 0]
    d_out[..., 3] = d_uphi[..., 1]
    d_out[..., 5] = d_uphi[..., 2]

    return d_out


class TrajectoryChunkDataset(Dataset):
    """
    Generates (x0, goal, obstacles, deltas*) on the fly using "repair".
    For real scale, you may want to precompute to disk, but this is easiest to start.
    """
    def __init__(
        self,
        num_samples: int,
        angle_spec: AngleSpec,
        obs_spec: ObstacleSpec,
        gen_spec: GenSpec,
        device: str = "cpu",
    ):
        self.num_samples = num_samples
        self.angle_spec = angle_spec
        self.obs_spec = obs_spec
        self.gen_spec = gen_spec
        self.device = torch.device(device)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # single-sample generation (kept simple)
        x0, goal = sample_random_start_goal_latent(1, self.angle_spec, self.device)

        # Fast random-walk trajectory (this is the training target distribution)
        deltas_star = random_walk_deltas(1, self.gen_spec.horizon, self.angle_spec, self.device)

        # Roll out to get the swept arm points, then sample obstacles that DO NOT intersect the sweep.
        x_lat = integrate_latents(x0, deltas_star)
        x_phys = decode_latent_to_phys(x_lat, self.angle_spec)
        pts, tip = fk_points(x_phys, points_per_segment=self.gen_spec.points_per_segment)

        spheres = sample_nonintersecting_spheres(
            pts,
            self.obs_spec,
            arm_radius=self.gen_spec.arm_radius,
            margin=max(self.gen_spec.coll_sigma, 0.01),
            max_tries=400,
        )

        return {
            "x0_latent": x0.squeeze(0),           # (6,)
            "goal_xyz": goal.squeeze(0),          # (3,)
            "obs_spheres": spheres.squeeze(0),    # (max_obs,4)
            "deltas_latent": deltas_star.squeeze(0),  # (H,6)
        }