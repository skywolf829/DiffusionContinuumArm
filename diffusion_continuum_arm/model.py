# model.py
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn

from diffusers import UNet1DModel


@dataclass
class AngleSpec:
    # theta is periodic; phi is bounded via sigmoid
    phi_min: float = -3.141592653589793
    phi_max: float = 3.141592653589793
    # per-step rate limits (for delta outputs)
    dtheta_max: float = 0.15  # radians per step
    du_phi_max: float = 0.25  # latent step per step


def decode_phi(u_phi: torch.Tensor, phi_min: float, phi_max: float) -> torch.Tensor:
    # Hard-bounded phi via sigmoid
    return phi_min + (phi_max - phi_min) * torch.sigmoid(u_phi)


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    # Optional: keep theta in [-pi, pi] for numerical neatness
    return (theta + torch.pi) % (2 * torch.pi) - torch.pi


class SceneEncoder(nn.Module):
    """
    Encodes conditioning into a fixed vector:
      - start config x0 (theta, u_phi for 3 segments) -> 6
      - goal position g -> 3
      - obstacles (spheres): (cx, cy, cz, r) * max_obs
    """
    def __init__(self, max_obs: int, cond_dim: int = 128):
        super().__init__()
        self.max_obs = max_obs
        in_dim = 6 + 3 + max_obs * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, cond_dim),
        )

    def forward(self, x0_latent: torch.Tensor, goal_xyz: torch.Tensor, obs_spheres: torch.Tensor) -> torch.Tensor:
        """
        x0_latent: (B, 6)  [theta1,u1, theta2,u2, theta3,u3]
        goal_xyz:  (B, 3)
        obs_spheres: (B, max_obs, 4) [cx,cy,cz,r], padded with zeros if fewer
        """
        B = x0_latent.shape[0]
        obs_flat = obs_spheres.reshape(B, -1)
        inp = torch.cat([x0_latent, goal_xyz, obs_flat], dim=-1)
        return self.net(inp)  # (B, cond_dim)


class TrajectoryDeltaDiffusion(nn.Module):
    """
    Diffusion model over delta-latents:
      - outputs delta theta (3) and delta u_phi (3) per timestep
      - the UNet predicts noise eps on the noisy deltas
    """
    def __init__(
        self,
        horizon: int,
        max_obs: int,
        angle_spec: AngleSpec = AngleSpec(),
        cond_dim: int = 128,
        unet_base_channels: int = 128,
        cond_channels: int = 32,
    ):
        super().__init__()
        self.horizon = horizon
        self.max_obs = max_obs
        self.angle_spec = angle_spec
        self.cond_dim = cond_dim
        self.cond_channels = cond_channels

        self.scene_encoder = SceneEncoder(max_obs=max_obs, cond_dim=cond_dim)
        self.cond_to_channels = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.SiLU(),
            nn.Linear(128, cond_channels),
        )

       

        min_len = 8
        max_down = 2  # stable for horizons like 16-64 (e.g., 32 -> 16 -> 8)
        num_down = int(math.floor(math.log2(max(horizon / float(min_len), 1.0))))
        num_down = max(1, min(max_down, num_down))

        down_block_types = ("DownBlock1D",) * num_down
        up_block_types = ("UpBlock1D",) * num_down
        block_out_channels = tuple([unet_base_channels] * num_down)

        self.unet = UNet1DModel(
            sample_size=horizon,                 # length L
            in_channels=6 + cond_channels,        # 6 delta dims + cond channels
            out_channels=6,                       # predict eps for the 6 delta dims
            layers_per_block=2,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            act_fn="silu",
        )

    @torch.no_grad()
    def clamp_deltas(self, d: torch.Tensor) -> torch.Tensor:
        """
        d: (B, T, 6) delta latents: [dtheta1, du1, dtheta2, du2, dtheta3, du3]
        Apply per-step rate limits.
        """
        dtheta_max = self.angle_spec.dtheta_max
        du_max = self.angle_spec.du_phi_max

        d_out = d.clone()
        # theta deltas at indices 0,2,4; u deltas at 1,3,5
        for idx in (0, 2, 4):
            d_out[..., idx] = torch.clamp(d_out[..., idx], -dtheta_max, dtheta_max)
        for idx in (1, 3, 5):
            d_out[..., idx] = torch.clamp(d_out[..., idx], -du_max, du_max)
        return d_out

    def forward_eps(
        self,
        noisy_deltas: torch.Tensor,
        timesteps: torch.Tensor,
        x0_latent: torch.Tensor,
        goal_xyz: torch.Tensor,
        obs_spheres: torch.Tensor,
    ) -> torch.Tensor:
        """
        noisy_deltas: (B, T, 6)
        timesteps: (B,) int64 or int tensor
        returns eps_pred: (B, T, 6)
        """
        B, T, D = noisy_deltas.shape
        assert T == self.horizon and D == 6

        cond_vec = self.scene_encoder(x0_latent, goal_xyz, obs_spheres)          # (B, cond_dim)
        cond_ch = self.cond_to_channels(cond_vec)                                # (B, cond_channels)
        cond_ch = cond_ch.unsqueeze(-1).expand(B, self.cond_channels, T)         # (B, cond_channels, T)

        x = noisy_deltas.permute(0, 2, 1)                                        # (B, 6, T)
        x_in = torch.cat([x, cond_ch], dim=1)                                    # (B, 6+cond, T)

        out = self.unet(x_in, timesteps).sample                                  # (B, 6, T)
        return out.permute(0, 2, 1)                                              # (B, T, 6)

    def rollout_from_deltas(
        self,
        x0_latent: torch.Tensor,
        deltas_latent: torch.Tensor,
        wrap_theta: bool = False,
    ) -> torch.Tensor:
        """
        Integrate latent deltas to get latent configs over time.
        x0_latent: (B, 6)
        deltas_latent: (B, T, 6)
        returns x_latent: (B, T, 6) latent configs at each timestep
        """
        B, T, D = deltas_latent.shape
        x = torch.zeros((B, T, D), device=deltas_latent.device, dtype=deltas_latent.dtype)

        prev = x0_latent
        for t in range(T):
            prev = prev + deltas_latent[:, t, :]
            if wrap_theta:
                prev = prev.clone()
                prev[:, 0] = wrap_angle(prev[:, 0])
                prev[:, 2] = wrap_angle(prev[:, 2])
                prev[:, 4] = wrap_angle(prev[:, 4])
            x[:, t, :] = prev
        return x

    def decode_latent_to_physical(self, x_latent: torch.Tensor) -> torch.Tensor:
        """
        x_latent: (B, T, 6) [theta1,u1, theta2,u2, theta3,u3]
        returns x_phys: (B, T, 6) [theta1,phi1, theta2,phi2, theta3,phi3]
        """
        phi_min, phi_max = self.angle_spec.phi_min, self.angle_spec.phi_max
        x = x_latent.clone()
        # decode u->phi at indices 1,3,5
        x[..., 1] = decode_phi(x[..., 1], phi_min, phi_max)
        x[..., 3] = decode_phi(x[..., 3], phi_min, phi_max)
        x[..., 5] = decode_phi(x[..., 5], phi_min, phi_max)
        return x