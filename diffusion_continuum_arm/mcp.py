# mpc.py
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from tqdm import tqdm

from diffusers import DDIMScheduler

from model import AngleSpec, TrajectoryDeltaDiffusion
from data import (
    ObstacleSpec,
    GenSpec,
    sample_random_start_goal_latent,
    nominal_deltas,
    integrate_latents,
    decode_latent_to_phys,
    fk_points,
    collision_loss_spheres,
    make_obstacles_adversarial,
)

# ----------------------------
# Scoring / MPC utilities
# ----------------------------

@dataclass
class MPCSpec:
    horizon: int = 20
    candidates: int = 32          # how many chunks to sample each MPC step
    execute_k: int = 1            # how many steps to execute from best chunk
    steps: int = 40               # MPC iterations
    goal_eps: float = 0.05        # stop when tip within this distance
    ddim_steps: int = 30          # diffusion sampling steps per chunk
    guidance_scale: float = 0.0   # optional: classifier-free guidance later (kept 0 here)


def score_chunks(
    x0_latent: torch.Tensor,          # (B,6)
    deltas: torch.Tensor,             # (B,H,6)
    goal_xyz: torch.Tensor,           # (B,3)
    spheres: torch.Tensor,            # (B,O,4)
    angle_spec: AngleSpec,
    gen: GenSpec,
) -> torch.Tensor:
    """
    Returns score (lower is better): (B,)
    """
    x_lat = integrate_latents(x0_latent, deltas)
    x_phys = decode_latent_to_phys(x_lat, angle_spec)
    points, tip = fk_points(x_phys, points_per_segment=gen.points_per_segment)

    # goal
    d_tip = torch.linalg.norm(tip - goal_xyz.unsqueeze(1), dim=-1)  # (B,H)
    goal_terminal = d_tip[:, -1] ** 2
    goal_path = (d_tip ** 2).mean(dim=1)

    # collision
    coll = collision_loss_spheres(points, spheres, arm_radius=gen.arm_radius, sigma=gen.coll_sigma)

    # smoothness / jerk (on deltas)
    smooth = (deltas ** 2).mean(dim=(1, 2))
    jerk = ((deltas[:, 1:, :] - deltas[:, :-1, :]) ** 2).mean(dim=(1, 2))

    score = (
        gen.w_goal_terminal * goal_terminal
        + gen.w_goal_path * goal_path
        + gen.w_coll * coll
        + gen.w_smooth * smooth
        + gen.w_jerk * jerk
    )
    return score


@torch.no_grad()
def ddim_sample_deltas(
    model: TrajectoryDeltaDiffusion,
    scheduler: DDIMScheduler,
    x0_latent: torch.Tensor,      # (B,6)
    goal_xyz: torch.Tensor,       # (B,3)
    spheres: torch.Tensor,        # (B,O,4)
    num_candidates: int,
    ddim_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample candidate delta chunks with DDIM.
    Returns: deltas (B*num_candidates, H, 6)
    """
    B = x0_latent.shape[0]
    H = model.horizon

    # tile conditioning to match candidates
    x0_t = x0_latent.repeat_interleave(num_candidates, dim=0)
    g_t = goal_xyz.repeat_interleave(num_candidates, dim=0)
    s_t = spheres.repeat_interleave(num_candidates, dim=0)

    # start from noise in delta space
    sample = torch.randn((B * num_candidates, H, 6), device=device)

    scheduler.set_timesteps(ddim_steps, device=device)

    for t in scheduler.timesteps:
        # diffusers passes timesteps as tensor-ish; ensure correct shape
        tt = torch.full((B * num_candidates,), int(t), device=device, dtype=torch.long)
        eps = model.forward_eps(sample, tt, x0_t, g_t, s_t)

        out = scheduler.step(eps, t, sample)
        sample = out.prev_sample

    # rate-limit clamp for physical plausibility
    sample = model.clamp_deltas(sample)
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to final.pt or ckpt_*.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--max_obs", type=int, default=4)

    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--execute_k", type=int, default=1)
    ap.add_argument("--mpc_steps", type=int, default=40)
    ap.add_argument("--ddim_steps", type=int, default=30)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--goal_eps", type=float, default=0.05)

    # data/scoring params (match training defaults)
    ap.add_argument("--points_per_segment", type=int, default=10)
    ap.add_argument("--arm_radius", type=float, default=0.02)
    ap.add_argument("--coll_sigma", type=float, default=0.02)
    ap.add_argument("--w_goal_terminal", type=float, default=10.0)
    ap.add_argument("--w_goal_path", type=float, default=0.25)
    ap.add_argument("--w_coll", type=float, default=25.0)
    ap.add_argument("--w_smooth", type=float, default=0.5)
    ap.add_argument("--w_jerk", type=float, default=0.25)

    args = ap.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    angle = AngleSpec()
    obs_spec = ObstacleSpec(max_obs=args.max_obs)
    gen = GenSpec(
        horizon=args.horizon,
        points_per_segment=args.points_per_segment,
        arm_radius=args.arm_radius,
        coll_sigma=args.coll_sigma,
        w_goal_terminal=args.w_goal_terminal,
        w_goal_path=args.w_goal_path,
        w_coll=args.w_coll,
        w_smooth=args.w_smooth,
        w_jerk=args.w_jerk,
    )

    # Build model and load weights
    model = TrajectoryDeltaDiffusion(
        horizon=args.horizon,
        max_obs=args.max_obs,
        angle_spec=angle,
        cond_dim=128,
        cond_channels=32,
        unet_base_channels=128,
    ).to(device)
    model.eval()

    ckpt = torch.load(args.ckpt, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        # final.pt saved as {"model": state_dict, ...}
        model.load_state_dict(ckpt["model"], strict=True)

    # DDIM scheduler (faster sampling)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        prediction_type="epsilon",
        clip_sample=False,
    )

    # Make a single demo scenario (B=1)
    x0_latent, goal_xyz = sample_random_start_goal_latent(1, angle, device)

    # create obstacles near a nominal path so they matter
    d_nom = nominal_deltas(1, gen.horizon, angle, device)
    x_nom_lat = integrate_latents(x0_latent, d_nom)
    x_nom_phys = decode_latent_to_phys(x_nom_lat, angle)
    pts_nom, tip_nom = fk_points(x_nom_phys, points_per_segment=gen.points_per_segment)
    spheres = make_obstacles_adversarial(pts_nom, tip_nom, obs_spec, collide_prob=0.75)

    # MPC loop
    print("Starting MPC...")
    for it in tqdm(range(args.mpc_steps)):
        # Check stop condition
        x0_phys = decode_latent_to_phys(x0_latent.unsqueeze(0), angle)  # (1,1,6) hacky
        _, tip0 = fk_points(x0_phys, points_per_segment=gen.points_per_segment)
        dist0 = torch.linalg.norm(tip0[:, -1, :] - goal_xyz, dim=-1).item()

        if dist0 < args.goal_eps:
            print(f"\nReached goal at iter={it}, dist={dist0:.4f}")
            break

        # Sample candidates
        deltas_all = ddim_sample_deltas(
            model=model,
            scheduler=scheduler,
            x0_latent=x0_latent,
            goal_xyz=goal_xyz,
            spheres=spheres,
            num_candidates=args.candidates,
            ddim_steps=args.ddim_steps,
            device=device,
        )  # (Nc, H, 6)

        # Score candidates
        x0_rep = x0_latent.repeat_interleave(args.candidates, dim=0)
        g_rep = goal_xyz.repeat_interleave(args.candidates, dim=0)
        s_rep = spheres.repeat_interleave(args.candidates, dim=0)

        scores = score_chunks(x0_rep, deltas_all, g_rep, s_rep, angle, gen)  # (Nc,)
        best = torch.argmin(scores).item()

        best_deltas = deltas_all[best : best + 1]  # (1,H,6)

        # Execute first k steps
        k = max(1, min(args.execute_k, args.horizon))
        x0_latent = x0_latent + best_deltas[:, :k, :].sum(dim=1)

        # Optional: wrap theta components for neatness
        # x0_latent[:, 0] = (x0_latent[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi
        # x0_latent[:, 2] = (x0_latent[:, 2] + torch.pi) % (2 * torch.pi) - torch.pi
        # x0_latent[:, 4] = (x0_latent[:, 4] + torch.pi) % (2 * torch.pi) - torch.pi

        if it % 5 == 0:
            print(f"\niter={it} dist={dist0:.4f} best_score={scores[best].item():.3f}")

    print("Done.")


if __name__ == "__main__":
    main()

# python mpc.py --ckpt runs/diff_mpc/final.pt --candidates 32 --ddim_steps 30 --execute_k 1
