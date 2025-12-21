# train.py
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusers import DDPMScheduler

from model import AngleSpec, TrajectoryDeltaDiffusion
from data import ObstacleSpec, GenSpec, TrajectoryChunkDataset


def collate_fn(batch):
    # stack dict of tensors
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="runs/diff_mpc")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--max_obs", type=int, default=4)

    ap.add_argument("--train_samples", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--diffusion_steps", type=int, default=1000)

    # data gen controls
    ap.add_argument("--repair_steps", type=int, default=120)
    ap.add_argument("--repair_lr", type=float, default=0.05)

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device(args.device)

    angle = AngleSpec()
    obs = ObstacleSpec(max_obs=args.max_obs)
    gen = GenSpec(horizon=args.horizon, repair_steps=args.repair_steps, repair_lr=args.repair_lr)

    ds = TrajectoryChunkDataset(
        num_samples=args.train_samples,
        angle_spec=angle,
        obs_spec=obs,
        gen_spec=gen,
        device=str(device),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    model = TrajectoryDeltaDiffusion(
        horizon=args.horizon,
        max_obs=args.max_obs,
        angle_spec=angle,
        cond_dim=128,
        cond_channels=32,
        unet_base_channels=128,
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.diffusion_steps)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # save config
    with open(os.path.join(args.out, "config.txt"), "w") as f:
        f.write(str(vars(args)) + "\n")
        f.write("angle=" + str(asdict(angle)) + "\n")
        f.write("obs=" + str(asdict(obs)) + "\n")
        f.write("gen=" + str(asdict(gen)) + "\n")

    step = 0
    model.train()
    for epoch in range(args.epochs):
        for batch in dl:
            x0 = batch["x0_latent"].to(device)                 # (B,6)
            goal = batch["goal_xyz"].to(device)                # (B,3)
            spheres = batch["obs_spheres"].to(device)          # (B,O,4)
            deltas = batch["deltas_latent"].to(device)         # (B,H,6)

            B, H, D = deltas.shape
            assert H == args.horizon and D == 6

            # diffusion: sample noise and timestep
            noise = torch.randn_like(deltas)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

            noisy = noise_scheduler.add_noise(deltas, noise, t)

            eps_pred = model.forward_eps(noisy, t, x0, goal, spheres)

            loss = F.mse_loss(eps_pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 50 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.6f}")

            if step % 500 == 0 and step > 0:
                ckpt = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "step": step,
                    "args": vars(args),
                    "angle": asdict(angle),
                    "obs": asdict(obs),
                    "gen": asdict(gen),
                }
                torch.save(ckpt, os.path.join(args.out, f"ckpt_{step:07d}.pt"))

            step += 1

    torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.out, "final.pt"))
    print("Done. Saved:", os.path.join(args.out, "final.pt"))


if __name__ == "__main__":
    main()