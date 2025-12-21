# train.py
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
import contextlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusers import DDPMScheduler

from model import AngleSpec, TrajectoryDeltaDiffusion
from data import ObstacleSpec, GenSpec, TrajectoryChunkDataset

from torch.utils.tensorboard import SummaryWriter
import time


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

    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--max_obs", type=int, default=4)

    ap.add_argument("--train_samples", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--diffusion_steps", type=int, default=1000)

    # data gen controls
    ap.add_argument("--repair_steps", type=int, default=120)
    ap.add_argument("--repair_lr", type=float, default=0.05)

    ap.add_argument("--tb", action="store_true", help="Enable TensorBoard logging")
    ap.add_argument("--log_every", type=int, default=1, help="Log scalars every N steps")
    ap.add_argument("--metrics_every", type=int, default=1, help="Compute extra metrics every N steps (adds a bit of overhead)")

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

    writer = None
    if args.tb:
        tb_dir = os.path.join(args.out, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_text("hparams", str(vars(args)))
    start_time = time.time()

    def _sync_if_needed():
        # Ensure accurate timing on GPU backends
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    prev_iter_end = time.perf_counter()

    # save config
    with open(os.path.join(args.out, "config.txt"), "w") as f:
        f.write(str(vars(args)) + "\n")
        f.write("angle=" + str(asdict(angle)) + "\n")
        f.write("obs=" + str(asdict(obs)) + "\n")
        f.write("gen=" + str(asdict(gen)) + "\n")

    step = 0
    model.train()
    for epoch in range(args.epochs):
        dl_iter = iter(dl)
        while True:
            t_data_start = time.perf_counter()
            try:
                batch = next(dl_iter)
            except StopIteration:
                break
            data_time = time.perf_counter() - prev_iter_end

            x0 = batch["x0_latent"].to(device)                 # (B,6)
            goal = batch["goal_xyz"].to(device)                # (B,3)
            spheres = batch["obs_spheres"].to(device)          # (B,O,4)
            deltas = batch["deltas_latent"].to(device)         # (B,H,6)

            # Start compute timing after data is on device
            _sync_if_needed()
            t_compute_start = time.perf_counter()

            B, H, D = deltas.shape
            assert H == args.horizon and D == 6

            # diffusion: sample noise and timestep
            noise = torch.randn_like(deltas)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

            noisy = noise_scheduler.add_noise(deltas, noise, t)

            eps_pred = model.forward_eps(noisy, t, x0, goal, spheres)

            _sync_if_needed()
            fwd_time = time.perf_counter() - t_compute_start

            loss = F.mse_loss(eps_pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            _sync_if_needed()
            bwd_time = time.perf_counter() - t_compute_start - fwd_time

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            _sync_if_needed()
            step_time = time.perf_counter() - t_compute_start - fwd_time - bwd_time
            total_time = time.perf_counter() - t_data_start
            prev_iter_end = time.perf_counter()

            if step % args.log_every == 0:
                # Console
                elapsed = time.time() - start_time
                it_per_sec = (step + 1) / max(elapsed, 1e-9)
                lr = opt.param_groups[0]["lr"]
                print(
                    f"epoch={epoch} step={step} loss={loss.item():.6f} lr={lr:.2e} it/s={it_per_sec:.2f} "
                    f"data={data_time:.3f}s fwd={fwd_time:.3f}s bwd={bwd_time:.3f}s opt={step_time:.3f}s total={total_time:.3f}s"
                )

                # TensorBoard
                if writer is not None:
                    writer.add_scalar("train/loss", loss.item(), step)
                    writer.add_scalar("train/lr", lr, step)
                    writer.add_scalar("train/iters_per_sec", it_per_sec, step)
                    writer.add_scalar("time/data_s", data_time, step)
                    writer.add_scalar("time/fwd_s", fwd_time, step)
                    writer.add_scalar("time/bwd_s", bwd_time, step)
                    writer.add_scalar("time/opt_s", step_time, step)
                    writer.add_scalar("time/iter_total_s", total_time, step)

            if (writer is not None) and (step % args.metrics_every == 0):
                # Simple trajectory-stat metrics (no FK required)
                with torch.no_grad():
                    # deltas stats
                    writer.add_scalar("data/delta_abs_mean", deltas.abs().mean().item(), step)
                    writer.add_scalar("data/delta_l2_mean", torch.linalg.norm(deltas, dim=-1).mean().item(), step)

                    if deltas.shape[1] > 1:
                        jerk = (deltas[:, 1:, :] - deltas[:, :-1, :]).pow(2).mean().item()
                        writer.add_scalar("data/jerk_mse", jerk, step)

                    # noise prediction quality summary
                    # (correlation-ish proxy: mse already logged; log mean abs error too)
                    writer.add_scalar("train/eps_abs_err_mean", (eps_pred - noise).abs().mean().item(), step)

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
    if writer is not None:
        writer.flush()
        writer.close()
    print("Done. Saved:", os.path.join(args.out, "final.pt"))


if __name__ == "__main__":
    main()
