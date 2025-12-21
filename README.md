# Diffusion continuum arm

A project for path planning a 3 segment continuum arm with obstacles.

## Installation

```sh
uv sync --frozen
```

## Train

```sh
python diffusion_continuum_arm/train.py \
  --out runs/diff_mpc \
  --device cuda \
  --horizon 20 \
  --max_obs 4 \
  --train_samples 5000 \
  --batch_size 16 \
  --epochs 5 \
  --lr 2e-4 \
  --diffusion_steps 1000 \
  --repair_steps 120 \
  --repair_lr 0.05
```

## Inference

```sh
python diffusion_continuum_arm/mpc.py --ckpt runs/diff_mpc/final.pt --candidates 32 --ddim_steps 30 --execute_k 1
```