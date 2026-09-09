# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Pure-Jittor inference entry.
The whole inference pipeline in this file uses Jittor only.
"""

import argparse
import math
import os

import jittor as jt
import numpy as np
from PIL import Image
from tqdm import tqdm


def create_npz_from_sample_folder(sample_dir, num=50_000):
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def mpi_barrier():
    jt.sync_all()
    if jt.in_mpi:
        token = jt.array([1.0]).mpi_all_reduce()
        token.sync()


def var_numel(var):
    return int(np.prod(var.shape))


def load_legacy_checkpoints(state_dict, encoder_depth):
    new_state_dict = {}
    for key, value in state_dict.items():
        if "decoder_blocks" in key:
            parts = key.split(".")
            new_idx = int(parts[1]) + encoder_depth
            parts[0] = "blocks"
            parts[1] = str(new_idx)
            new_key = ".".join(parts)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def expand_t_like_x(t, x_cur):
    dims = [1] * (len(x_cur.shape) - 1)
    return t.reshape((t.shape[0], *dims))


def get_score_from_velocity(vt, xt, t, path_type="linear"):
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t = 1.0 - t
        d_alpha_t = jt.ones_like(xt) * -1.0
        sigma_t = t
        d_sigma_t = jt.ones_like(xt)
    elif path_type == "cosine":
        alpha_t = jt.cos(t * np.pi / 2.0)
        sigma_t = jt.sin(t * np.pi / 2.0)
        d_alpha_t = -np.pi / 2.0 * jt.sin(t * np.pi / 2.0)
        d_sigma_t = np.pi / 2.0 * jt.cos(t * np.pi / 2.0)
    else:
        raise NotImplementedError(f"path_type={path_type} is not supported.")

    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t ** 2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    return (reverse_alpha_ratio * vt - xt) / var


def compute_diffusion(t_cur):
    return 2.0 * t_cur


def split_two(x):
    half = x.shape[0] // 2
    return x[:half], x[half:]


def euler_maruyama_sampler_jittor(
    model,
    latents,
    y,
    num_steps=20,
    cfg_scale=1.0,
    guidance_low=0.0,
    guidance_high=1.0,
    path_type="linear",
    cls_latents=None,
    cls_cfg_scale=1.5,
):
    if cfg_scale > 1.0:
        y_null = jt.ones((y.shape[0],), dtype=y.dtype) * 1000

    x_next = latents.astype("float32")
    cls_x_next = cls_latents.astype("float32")
    t_steps = np.linspace(1.0, 0.04, num_steps).astype(np.float32)
    t_steps = np.concatenate([t_steps, np.array([0.0], dtype=np.float32)], axis=0)

    for i in range(len(t_steps) - 2):
        t_cur = float(t_steps[i])
        t_next = float(t_steps[i + 1])
        dt = t_next - t_cur
        x_cur = x_next
        cls_x_cur = cls_x_next

        if cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high:
            model_input = jt.concat([x_cur, x_cur], dim=0)
            cls_model_input = jt.concat([cls_x_cur, cls_x_cur], dim=0)
            y_cur = jt.concat([y, y_null], dim=0)
        else:
            model_input = x_cur
            cls_model_input = cls_x_cur
            y_cur = y

        time_input = jt.ones((model_input.shape[0],), dtype="float32") * t_cur
        eps = jt.randn(x_cur.shape, dtype="float32") * np.sqrt(abs(dt))
        cls_eps = jt.randn(cls_x_cur.shape, dtype="float32") * np.sqrt(abs(dt))
        diffusion = compute_diffusion(t_cur)

        v_cur, _, cls_v_cur = model(
            model_input.astype("float32"),
            time_input.astype("float32"),
            y=y_cur,
            cls_token=cls_model_input.astype("float32"),
        )
        v_cur = v_cur.astype("float32")
        cls_v_cur = cls_v_cur.astype("float32")

        s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
        cls_s_cur = get_score_from_velocity(cls_v_cur, cls_model_input, time_input, path_type=path_type)
        d_cur = v_cur - 0.5 * diffusion * s_cur
        cls_d_cur = cls_v_cur - 0.5 * diffusion * cls_s_cur

        if cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high:
            d_cond, d_uncond = split_two(d_cur)
            d_cur = d_uncond + cfg_scale * (d_cond - d_uncond)
            cls_d_cond, cls_d_uncond = split_two(cls_d_cur)
            if cls_cfg_scale > 0:
                cls_d_cur = cls_d_uncond + cls_cfg_scale * (cls_d_cond - cls_d_uncond)
            else:
                cls_d_cur = cls_d_cond

        x_next = x_cur + d_cur * dt + np.sqrt(diffusion) * eps
        cls_x_next = cls_x_cur + cls_d_cur * dt + np.sqrt(diffusion) * cls_eps

    t_cur = float(t_steps[-2])
    t_next = float(t_steps[-1])
    dt = t_next - t_cur
    x_cur = x_next
    cls_x_cur = cls_x_next

    if cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high:
        model_input = jt.concat([x_cur, x_cur], dim=0)
        cls_model_input = jt.concat([cls_x_cur, cls_x_cur], dim=0)
        y_cur = jt.concat([y, y_null], dim=0)
    else:
        model_input = x_cur
        cls_model_input = cls_x_cur
        y_cur = y

    time_input = jt.ones((model_input.shape[0],), dtype="float32") * t_cur
    v_cur, _, cls_v_cur = model(
        model_input.astype("float32"),
        time_input.astype("float32"),
        y=y_cur,
        cls_token=cls_model_input.astype("float32"),
    )
    v_cur = v_cur.astype("float32")
    cls_v_cur = cls_v_cur.astype("float32")

    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    cls_s_cur = get_score_from_velocity(cls_v_cur, cls_model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    cls_d_cur = cls_v_cur - 0.5 * diffusion * cls_s_cur

    if cfg_scale > 1.0 and guidance_low <= t_cur <= guidance_high:
        d_cond, d_uncond = split_two(d_cur)
        d_cur = d_uncond + cfg_scale * (d_cond - d_uncond)
        cls_d_cond, cls_d_uncond = split_two(cls_d_cur)
        if cls_cfg_scale > 0:
            cls_d_cur = cls_d_uncond + cls_cfg_scale * (cls_d_cond - cls_d_uncond)
        else:
            cls_d_cur = cls_d_cond

    return x_cur + dt * d_cur


def load_jittor_infer_components():
    try:
        from models.sit_jittor import SiT_models  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing Jittor SiT model. Please provide models/sit_jittor.py with SiT_models."
        ) from exc
    try:
        from models.vae_jittor import AutoencoderKLJittor  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing Jittor VAE decoder. Please provide models/vae_jittor.py with AutoencoderKLJittor."
        ) from exc
    return SiT_models, AutoencoderKLJittor


def main(args):
    jt.flags.use_cuda = 1
    rank = jt.rank
    world_size = jt.world_size
    seed = args.global_seed * world_size + rank
    jt.set_global_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    SiT_models, AutoencoderKLJittor = load_jittor_infer_components()

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=True,
        z_dims=[int(z_dim) for z_dim in args.projector_embed_dims.split(",")],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    )

    if args.ckpt:
        if args.ckpt.endswith(".pt"):
            raise RuntimeError(
                "Detected PyTorch checkpoint (.pt). Please convert first:\n"
                "python tools/convert_pt_to_jt.py --pt YOUR.pt --out YOUR_jt.pkl --state-key ema"
            )
        state_dict = jt.load(args.ckpt)
        if isinstance(state_dict, dict) and "ema" in state_dict:
            state_dict = state_dict["ema"]
        if args.legacy:
            state_dict = load_legacy_checkpoints(state_dict=state_dict, encoder_depth=args.encoder_depth)
        model.load_state_dict(state_dict)
    elif rank == 0:
        print("Warning: --ckpt not provided, using random-initialized SiT Jittor model.")
    model.eval()

    vae = AutoencoderKLJittor.from_pretrained(args.vae_ckpt)
    vae.eval()

    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = (
        os.path.basename(args.ckpt).replace(".pt", "").replace(".pkl", "")
        if args.ckpt else "random-init"
    )
    folder_name = (
        f"{model_string_name}-{ckpt_string_name}-size-{args.resolution}-vae-jt-"
        f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}-{args.guidance_high}-{args.cls_cfg_scale}"
    )
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    mpi_barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * world_size
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(var_numel(p) for p in model.parameters()):,}")
        print(f"projector Parameters: {sum(var_numel(p) for p in model.projectors.parameters()):,}")
    assert total_samples % world_size == 0
    samples_needed_this_gpu = int(total_samples // world_size)
    assert samples_needed_this_gpu % n == 0
    iterations = int(samples_needed_this_gpu // n)

    pbar = tqdm(range(iterations)) if rank == 0 else range(iterations)
    total = 0
    for _ in pbar:
        with jt.no_grad():
            z = jt.randn((n, model.in_channels, latent_size, latent_size), dtype="float32")
            y = jt.randint(0, args.num_classes, shape=(n,), dtype="int32")
            cls_z = jt.randn((n, args.cls), dtype="float32")

            if args.mode != "sde":
                raise NotImplementedError("Only sde mode is supported.")

            samples = euler_maruyama_sampler_jittor(
                model=model,
                latents=z,
                y=y,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                guidance_low=args.guidance_low,
                guidance_high=args.guidance_high,
                path_type=args.path_type,
                cls_latents=cls_z,
                cls_cfg_scale=args.cls_cfg_scale,
            ).astype("float32")

            latents_scale = jt.array([0.18215, 0.18215, 0.18215, 0.18215], dtype="float32").reshape((1, 4, 1, 1))
            latents_bias = -jt.array([0.0, 0.0, 0.0, 0.0], dtype="float32").reshape((1, 4, 1, 1))
            decoded = vae.decode((samples - latents_bias) / latents_scale)
            decoded = (decoded + 1.0) / 2.0
            decoded = jt.clamp(255.0 * decoded, 0, 255).permute(0, 2, 3, 1).numpy().astype(np.uint8)

        for i, sample in enumerate(decoded):
            index = i * world_size + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

        total += global_batch_size

    mpi_barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    mpi_barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to a Jittor SiT checkpoint.")
    parser.add_argument("--vae-ckpt", type=str, default=None, help="Path to a Jittor VAE checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--model", type=str, default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--mode", type=str, default="sde")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--cls-cfg-scale", type=float, default=1.5)
    parser.add_argument("--projector-embed-dims", type=str, default="768,1024")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--cls", default=768, type=int)
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    main(parser.parse_args())
