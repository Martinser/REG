import argparse

import jittor as jt
import torch
from diffusers import AutoencoderKL


def main():
    parser = argparse.ArgumentParser(description="Convert diffusers SD-VAE weights to Jittor.")
    parser.add_argument(
        "--src",
        type=str,
        default="stabilityai/sd-vae-ft-ema",
        help="HuggingFace repo id or local diffusers checkpoint directory.",
    )
    parser.add_argument("--out", type=str, required=True, help="Output Jittor checkpoint path (.pkl).")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load local files (no network).",
    )
    args = parser.parse_args()

    vae = AutoencoderKL.from_pretrained(args.src, local_files_only=args.local_files_only)
    sd = vae.state_dict()

    # Keep only decode path needed by generate_jittor.py
    keep_prefix = ("post_quant_conv.", "decoder.")
    out_sd = {}
    for k, v in sd.items():
        if k.startswith(keep_prefix):
            out_sd[k] = jt.array(v.detach().cpu().numpy())

    jt.save(out_sd, args.out)
    print(f"Saved Jittor VAE checkpoint to {args.out}, params={len(out_sd)}")


if __name__ == "__main__":
    main()
