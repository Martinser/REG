import argparse
from collections import OrderedDict

import jittor as jt
import numpy as np
import torch


def convert_obj(obj):
    if isinstance(obj, torch.Tensor):
        return jt.array(obj.detach().cpu().numpy())
    if isinstance(obj, (dict, OrderedDict)):
        return {k: convert_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_obj(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jt.array(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint (.pt) to Jittor checkpoint.")
    parser.add_argument("--pt", type=str, required=True, help="Path to PyTorch checkpoint.")
    parser.add_argument("--out", type=str, required=True, help="Path to output Jittor checkpoint.")
    parser.add_argument(
        "--state-key",
        type=str,
        default="ema",
        help="If key exists in checkpoint dict, convert checkpoint[state_key] as main weights.",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.pt, map_location="cpu")
    payload = ckpt
    if isinstance(ckpt, dict) and args.state_key in ckpt:
        payload = ckpt[args.state_key]
        converted = {"ema": convert_obj(payload)}
    else:
        converted = convert_obj(payload)

    jt.save(converted, args.out)
    if isinstance(converted, dict):
        print(f"Saved Jittor checkpoint to {args.out}. keys={list(converted.keys())[:5]}")
    else:
        print(f"Saved Jittor checkpoint to {args.out}.")


if __name__ == "__main__":
    main()
