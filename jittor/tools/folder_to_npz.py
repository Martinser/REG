import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-npz", type=str, required=True)
    args = parser.parse_args()

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(exts)]
    files.sort()
    if len(files) == 0:
        raise RuntimeError(f"No images found in: {args.input_dir}")

    arrs = []
    for name in tqdm(files, desc="Packing images"):
        path = os.path.join(args.input_dir, name)
        img = Image.open(path).convert("RGB")
        arrs.append(np.asarray(img, dtype=np.uint8))

    data = np.stack(arrs, axis=0)
    np.savez(args.output_npz, arr_0=data)
    print(f"Saved {args.output_npz}, shape={data.shape}")


if __name__ == "__main__":
    main()
