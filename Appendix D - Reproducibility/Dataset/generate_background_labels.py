
#Generate .txt label files for a folder of background images (no bounding boxes).
# python generate_background_labels.py /path/to/images /path/to/labels_output
#images_dir:  Path to the folder containing background images.
#labels_out:  Path to the folder where .txt label files will be written.

import sys
import os
from pathlib import Path
import argparse

def is_image_file(p: Path) -> bool:
    if not p.is_file():
        return False
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def ensure_out_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

def write_label_file(label_path: Path, mode: str):
    # mode: "empty" or "zero"
    if mode == "empty":
        label_path.write_text("", encoding="utf-8")
    elif mode == "zero":
        label_path.write_text("0", encoding="utf-8")
    else:
        # default to empty
        label_path.write_text("", encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", type=str)
    parser.add_argument("labels_out", type=str)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--empty-file", action="store_true")
    group.add_argument("--zero-line", action="store_true")

    args = parser.parse_args()

    images_dir = Path(args.images_dir).resolve()
    labels_out = Path(args.labels_out).resolve()

    mode = "empty"
    if args.zero_line:
        mode = "zero"
    elif args.empty_file:
        mode = "empty"

    if not images_dir.is_dir():
        print(f"Error: images_dir '{images_dir}' is not a directory.", file=sys.stderr)
        sys.exit(2)

    ensure_out_dir(labels_out)

    # Collect image files and produce corresponding label files
    image_files = sorted([p for p in images_dir.iterdir() if is_image_file(p)])

    if not image_files:
        print(f"No images found  {images_dir}.", file=sys.stderr)
        sys.exit(1)

    for img_path in image_files:
        base = img_path.stem
        label_path = labels_out / f"{base}.txt"
        write_label_file(label_path, mode)

    print(f"Generated {len(image_files)} label files in '{labels_out}'.")
    if mode == "empty":
        print("Label files are empty with no bounding boxes")
    elif mode == "zero":
        print("Label files contain a '0' to indicate zero objects.")

if __name__ == "__main__":
    main()