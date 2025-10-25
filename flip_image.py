import argparse
from pathlib import Path
from PIL import Image, ExifTags
import sys

# Optional OpenCV import (only used if backend is opencv)
try:
    import cv2
    _cv2_available = True
except Exception:
    _cv2_available = False

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def flip_pillow(img: Image.Image, mode: str) -> Image.Image:
    mode = mode.lower()
    if mode == "horizontal":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif mode == "vertical":
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    elif mode == "both":
        # horizontal then vertical == rotate 180 or both flips
        return img.transpose(Image.ROTATE_180)
    else:
        raise ValueError("Unknown mode. Choose from horizontal|vertical|both")


def flip_opencv_image_bytes(img_path: Path, out_path: Path, mode: str):
    if not _cv2_available:
        raise RuntimeError("OpenCV backend requested but opencv-python is not installed.")
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image via OpenCV: {img_path}")
    if mode == "horizontal":
        flipped = cv2.flip(img, 1)
    elif mode == "vertical":
        flipped = cv2.flip(img, 0)
    elif mode == "both":
        flipped = cv2.flip(img, -1)
    else:
        raise ValueError("Unknown mode. Choose from horizontal|vertical|both")
    # Use imencode + tofile to preserve unicode paths on Windows
    ext = out_path.suffix.lower().lstrip(".")
    ok, enc = cv2.imencode(f".{ext}", flipped)
    if not ok:
        raise RuntimeError("Failed to encode flipped image")
    enc.tofile(str(out_path))


def save_pillow_with_exif(img: Image.Image, out_path: Path, original: Image.Image = None):
    """
    Save using Pillow; attempt to preserve orientation EXIF if present.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    # preserve format
    fmt = original.format if original and original.format else None
    if fmt:
        save_kwargs["format"] = fmt
    # preserve quality for JPEG
    if fmt and fmt.upper() in ("JPEG", "JPG"):
        save_kwargs.setdefault("quality", 95)
        save_kwargs.setdefault("subsampling", 0)
    # attempt to keep ICC/profile and exif if present
    try:
        info = original.info
        if "icc_profile" in info:
            save_kwargs["icc_profile"] = info["icc_profile"]
        if "exif" in info:
            save_kwargs["exif"] = info["exif"]
    except Exception:
        pass
    img.save(out_path, **save_kwargs)


def process_file(in_path: Path, out_path: Path, mode: str, overwrite: bool = False, backend: str = "pillow"):
    if not in_path.exists():
        print(f"[!] Input file not found: {in_path}")
        return False

    if not overwrite and out_path.exists():
        print(f"[!] Output file already exists (use --overwrite to replace): {out_path}")
        return False

    try:
        if backend == "opencv":
            if not _cv2_available:
                raise RuntimeError("opencv backend not available; install opencv-python or use pillow backend")
            # use OpenCV method that handles unicode paths safely
            import numpy as np  # local import to avoid requiring numpy unless opencv used
            img = cv2.imdecode(np.fromfile(str(in_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError("OpenCV failed to read image.")
            if mode == "horizontal":
                flipped = cv2.flip(img, 1)
            elif mode == "vertical":
                flipped = cv2.flip(img, 0)
            elif mode == "both":
                flipped = cv2.flip(img, -1)
            else:
                raise ValueError("Unknown flip mode")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ok, enc = cv2.imencode(out_path.suffix, flipped)
            if not ok:
                raise RuntimeError("OpenCV failed to encode output image")
            enc.tofile(str(out_path))
        else:
            # default pillow backend
            with Image.open(in_path) as im:
                flipped = flip_pillow(im, mode)
                save_pillow_with_exif(flipped, out_path, original=im)
        print(f"[+] Saved: {out_path}")
        return True
    except Exception as e:
        print(f"[!] Failed to process {in_path}: {e}")
        return False


def is_image_file(p: Path):
    return p.is_file() and p.suffix.lower() in SUPPORTED_EXTS


def batch_process_folder(in_folder: Path, out_folder: Path, mode: str, overwrite: bool, backend: str):
    if not in_folder.exists() or not in_folder.is_dir():
        print(f"[!] Input folder not found or not a directory: {in_folder}")
        return
    out_folder.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in in_folder.iterdir() if is_image_file(p)])
    if not files:
        print("[!] No supported image files found in input folder.")
        return
    for p in files:
        out_p = out_folder / p.name
        process_file(p, out_p, mode, overwrite=overwrite, backend=backend)


def parse_args():
    p = argparse.ArgumentParser(description="Flip images horizontally, vertically, or both.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Input image file")
    group.add_argument("--input-folder", type=str, help="Input folder (batch). All supported images will be processed.")
    p.add_argument("--output", type=str, help="Output file path (for single image). If not given, auto-generates by appending _flipped.")
    p.add_argument("--output-folder", type=str, help="Output folder for batch (required if using --input-folder).")
    p.add_argument("--mode", type=str, choices=["horizontal", "vertical", "both"], default="horizontal", help="Flip mode")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    p.add_argument("--backend", type=str, choices=["pillow", "opencv"], default="pillow", help="Backend to use (pillow is default)")
    return p.parse_args()


def main():
    args = parse_args()
    backend = args.backend.lower()
    if backend == "opencv" and not _cv2_available:
        print("[!] OpenCV backend selected but opencv-python not installed. Falling back to pillow.", file=sys.stderr)
        backend = "pillow"

    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            print(f"[!] Input file does not exist: {in_path}", file=sys.stderr)
            sys.exit(2)
        # determine output path
        if args.output:
            out_path = Path(args.output)
        else:
            suffix = in_path.suffix
            stem = in_path.stem + f"_flipped_{args.mode}"
            out_path = in_path.with_name(stem + suffix)
        success = process_file(in_path, out_path, args.mode, overwrite=args.overwrite, backend=backend)
        if not success:
            sys.exit(1)

    else:
        # batch folder mode
        in_folder = Path(args.input_folder)
        if not args.output_folder:
            print("[!] --output-folder is required when using --input-folder", file=sys.stderr)
            sys.exit(2)
        out_folder = Path(args.output_folder)
        batch_process_folder(in_folder, out_folder, args.mode, args.overwrite, backend)
    

if __name__ == "__main__":
    main()
