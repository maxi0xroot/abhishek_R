"""
Prepare the portrait photo for clean ASCII conversion:
  1. isolate subject from background (using rembg if needed)
  2. crop and scale bust onto 920x920 canvas with ideal framing for the hero card
  3. tone-map person so darks (hair, glasses, beard) are rich and lights (white hoodie)
     map to subtle stippling without getting clipped to space
  4. enhance facial features and linework contrast
  5. output: source-prepped.png (grayscale), consumed by make_ascii_svg.py.

Usage:
    python scripts/prep_photo.py [input.png] [output.png]
"""
import os
import sys
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "source-photo.png")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "source-prepped.png")


def prep_image(inp_path, out_path):
    im = Image.open(inp_path)
    
    cache_cutout = os.path.join(HERE, "..", "user_cutout.png")
    if im.mode == "RGBA" and np.any(np.array(im)[:, :, 3] < 250):
        cutout = im
    elif os.path.exists(cache_cutout):
        cutout = Image.open(cache_cutout)
    else:
        print("Removing background with rembg...")
        from rembg import remove
        cutout = remove(im)
        cutout.save(cache_cutout)

    alpha = np.array(cutout)[:, :, 3]
    person_pts = np.where(alpha > 40)
    if len(person_pts[0]) == 0:
        raise ValueError("No person detected in image!")

    min_y, max_y = person_pts[0].min(), person_pts[0].max()

    # Head coordinates in cutout:
    # Hair top: y=363. Chin: y=530.
    head_top = min_y
    head_cx = 225.0

    target_size = 920
    scale = 2.17

    new_w = int(cutout.width * scale)
    new_h = int(cutout.height * scale)
    scaled_cutout = cutout.resize((new_w, new_h), Image.Resampling.LANCZOS)
    scaled_alpha = scaled_cutout.split()[3]

    # Target position on 920 canvas:
    paste_x = int(520 - head_cx * scale)
    paste_y = int(115 - head_top * scale)

    canvas = Image.new("RGBA", (target_size, target_size), (255, 255, 255, 255))
    canvas.paste(scaled_cutout, (paste_x, paste_y), mask=scaled_alpha)

    # Reconstruct mask
    mask_canvas = Image.new("L", (target_size, target_size), 0)
    mask_canvas.paste(scaled_alpha, (paste_x, paste_y))
    mask = np.array(mask_canvas) > 40

    gray = canvas.convert("L")
    gray_arr = np.array(gray).astype(np.float32)

    # Separate head and body for targeted tone mapping
    head_mask = mask.copy()
    head_mask[490:, :] = False

    body_mask = mask.copy()
    body_mask[:490, :] = False

    # Tone map head:
    # Darks (hair, glasses, beard) stay rich 0..20, while skin is lifted for clear separation
    head_pix = gray_arr[head_mask]
    h_min, h_max = head_pix.min(), head_pix.max()
    norm_head = np.clip((head_pix - h_min) / (h_max - h_min), 0.0, 1.0)
    
    # Piecewise curve for face
    threshold = 0.20
    above_diff = np.maximum(0.0, (norm_head - threshold) / (1.0 - threshold))
    tuned_head = np.where(
        norm_head < threshold,
        norm_head / threshold * 20.0,
        20.0 + np.power(above_diff, 0.70) * 210.0
    )
    gray_arr[head_mask] = tuned_head

    # Tone map body (white hoodie) so zipper and seams pop while fabric stays light
    body_pix = gray_arr[body_mask]
    b_min, b_max = body_pix.min(), body_pix.max()
    norm_body = np.clip((body_pix - b_min) / (b_max - b_min), 0.0, 1.0)
    tuned_body = np.power(norm_body, 0.85) * 190.0 + 20.0
    gray_arr[body_mask] = tuned_body

    # Clean pure white background
    gray_arr[~mask] = 255.0

    res = Image.fromarray(gray_arr.astype(np.uint8))
    # Unsharp mask to make spectacles, eyes, curls and beard super crisp
    res = res.filter(ImageFilter.UnsharpMask(radius=2.5, percent=220, threshold=1))
    res = ImageEnhance.Contrast(res).enhance(1.20)

    res.save(out_path)
    print(f"Wrote {out_path} ({res.size[0]}x{res.size[1]})")


if __name__ == "__main__":
    prep_image(INP, OUT)
