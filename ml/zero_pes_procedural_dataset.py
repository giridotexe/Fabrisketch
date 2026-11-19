"""
Fabrisketch High-Accuracy Procedural Dataset Generator (`ml/zero_pes_dataset.py` v2)
====================================================================================
Upgrades for High Accuracy:
  1. Balanced Class Generation: Guarantees every sample contains a rich mix of
     Tatami Fill (Class 1), Satin Columns/Borders (Class 2), and thin Running
     Stitches (Class 3) so the model never starves on thin lines.
  2. Logo Typography & Compound Badges: Renders real OpenCV text/monograms,
     polygons with concentric Satin borders, stars, shields, and Bezier-like curves.
  3. Exact Whole-Shape Class Labeling: Labels each generated primitive using its true
     stroke width rather than noisy post-hoc approximations.
"""

import os
import math
import random
import string
import numpy as np
import cv2
from PIL import Image, ImageFilter

CLASS_BG = 0
CLASS_TATAMI_FILL = 1
CLASS_SATIN_COLUMN = 2
CLASS_RUNNING_STITCH = 3


def extract_corners_and_angle(binary_mask: np.ndarray):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, np.zeros_like(binary_mask)

    largest_cnt = max(contours, key=cv2.contourArea)
    epsilon = 0.010 * cv2.arcLength(largest_cnt, True)
    approx_corners = cv2.approxPolyDP(largest_cnt, epsilon, True)

    rect = cv2.minAreaRect(largest_cnt)
    (_, _), (w, h), angle_deg = rect
    if w < h:
        angle_deg += 90.0
    angle_rad = math.radians(angle_deg % 180.0)

    corner_edge_map = np.zeros_like(binary_mask)
    cv2.drawContours(corner_edge_map, [largest_cnt], -1, 160, thickness=2)
    for pt in approx_corners:
        cx, cy = int(pt[0][0]), int(pt[0][1])
        cv2.circle(corner_edge_map, (cx, cy), radius=4, color=255, thickness=-1)

    return angle_rad, corner_edge_map


def draw_balanced_layer(canvas_size: int, target_class: int):
    """
    Draws a shape specifically tailored to `target_class` (1=Fill, 2=Satin, 3=Run)
    so every stitch class is equally represented during training.
    """
    mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    cx = random.randint(int(canvas_size * 0.2), int(canvas_size * 0.8))
    cy = random.randint(int(canvas_size * 0.2), int(canvas_size * 0.8))
    rgb = (random.randint(15, 235), random.randint(15, 235), random.randint(15, 235))

    if target_class == CLASS_TATAMI_FILL:
        kind = random.choice(["polygon", "star", "shield_rect", "circle"])
        if kind == "polygon":
            n = random.randint(3, 8)
            r0 = random.randint(45, 115)
            angs = sorted([random.uniform(0, 2 * math.pi) for _ in range(n)])
            pts = np.array([
                [int(cx + r0 * random.uniform(0.75, 1.15) * math.cos(a)),
                 int(cy + r0 * random.uniform(0.75, 1.15) * math.sin(a))]
                for a in angs
            ], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        elif kind == "star":
            spikes = random.randint(4, 7)
            outer_r = random.randint(55, 115)
            inner_r = int(outer_r * random.uniform(0.42, 0.60))
            rot0 = random.uniform(0, math.pi)
            pts = []
            for i in range(spikes * 2):
                r = outer_r if i % 2 == 0 else inner_r
                a = rot0 + i * math.pi / spikes
                pts.append([int(cx + r * math.cos(a)), int(cy + r * math.sin(a))])
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        elif kind == "shield_rect":
            w, h = random.randint(70, 150), random.randint(55, 130)
            box = cv2.boxPoints(((cx, cy), (w, h), random.uniform(0, 180))).astype(np.int32)
            cv2.fillPoly(mask, [box], 255)
        else:
            cv2.circle(mask, (cx, cy), random.randint(45, 100), 255, -1)

    elif target_class == CLASS_SATIN_COLUMN:
        kind = random.choice(["ring", "ribbon", "text_monogram", "polygon_border"])
        thickness = random.randint(10, 22)  # Medium satin column width
        if kind == "ring":
            cv2.circle(mask, (cx, cy), random.randint(40, 105), 255, thickness=thickness)
        elif kind == "ribbon":
            pts = np.array([
                [np.clip(cx + random.randint(-110, 110), 30, canvas_size - 30),
                 np.clip(cy + random.randint(-110, 110), 30, canvas_size - 30)]
                for _ in range(random.randint(3, 5))
            ], dtype=np.int32)
            cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)
        elif kind == "text_monogram":
            text = "".join(random.choices(string.ascii_uppercase, k=random.randint(1, 3)))
            font_scale = random.uniform(2.2, 3.8)
            cv2.putText(mask, text, (max(20, cx - 80), cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness=random.randint(8, 15))
        else:
            w, h = random.randint(70, 150), random.randint(60, 130)
            box = cv2.boxPoints(((cx, cy), (w, h), random.uniform(0, 180))).astype(np.int32)
            cv2.polylines(mask, [box], isClosed=True, color=255, thickness=thickness)

    elif target_class == CLASS_RUNNING_STITCH:
        kind = random.choice(["thin_polyline", "thin_circle", "fine_detail"])
        thickness = random.randint(2, 4)  # Crisp running stitch width
        if kind == "thin_polyline":
            pts = np.array([
                [np.clip(cx + random.randint(-130, 130), 20, canvas_size - 20),
                 np.clip(cy + random.randint(-130, 130), 20, canvas_size - 20)]
                for _ in range(random.randint(3, 6))
            ], dtype=np.int32)
            cv2.polylines(mask, [pts], isClosed=random.choice([True, False]), color=255, thickness=thickness)
        elif kind == "thin_circle":
            cv2.circle(mask, (cx, cy), random.randint(35, 115), 255, thickness=thickness)
        else:
            # Fine crosshatch / detail accent
            for _ in range(2):
                p1 = (np.clip(cx + random.randint(-90, 90), 20, canvas_size - 20),
                      np.clip(cy + random.randint(-90, 90), 20, canvas_size - 20))
                p2 = (np.clip(cx + random.randint(-90, 90), 20, canvas_size - 20),
                      np.clip(cy + random.randint(-90, 90), 20, canvas_size - 20))
                cv2.line(mask, p1, p2, 255, thickness=thickness)

    return mask, rgb


def degrade_to_realistic_png(clean_rgb: np.ndarray) -> np.ndarray:
    pil_img = Image.fromarray(clean_rgb)
    if random.random() < 0.70:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.3)))
    arr = np.array(pil_img, dtype=np.uint8)
    if random.random() < 0.65:
        q = random.randint(55, 90)
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
        arr = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return arr


def generate_procedural_dataset(output_dir: str, num_samples: int = 1000, canvas_size: int = 512):
    for sub in ("images", "masks", "angles", "corners_and_edges"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    for idx in range(num_samples):
        clean_rgb = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
        class_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        angle_field = np.zeros((2, canvas_size, canvas_size), dtype=np.float32)
        corner_map = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        # Draw in realistic embroidery order: Fill base -> Satin columns/borders -> Running details
        layer_schedule = [
            CLASS_TATAMI_FILL,
            random.choice([CLASS_TATAMI_FILL, CLASS_SATIN_COLUMN]),
            CLASS_SATIN_COLUMN,
            CLASS_RUNNING_STITCH,
            random.choice([CLASS_SATIN_COLUMN, CLASS_RUNNING_STITCH]),
        ]

        for target_cls in layer_schedule:
            layer_mask, rgb = draw_balanced_layer(canvas_size, target_cls)
            if not np.any(layer_mask > 0):
                continue
            angle_rad, layer_corners = extract_corners_and_angle(layer_mask)
            active = layer_mask > 0
            clean_rgb[active] = rgb
            class_mask[active] = target_cls
            angle_field[0][active] = math.cos(2.0 * angle_rad)
            angle_field[1][active] = math.sin(2.0 * angle_rad)
            corner_map = np.maximum(corner_map, layer_corners)

        noisy_png = degrade_to_realistic_png(clean_rgb)
        stem = f"proc_{idx:05d}"

        Image.fromarray(noisy_png).save(os.path.join(output_dir, "images", f"{stem}.png"))
        Image.fromarray(class_mask).save(os.path.join(output_dir, "masks", f"{stem}.png"))
        np.save(os.path.join(output_dir, "angles", f"{stem}.npy"), angle_field)
        Image.fromarray(corner_map).save(os.path.join(output_dir, "corners_and_edges", f"{stem}.png"))

        if (idx + 1) % 100 == 0:
            print(f"Generated {idx + 1}/{num_samples} balanced training pairs...")

    print(f"Done! Saved {num_samples} balanced training samples in '{output_dir}'.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./procedural_embroidery_dataset")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    generate_procedural_dataset(args.output_dir, num_samples=args.samples)