"""
STEP 1 STARTER SCRIPT (v2 - Fixed Concave Shapes & Crisp Edges):
PNG -> Corner Map -> Visual Stitch Preview & .PES File
================================================================
What was fixed in v2:
  1. Exact Edge Anchoring: Always places needle points right on the polygon border
     (staggering only interior points) so circles and star tips are crisp, not jagged.
  2. Monotone Chain Tracking (No Bridging!): When a shape is concave like a 5-point star,
     the scanline splits across two arms/legs. Instead of stitching straight across
     the empty gap, it finishes one arm/leg cleanly, then JUMPS to the next arm.
  3. Clean Contour Outline: Adds a perimeter running-stitch border around the detected
     corners so the final embroidery has sharp, clean edges.
"""

import math
import numpy as np
import cv2
from shapely.geometry import Polygon, LineString, MultiLineString
from shapely.affinity import rotate
import pyembroidery


def create_sample_logo_png(filepath: str = "1_input_logo.png", size: int = 400):
    """Creates a clean 2-color circle + star badge PNG so you can test immediately."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)  # White background

    # Color 1: Royal Blue Circle Badge (BGR for OpenCV)
    cv2.circle(img, (size // 2, size // 2), 140, (190, 90, 30), thickness=-1)

    # Color 2: Gold 5-Point Star inside the badge
    cx, cy = size // 2, size // 2
    outer_r, inner_r = 95, 42
    pts = []
    for i in range(10):
        r = outer_r if i % 2 == 0 else inner_r
        angle = -math.pi / 2 + i * math.pi / 5
        pts.append([int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))])
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], (20, 175, 235))

    cv2.imwrite(filepath, img)
    print(f"[1/4] Created test image: {filepath}")
    return filepath


def extract_color_layers_and_corners(image_path: str, num_colors: int = 2):
    """
    Uses K-Means color clustering to split the PNG into thread color layers,
    then detects the exact polygon corners of each shape.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    non_white_mask = np.any(img_rgb < 240, axis=-1)
    fg_pixels = img_rgb[non_white_mask].astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(
        fg_pixels, num_colors, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    centers = centers.astype(np.uint8)

    label_map = np.full((h, w), -1, dtype=np.int32)
    label_map[non_white_mask] = labels.flatten()

    corner_preview = img_bgr.copy()
    layers = []

    for c_idx in range(num_colors):
        mask = np.uint8(label_map == c_idx) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 150:
                continue

            # Detect corners around the shape
            epsilon = 0.008 * cv2.arcLength(cnt, True)
            corners = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)

            cv2.drawContours(corner_preview, [cnt], -1, (0, 220, 0), 2)
            for (x, y) in corners:
                cv2.circle(corner_preview, (int(x), int(y)), 4, (0, 0, 255), -1)

            rect = cv2.minAreaRect(cnt)
            (_, _), (rw, rh), angle_deg = rect
            if rw < rh:
                angle_deg += 90.0

            rgb_tuple = tuple(int(v) for v in centers[c_idx])
            layers.append({
                "corners_px": corners.tolist(),
                "rgb": rgb_tuple,
                "angle_deg": angle_deg % 180.0,
            })

    cv2.imwrite("2_detected_corners.png", corner_preview)
    print(f"[2/4] Saved corner & contour detection preview: 2_detected_corners.png ({len(layers)} thread layers)")
    return layers, (w, h)


def _subdivide_segment(p0, p1, stitch_len_mm: float, row_idx: int):
    """
    Subdivides a single scanline segment into needle points, ALWAYS anchoring
    both exact border endpoints (p0 and p1) and staggering only interior points.
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 0.2:
        return [p0]

    pts = [p0]
    stagger = (row_idx % 3) * (stitch_len_mm / 3.0)
    dist = stagger if stagger > 0.3 else stitch_len_mm

    while dist < length - 0.3:
        t = dist / length
        pts.append((p0[0] + t * dx, p0[1] + t * dy))
        dist += stitch_len_mm

    pts.append(p1)
    return pts


def fill_polygon_with_stitches(
    corners_px: list,
    angle_deg: float,
    mm_per_px: float = 0.25,
    spacing_mm: float = 0.40,
    stitch_len_mm: float = 3.5,
):
    """
    Slices a polygon into monotone stitch chains so concave shapes (like stars)
    never stitch across empty gaps. Returns a list of separate stitch chains:
    [ [(x,y), ...], [(x,y), ...], ... ]
    """
    if len(corners_px) < 3:
        return []

    pts_mm = [(x * mm_per_px, y * mm_per_px) for (x, y) in corners_px]
    poly = Polygon(pts_mm).buffer(0.15)  # +0.15mm pull compensation
    if poly.is_empty:
        return []

    rot_poly = rotate(poly, -angle_deg, origin="centroid", use_radians=False)
    minx, miny, maxx, maxy = rot_poly.bounds

    y = miny + spacing_mm / 2.0
    row = 0

    # Active monotone chains: each entry is {'last_x_range': (x0, x1), 'pts': [...]}
    active_chains = []
    finished_chains = []

    while y < maxy:
        ray = LineString([(minx - 1.0, y), (maxx + 1.0, y)])
        inter = rot_poly.intersection(ray)

        segs = []
        if isinstance(inter, LineString) and not inter.is_empty:
            segs = [inter]
        elif isinstance(inter, MultiLineString):
            segs = [g for g in inter.geoms if not g.is_empty]

        # Sort segments left-to-right
        segs.sort(key=lambda s: min(s.coords[0][0], s.coords[-1][0]))

        new_active = []
        used_chain_indices = set()

        for seg in segs:
            coords = list(seg.coords)
            x_min = min(coords[0][0], coords[-1][0])
            x_max = max(coords[0][0], coords[-1][0])

            # Find an existing active chain whose x-range overlaps this segment
            matched_idx = None
            for c_idx, chain in enumerate(active_chains):
                if c_idx in used_chain_indices:
                    continue
                cx0, cx1 = chain["last_x_range"]
                if not (x_max < cx0 - 0.5 or x_min > cx1 + 0.5):
                    matched_idx = c_idx
                    break

            if matched_idx is not None:
                chain = active_chains[matched_idx]
                used_chain_indices.add(matched_idx)
                chain_row = chain["row"]
            else:
                chain = {"last_x_range": (x_min, x_max), "pts": [], "row": 0}
                chain_row = 0

            p0, p1 = coords[0], coords[-1]
            if chain_row % 2 == 1:
                p0, p1 = p1, p0  # Zig-zag direction

            seg_pts = _subdivide_segment(p0, p1, stitch_len_mm, chain_row)
            chain["pts"].extend(seg_pts)
            chain["last_x_range"] = (x_min, x_max)
            chain["row"] = chain_row + 1
            new_active.append(chain)

        # Any active chain not continued on this row is finished
        for c_idx, chain in enumerate(active_chains):
            if c_idx not in used_chain_indices and chain["pts"]:
                finished_chains.append(chain["pts"])

        active_chains = new_active
        y += spacing_mm
        row += 1

    for chain in active_chains:
        if chain["pts"]:
            finished_chains.append(chain["pts"])

    # Rotate every chain back to original design orientation
    restored_chains = []
    for pts in finished_chains:
        if len(pts) >= 2:
            restored = rotate(LineString(pts), angle_deg, origin=poly.centroid, use_radians=False)
            restored_chains.append(list(restored.coords))
    return restored_chains


def generate_contour_border(corners_px: list, mm_per_px: float = 0.25, stitch_len_mm: float = 2.5):
    """Generates a clean running-stitch outline right along the detected corners."""
    if len(corners_px) < 3:
        return []
    pts_mm = [(x * mm_per_px, y * mm_per_px) for (x, y) in corners_px]
    pts_mm.append(pts_mm[0])  # Close the loop

    border_pts = []
    for i in range(len(pts_mm) - 1):
        seg = _subdivide_segment(pts_mm[i], pts_mm[i + 1], stitch_len_mm, i)
        if border_pts and seg:
            border_pts.extend(seg[1:])
        else:
            border_pts.extend(seg)
    return border_pts


def export_pes_and_visual_preview(layers: list, canvas_wh: tuple, target_width_mm: float = 100.0):
    """
    Generates underlay + monotone topstitch chains + crisp border for every layer,
    renders a visual stitch simulation image, and writes the final Brother .pes file.
    """
    w_px, h_px = canvas_wh
    mm_per_px = target_width_mm / float(w_px)

    preview_img = np.full((h_px, w_px, 3), 245, dtype=np.uint8)
    pattern = pyembroidery.EmbPattern()
    total_stitches = 0

    for idx, layer in enumerate(layers):
        r, g, b = layer["rgb"]
        thread = pyembroidery.EmbThread()
        thread.set_hex_color(f"{r:02x}{g:02x}{b:02x}")
        thread.description = f"Layer_{idx + 1}_RGB({r},{g},{b})"
        pattern.add_thread(thread)

        # 1. Underlay chains (sparse perpendicular)
        underlay_chains = fill_polygon_with_stitches(
            layer["corners_px"], layer["angle_deg"] + 90.0, mm_per_px, spacing_mm=2.4, stitch_len_mm=4.0
        )
        # 2. Topstitch chains (dense fill)
        topstitch_chains = fill_polygon_with_stitches(
            layer["corners_px"], layer["angle_deg"], mm_per_px, spacing_mm=0.40, stitch_len_mm=3.5
        )
        # 3. Crisp corner-to-corner outline border
        border_pts = generate_contour_border(layer["corners_px"], mm_per_px, stitch_len_mm=2.5)

        all_chains = underlay_chains + topstitch_chains + ([border_pts] if border_pts else [])

        for chain_pts in all_chains:
            if len(chain_pts) < 2:
                continue

            # JUMP cleanly to start of each disconnected chain (no ugly bridging stitch!)
            sx0, sy0 = chain_pts[0][0] * 10.0, chain_pts[0][1] * 10.0
            pattern.add_stitch_absolute(pyembroidery.JUMP, sx0, sy0)

            prev_px = None
            for (xmm, ymm) in chain_pts:
                pattern.add_stitch_absolute(pyembroidery.STITCH, xmm * 10.0, ymm * 10.0)
                total_stitches += 1

                cur_px = (int(xmm / mm_per_px), int(ymm / mm_per_px))
                if prev_px is not None:
                    cv2.line(preview_img, prev_px, cur_px, (b, g, r), 1, cv2.LINE_AA)
                cv2.circle(preview_img, cur_px, 1, (max(0, b - 50), max(0, g - 50), max(0, r - 50)), -1)
                prev_px = cur_px

        pattern.add_command(pyembroidery.TRIM)
        if idx < len(layers) - 1:
            pattern.add_command(pyembroidery.COLOR_CHANGE)

    pattern.add_command(pyembroidery.END)
    cv2.imwrite("3_visual_stitch_preview.png", preview_img)
    print(f"[3/4] Saved visual stitch preview: 3_visual_stitch_preview.png ({total_stitches} total stitches)")

    pes_filename = "4_output_embroidery.pes"
    pyembroidery.write_pes(pattern, pes_filename)
    print(f"[4/4] SUCCESS! Saved machine-ready Brother embroidery file: {pes_filename}")


if __name__ == "__main__":
    logo_path = create_sample_logo_png("1_input_logo.png")
    detected_layers, wh = extract_color_layers_and_corners(logo_path, num_colors=2)
    export_pes_and_visual_preview(detected_layers, wh, target_width_mm=100.0)   