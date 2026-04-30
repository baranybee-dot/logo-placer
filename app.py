from flask import Flask, request, send_file, render_template, jsonify
from PIL import Image
import numpy as np
import io
import base64
import rembg
from rembg import new_session
from scipy.ndimage import label as scipy_label
from scipy.spatial import ConvexHull

# Lazily-loaded BiRefNet session (downloaded from HuggingFace on first use, ~100 MB)
_birefnet_session = None

def _get_birefnet_session():
    global _birefnet_session
    if _birefnet_session is None:
        print("Loading BiRefNet model (first run — downloading if needed)…")
        try:
            _birefnet_session = new_session("birefnet-general")
        except Exception as e:
            print(f"BiRefNet unavailable ({e}), falling back to u2net")
            _birefnet_session = new_session("u2net")
    return _birefnet_session

app = Flask(__name__)

BASE_IMAGE_PATH = "static/base.png"
PLACEHOLDER_IMAGE_PATH = "static/placeholder.png"

_placements = None


def _rot(points, angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    return points @ R.T


def min_bounding_rect(xs, ys):
    """Return (cx, cy, width, height, angle_deg) of the minimum-area bounding rectangle.

    angle_deg is normalised to [-45, 45]: the clockwise tilt of the rectangle's
    horizontal axis from the image horizontal.
    """
    pts = np.column_stack([xs.astype(float), ys.astype(float)])
    try:
        hull_pts = pts[ConvexHull(pts).vertices]
    except Exception:
        hull_pts = pts

    cycled = np.vstack([hull_pts, hull_pts[0]])
    edge_angles = np.degrees(np.arctan2(
        np.diff(cycled[:, 1]), np.diff(cycled[:, 0])
    ))

    best_area, result = float("inf"), {}
    for angle in edge_angles:
        rot = _rot(hull_pts, -angle)
        min_x, max_x = rot[:, 0].min(), rot[:, 0].max()
        min_y, max_y = rot[:, 1].min(), rot[:, 1].max()
        area = (max_x - min_x) * (max_y - min_y)
        if area < best_area:
            best_area = area
            cx_r, cy_r = (min_x + max_x) / 2, (min_y + max_y) / 2
            cx, cy = _rot(np.array([[cx_r, cy_r]]), angle)[0]
            result = dict(cx=float(cx), cy=float(cy),
                          width=float(max_x - min_x),
                          height=float(max_y - min_y),
                          angle=float(angle))

    # Normalise angle to [-45, 45] so the logo is never placed upside-down or
    # rotated 90° when the rectangle is nearly square.
    a = result["angle"]
    w, h = result["width"], result["height"]
    # Fold into (-90, 90]
    a = ((a + 180) % 360) - 180
    if a < -90:
        a += 180
    elif a > 90:
        a -= 180
    # Fold into [-45, 45]; swap w/h to keep width = extent along 'a' direction
    if a > 45:
        a -= 90
        w, h = h, w
    elif a < -45:
        a += 90
        w, h = h, w
    result["angle"] = a
    result["width"] = w
    result["height"] = h
    return result


def detect_placements():
    img = Image.open(PLACEHOLDER_IMAGE_PATH).convert("RGB")
    arr = np.array(img)
    r, g = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int)

    # Hot-pink detection: high red, low green
    mask = (r > 180) & (g < 90)

    labeled, n = scipy_label(mask)
    rects = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) > 400:
            rects.append(min_bounding_rect(xs, ys))

    if len(rects) != 3:
        raise RuntimeError(f"Expected 3 pink rectangles, found {len(rects)}.")

    # Leftmost cx = vest (black); of the right two, smaller cy = bag (white), larger cy = mug (black)
    rects.sort(key=lambda r: r["cx"])
    rects[0]["color"] = "black"  # vest
    right = sorted(rects[1:], key=lambda r: r["cy"])
    right[0]["color"] = "white"  # bag
    right[1]["color"] = "black"  # mug
    return [rects[0]] + right


def get_placements():
    global _placements
    if _placements is None:
        _placements = detect_placements()
    return _placements


def has_background(img):
    # LA (grayscale+alpha), PA (palette+alpha), RGBA all carry an alpha channel
    if img.mode in ("RGBA", "LA", "PA"):
        alpha = np.array(img.convert("RGBA"))[:, :, 3]
        # Only skip removal when >5% of pixels are meaningfully transparent
        return (alpha < 128).mean() < 0.05
    return True


def remove_background(img):
    rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]

    # Sample the entire image border to judge background uniformity.
    # Using the full border (not just corners) catches gradients that vary
    # across the edge but whose corners happen to look similar.
    border_px = np.vstack([
        rgb[0, :].reshape(-1, 3),
        rgb[-1, :].reshape(-1, 3),
        rgb[:, 0].reshape(-1, 3),
        rgb[:, -1].reshape(-1, 3),
    ])
    bg = np.median(border_px, axis=0)
    border_std = float(border_px.std())

    if border_std < 18:
        # ── Solid / near-solid background ─────────────────────────────────
        # Flood-fill from borders: only pixel clusters that physically touch
        # the image edge are classified as background. This correctly handles
        # enclosed spaces (letter counters, icon cutouts) regardless of whether
        # the logo is dark or light.
        tolerance = 35
        diff = np.abs(rgb.astype(np.int16) - bg).max(axis=2)
        similar = diff < tolerance

        labeled, _ = scipy_label(similar)

        border_labels = set()
        for edge in (labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]):
            border_labels.update(edge.tolist())
        border_labels.discard(0)

        bg_mask = np.zeros((h, w), dtype=bool)
        for lbl in border_labels:
            bg_mask |= labeled == lbl

        alpha_f = np.where(bg_mask, 0.0, np.clip(diff.astype(float) / tolerance, 0.0, 1.0))
        alpha = (alpha_f * 255).astype(np.uint8)
        return Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA")

    else:
        # ── Gradient / complex background ─────────────────────────────────
        # Use BiRefNet (via rembg) — far more accurate than the default u2net
        # model for logos with gradients, shadows, or non-uniform backgrounds.
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = rembg.remove(buf.getvalue(), session=_get_birefnet_session())
        return Image.open(io.BytesIO(result)).convert("RGBA")


def colorize(logo, color):
    arr = np.array(logo.convert("RGBA"), dtype=np.uint8)
    rgb = (0, 0, 0) if color == "black" else (255, 255, 255)
    arr[:, :, :3] = rgb
    return Image.fromarray(arr, "RGBA")


def place_logo(base, logo_nobg, p, size_factor=1.0):
    cx, cy = p["cx"], p["cy"]
    rw, rh = p["width"], p["height"]
    angle = p["angle"]

    colored = colorize(logo_nobg.copy(), p["color"])
    lw, lh = colored.size
    scale = min(rw / lw, rh / lh) * size_factor
    scaled = colored.resize(
        (max(1, int(lw * scale)), max(1, int(lh * scale))), Image.LANCZOS
    )

    rotated = scaled.rotate(-angle, expand=True, resample=Image.BICUBIC)

    paste_x = int(cx - rotated.width / 2)
    paste_y = int(cy - rotated.height / 2)
    base.paste(rotated, (paste_x, paste_y), rotated)


def composite(logo_nobg, size_factor):
    base = Image.open(BASE_IMAGE_PATH).convert("RGBA")
    for p in get_placements():
        place_logo(base, logo_nobg, p, size_factor)
    result = base.convert("RGB")
    buf = io.BytesIO()
    result.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    if "logo" not in request.files:
        return jsonify({"error": "No logo file"}), 400
    try:
        logo_img = Image.open(request.files["logo"].stream)
        logo_img.load()
    except Exception:
        return jsonify({"error": "Could not open image"}), 400
    try:
        # Background removal happens once — reused for all three sizes
        logo_nobg = remove_background(logo_img) if has_background(logo_img) else logo_img.convert("RGBA")

        return jsonify({
            "images": [
                {"label": "100%",      "filename": "logo-100.jpg", "data": composite(logo_nobg, 1.00)},
                {"label": "−15%",      "filename": "logo-85.jpg",  "data": composite(logo_nobg, 0.85)},
                {"label": "−30%",      "filename": "logo-70.jpg",  "data": composite(logo_nobg, 0.70)},
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Detecting placements …")
    for p in get_placements():
        print(f"  {p['color']:5s}  cx={p['cx']:.0f} cy={p['cy']:.0f}  "
              f"{p['width']:.0f}×{p['height']:.0f}  angle={p['angle']:.1f}°")
    app.run(debug=True, port=5001, use_reloader=False)
