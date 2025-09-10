import json
import os
from typing import List, Dict, Any, Tuple

from PIL import Image, ImageDraw, ImageFont


WORKSPACE_DIR = "/Users/bin.wang/Projects/jets_clobotics"
RESPONSE_JSON_PATH = os.path.join(WORKSPACE_DIR, "response.json")
INPUT_IMAGE_PATH = os.path.join(WORKSPACE_DIR, "shelfphoto.jpg")
OUTPUT_IMAGE_PATH = os.path.join(WORKSPACE_DIR, "shelfphoto_annotated.jpg")
FILTER_LABEL = "urpc_e928d55f-b28d-44e0-9477-a166c72a0190"


def _extract_products(response_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = response_json.get("data") or {}
    result = data.get("result") or {}
    products = result.get("products") or []
    return products


def _normalized_to_absolute_bbox(
    bbox: Dict[str, float], image_width: int, image_height: int
) -> Tuple[int, int, int, int]:
    cx = float(bbox.get("center_x", 0.0))
    # Incoming coordinates appear to use origin at bottom-left.
    # Flip Y to convert to top-left origin used by PIL.
    cy_norm = float(bbox.get("center_y", 0.0))
    cy = 1.0 - cy_norm
    w = float(bbox.get("width", 0.0))
    h = float(bbox.get("height", 0.0))

    x1 = (cx - w / 2.0) * image_width
    y1 = (cy - h / 2.0) * image_height
    x2 = (cx + w / 2.0) * image_width
    y2 = (cy + h / 2.0) * image_height

    # Clamp to image bounds
    x1 = max(0, min(image_width - 1, int(round(x1))))
    y1 = max(0, min(image_height - 1, int(round(y1))))
    x2 = max(0, min(image_width - 1, int(round(x2))))
    y2 = max(0, min(image_height - 1, int(round(y2))))
    return x1, y1, x2, y2


def _pick_label(product: Dict[str, Any]) -> str:
    # Server may spell it as "identfier"; handle robustly.
    ident = (
        product.get("identifier")
        or product.get("identfier")
        or product.get("other_id")
        or product.get("msid")
        or product.get("bbox_id")
        or "unknown"
    )
    conf = product.get("confidence")
    conf_str = f"{float(conf):.2f}" if conf is not None else "--"
    return f"{ident} ({conf_str})"


def _matches_label(product: Dict[str, Any], target: str) -> bool:
    if not target:
        return True
    candidates = [
        product.get("identifier"),
        product.get("identfier"),
        product.get("other_id"),
        product.get("msid"),
        product.get("bbox_id"),
    ]
    try:
        render = _pick_label(product)
        base = render.split(" (")[0]
        candidates.append(base)
    except Exception:
        pass
    return any(c == target for c in candidates if c)


def draw_detections(
    image_path: str,
    detections: List[Dict[str, Any]],
    output_path: str,
) -> None:
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")

    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    image_width, image_height = image.size

    # Simple color cycle
    colors = [
        (255, 69, 0),      # orange-red
        (50, 205, 50),     # lime green
        (65, 105, 225),    # royal blue
        (255, 215, 0),     # gold
        (199, 21, 133),    # medium violet red
        (0, 206, 209),     # dark turquoise
        (138, 43, 226),    # blue violet
    ]

    for idx, product in enumerate(detections):
        bbox = product.get("bounding_box") or {}
        x1, y1, x2, y2 = _normalized_to_absolute_bbox(bbox, image_width, image_height)

        color = colors[idx % len(colors)]

        # Rectangle border (thickness 3)
        for t in range(3):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color + (255,))

        label = _pick_label(product)

        # Measure text robustly across Pillow versions
        try:
            # Newer Pillow
            bbox_text = font.getbbox(label)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
        except Exception:
            try:
                # Older Pillow
                text_w, text_h = font.getsize(label)
            except Exception:
                # Fallback conservative estimate
                text_w = int(8 * len(label))
                text_h = 14
        pad_x, pad_y = 6, 4
        box_w = text_w + 2 * pad_x
        box_h = text_h + 2 * pad_y

        # Position above the box if space allows, else inside at the top
        label_x1 = x1
        label_y1 = y1 - box_h - 1 if y1 - box_h - 1 >= 0 else y1 + 1
        label_x2 = min(x1 + box_w, image_width - 1)
        label_y2 = min(label_y1 + box_h, image_height - 1)

        # Semi-transparent background for label
        draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill=color + (140,))
        draw.text((label_x1 + pad_x, label_y1 + pad_y), label, fill=(0, 0, 0, 255), font=font)

    # Save to disk (convert back to RGB to avoid PNG alpha if saving as JPG)
    output_ext = os.path.splitext(output_path)[1].lower()
    if output_ext in (".jpg", ".jpeg"):
        image.convert("RGB").save(output_path, quality=95)
    else:
        image.save(output_path)


def main() -> None:
    if not os.path.exists(INPUT_IMAGE_PATH):
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE_PATH}")
    if not os.path.exists(RESPONSE_JSON_PATH):
        raise FileNotFoundError(f"Response JSON not found: {RESPONSE_JSON_PATH}")

    with open(RESPONSE_JSON_PATH, "r") as f:
        response_json = json.load(f)

    products = _extract_products(response_json)

    if not products:
        print("No products found in response.json. Nothing to draw.")
        return


    # FILTER_LABEL = "urpc_e928d55f-b28d-44e0-9477-a166c72a0190"
    # if FILTER_LABEL:
    #     products = [p for p in products if _matches_label(p, FILTER_LABEL)]
    #     if not products:
    #         print(f"No products match label '{FILTER_LABEL}'. Nothing to draw.")
    #         return

    draw_detections(INPUT_IMAGE_PATH, products, OUTPUT_IMAGE_PATH)
    print(f"Wrote annotated image to: {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()


