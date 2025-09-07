import argparse
import csv
import os
import sys
import time
import base64
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np


@dataclass
class FrameMetrics:
    frame_index: int
    timestamp_sec: float
    laplacian_var: float
    tenengrad_mean: float
    contrast_std: float
    brightness_mean: float


def compute_laplacian_variance(gray_image: np.ndarray) -> float:
    """Return variance of Laplacian, a common focus measure.

    Higher values indicate sharper images with more high-frequency content.
    """
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    return float(laplacian.var())


def compute_tenengrad_mean(gray_image: np.ndarray) -> float:
    """Compute Tenengrad focus measure using Sobel gradients.

    We return the mean gradient magnitude to keep values stable across resolutions.
    """
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(sobel_x, sobel_y)
    return float(np.mean(magnitude))


def compute_contrast_std(gray_image: np.ndarray) -> float:
    """Standard deviation of intensity as a simple global contrast measure."""
    return float(gray_image.std())


def robust_zscores(values: List[float]) -> List[float]:
    """Compute robust z-scores using median and MAD (scaled by 1.4826).

    This is resilient to outliers and different scales across measures.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    # Avoid division by zero; if mad is ~0, fall back to std
    if mad < 1e-9:
        std = arr.std() or 1.0
        return list((arr - arr.mean()) / std)
    scaled_mad = 1.4826 * mad
    return list((arr - median) / scaled_mad)


def select_candidates(
    metrics: List[FrameMetrics],
    top_percent: float,
    min_gap_seconds: float,
    max_candidates: Optional[int] = None,
) -> List[int]:
    """Select candidate frame indices based on combined sharpness score.

    We compute robust z-scores for each metric and combine them with weights.
    Then we take the top percentile and perform temporal non-maximum suppression
    using a minimum time gap.
    """
    if not metrics:
        return []

    lap_vals = [m.laplacian_var for m in metrics]
    ten_vals = [m.tenengrad_mean for m in metrics]
    con_vals = [m.contrast_std for m in metrics]
    bri_vals = [m.brightness_mean for m in metrics]

    z_lap = robust_zscores(lap_vals)
    z_ten = robust_zscores(ten_vals)
    z_con = robust_zscores(con_vals)

    # Combine with weights favoring Laplacian/edges
    combined_scores = [
        0.7 * z_l + 0.25 * z_t + 0.05 * z_c for z_l, z_t, z_c in zip(z_lap, z_ten, z_con)
    ]

    # Filter out frames that are too dark/bright; they are less useful for OCR/recognition
    # Keep frames with brightness roughly in [25, 230] mean intensity
    brightness_ok = [25.0 <= b <= 230.0 for b in bri_vals]

    # Determine score threshold at top_percent percentile
    scores_arr = np.asarray([
        s if ok else -np.inf for s, ok in zip(combined_scores, brightness_ok)
    ])
    finite_scores = scores_arr[np.isfinite(scores_arr)]
    if finite_scores.size == 0:
        return []
    threshold = float(np.quantile(finite_scores, 1.0 - (top_percent / 100.0)))

    # Build list of candidates above threshold
    candidates: List[Tuple[int, float, float]] = []  # (idx, time, score)
    for idx, (m, s, ok) in enumerate(zip(metrics, combined_scores, brightness_ok)):
        if ok and s >= threshold:
            candidates.append((idx, m.timestamp_sec, s))

    # Sort by score descending for greedy NMS, then enforce min gap
    candidates.sort(key=lambda x: x[2], reverse=True)
    kept: List[Tuple[int, float, float]] = []
    for cand in candidates:
        _, t, _ = cand
        if all(abs(t - kt) >= min_gap_seconds for _, kt, _ in kept):
            kept.append(cand)
        if max_candidates is not None and len(kept) >= max_candidates:
            break

    # Return original frame indices (sorted by frame index for saving in order)
    kept.sort(key=lambda x: x[0])
    return [k[0] for k in kept]


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def read_video_metadata(cap: cv2.VideoCapture) -> Tuple[float, int]:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, total_frames


def iterate_frames(
    video_path: str,
    limit_seconds: float,
    resize_max_width: Optional[int] = 1280,
) -> Tuple[List[FrameMetrics], List[int], float]:
    """Read frames up to limit_seconds and compute per-frame metrics.

    Returns (metrics_list, frame_indices_processed, fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps, total_frames = read_video_metadata(cap)
    max_frames = total_frames
    if np.isfinite(fps) and fps > 0 and total_frames > 0:
        max_frames = min(total_frames, int(limit_seconds * fps) + 1)

    results: List[FrameMetrics] = []
    frame_indices: List[int] = []

    start_time = time.time()
    last_log_time = start_time
    for idx in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if timestamp_sec > limit_seconds:
            break

        # Optionally downscale for metric stability and speed
        if resize_max_width is not None and frame.shape[1] > resize_max_width:
            scale = resize_max_width / frame.shape[1]
            new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = compute_laplacian_variance(gray)
        ten_mean = compute_tenengrad_mean(gray)
        con_std = compute_contrast_std(gray)
        bri_mean = float(gray.mean())

        results.append(
            FrameMetrics(
                frame_index=idx,
                timestamp_sec=timestamp_sec,
                laplacian_var=lap_var,
                tenengrad_mean=ten_mean,
                contrast_std=con_std,
                brightness_mean=bri_mean,
            )
        )
        frame_indices.append(idx)

        now = time.time()
        if now - last_log_time > 2.5:
            elapsed = now - start_time
            processed = idx + 1
            fps_est = processed / elapsed if elapsed > 0 else 0.0
            pct = (processed / max_frames) * 100.0 if max_frames else 0.0
            eta = ((max_frames - processed) / fps_est) if fps_est > 0 else float("inf")
            print(
                f"Extracting: {processed}/{max_frames} ({pct:.1f}%), elapsed {elapsed:.1f}s, ~{fps_est:.1f} fps, ETA {eta:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            last_log_time = now

    cap.release()
    return results, frame_indices, fps


def save_selected_frames(
    video_path: str,
    output_dir: str,
    selected_frame_indices: List[int],
    jpeg_quality: int = 95,
) -> List[Tuple[int, str]]:
    """Seek and save selected frames as JPEG files. Returns list of (index, path)."""
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    saved: List[Tuple[int, str]] = []
    for idx in selected_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        out_path = os.path.join(output_dir, f"frame_{idx:07d}.jpg")
        cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        saved.append((idx, out_path))

    cap.release()
    return saved


def write_csv_summary(
    csv_path: str,
    metrics: List[FrameMetrics],
    selected_indices: List[int],
    saved_map: Optional[dict] = None,
) -> None:
    ensure_dir(os.path.dirname(csv_path) or ".")
    selected_set = set(selected_indices)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_index",
                "timestamp_sec",
                "laplacian_var",
                "tenengrad_mean",
                "contrast_std",
                "brightness_mean",
                "selected",
                "saved_path",
            ]
        )
        for m in metrics:
            saved_path = ""
            if saved_map and m.frame_index in saved_map:
                saved_path = saved_map[m.frame_index]
            writer.writerow(
                [
                    m.frame_index,
                    f"{m.timestamp_sec:.3f}",
                    f"{m.laplacian_var:.6f}",
                    f"{m.tenengrad_mean:.6f}",
                    f"{m.contrast_std:.6f}",
                    f"{m.brightness_mean:.3f}",
                    1 if m.frame_index in selected_set else 0,
                    saved_path,
                ]
            )


def encode_image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def extract_json_from_text(text: str) -> Any:
    """Extract JSON from a model response that may include prose or code fences."""
    if text is None:
        raise ValueError("Empty response text")
    # Remove common code fences
    fenced = re.search(r"```(?:json)?\n([\s\S]*?)\n```", text)
    if fenced:
        text = fenced.group(1)
    # Trim junk before first { and after last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def build_llm_instructions() -> str:
    return (
        "You are assisting with onboarding a grocery/convenience store catalog from shelf photos. "
        "Analyze the given frame image and extract unique SKUs visible. Use on-pack text and your world knowledge. "
        "Avoid listing multiple rows for identical items in the same frame; de-duplicate by brand+variant+size.\n\n"
        "Output STRICT JSON ONLY with the following schema: \n"
        "{\n"
        "  \"frame_index\": <int>,\n"
        "  \"frame_path\": <string>,\n"
        "  \"skus\": [\n"
        "    {\n"
        "      \"sku_index\": <int>,  // 1-based index within the frame for unique SKUs\n"
        "      \"name\": <string>,    // include brand, product name, size (e.g., 'Coca-Cola 12oz Can 12-Pack')\n"
        "      \"price_full\": <string or null>,      // price before discount, include currency symbol if visible\n"
        "      \"price_discounted\": <string or null>, // discounted price if visible\n"
        "      \"taxonomy\": <string or null>,         // e.g., 'Beverages > Soda', 'Snacks > Chips'\n"
        "      \"other_text\": <string>               // free-form text transcribed from packaging or shelf tags\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Guidelines:\n"
        "- If prices are partially visible, infer cautiously; prefer exact prices if legible.\n"
        "- If multiple price tags for same SKU (e.g., different sizes), choose the one most likely attached to the SKU in the image.\n"
        "- Use a compact taxonomy. If uncertain, provide a best-effort category.\n"
        "- Keep sku_index contiguous starting from 1 for each frame.\n"
        "- Do not include any explanation or markdown, only the JSON."
    )


def call_portkey_chat_with_image(
    image_path: str,
    frame_index: int,
    model: str,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    # Lazy import to avoid dependency for non-analysis runs
    try:
        import portkey_ai  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "portkey-ai package is required for analysis mode. Install with 'pip install -U portkey-ai'."
        ) from e

    api_key = os.environ.get("PORTKEY_API_KEY")
    virtual_key = os.environ.get("OPENAI_VIRTUAL_KEY")
    if not api_key or not virtual_key:
        raise EnvironmentError(
            "Missing PORTKEY_API_KEY or OPENAI_VIRTUAL_KEY environment variables."
        )

    if not base_url:
        base_url = os.environ.get(
            "PORTKEY_BASE_URL", "https://cybertron-service-gateway.doordash.team/v1"
        )

    client = portkey_ai.Portkey(
        base_url=base_url,
        api_key=api_key,
        virtual_key=virtual_key,
    )

    data_url = encode_image_to_data_url(image_path)
    instructions = build_llm_instructions()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instructions.replace("\n", "\n")},
                {"type": "image_url", "image_url": {"url": data_url}},
                {
                    "type": "text",
                    "text": json.dumps({
                        "frame_index": frame_index,
                        "frame_path": os.path.abspath(image_path),
                    }),
                },
            ],
        }
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    text = completion.choices[0].message.content
    return extract_json_from_text(text)


def analyze_frames_dir(
    frames_dir: str,
    output_csv: str,
    model: str,
    max_frames: Optional[int] = None,
    start_offset: int = 0,
    retry: int = 2,
) -> None:
    ensure_dir(os.path.dirname(output_csv) or ".")

    # Collect frame files
    all_files = [
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    # Sort by frame index if present in filename
    def frame_num(path: str) -> int:
        m = re.search(r"(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else 0

    all_files.sort(key=frame_num)
    original_total = len(all_files)
    if start_offset > 0:
        all_files = all_files[start_offset:]
    if max_frames is not None:
        all_files = all_files[: max_frames]
    target_total = len(all_files)

    print(
        f"Analysis setup: found {original_total} frame images, processing {target_total} starting at offset {start_offset}. Model={model}",
        file=sys.stderr,
        flush=True,
    )

    # Prepare CSV
    header = [
        "frame_index",
        "frame_path",
        "sku_index",
        "name",
        "price_full",
        "price_discounted",
        "taxonomy",
        "other_text",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        processed = 0
        success_frames = 0
        failed_frames = 0
        rows_written = 0
        start_time = time.time()
        last_log_time = start_time

        for i, img_path in enumerate(all_files):
            # Derive frame index from filename
            fi = frame_num(img_path)
            attempt = 0
            while True:
                try:
                    resp = call_portkey_chat_with_image(
                        image_path=img_path,
                        frame_index=fi,
                        model=model,
                    )
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > retry:
                        print(
                            f"Analyze error (permanent) for frame {fi} at {img_path}: {e}",
                            file=sys.stderr,
                            flush=True,
                        )
                        resp = None
                        break
                    time.sleep(min(2.0 * attempt, 10.0))

            processed += 1
            if not resp:
                failed_frames += 1
            else:
                success_frames += 1
                # Expected resp schema: { frame_index, frame_path, skus: [ {...} ] }
                skus = resp.get("skus", []) if isinstance(resp, dict) else []
                for sku in skus:
                    writer.writerow(
                        [
                            resp.get("frame_index", fi),
                            resp.get("frame_path", os.path.abspath(img_path)),
                            sku.get("sku_index", ""),
                            sku.get("name", ""),
                            sku.get("price_full", ""),
                            sku.get("price_discounted", ""),
                            sku.get("taxonomy", ""),
                            sku.get("other_text", ""),
                        ]
                    )
                    rows_written += 1

            # periodic progress logging
            now = time.time()
            if now - last_log_time > 2.0 or processed == target_total:
                elapsed = now - start_time
                rate = processed / elapsed if elapsed > 0 else 0.0
                remaining = target_total - processed
                eta = remaining / rate if rate > 0 else float("inf")
                pct = (processed / target_total) * 100.0 if target_total else 0.0
                print(
                    f"Analyzing: {processed}/{target_total} ({pct:.1f}%), successes={success_frames}, failures={failed_frames}, rows={rows_written}, elapsed {elapsed:.1f}s, ~{rate:.2f} fps, ETA {eta:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                last_log_time = now

        total_elapsed = time.time() - start_time
        print(
            f"Analysis complete: processed {processed}/{target_total}, successes={success_frames}, failures={failed_frames}, rows={rows_written}. Took {total_elapsed:.1f}s.",
            file=sys.stderr,
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract sharp (non-blurry) frames from the first N seconds of a video, "
            "using focus measures and temporal de-duplication. Also supports analyzing saved frames via LLM to extract SKU data."
        )
    )
    extract_grp = parser.add_argument_group("Frame extraction")
    extract_grp.add_argument(
        "--video",
        required=False,
        help="Absolute path to the input video (.mov/.mp4/etc).",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.getcwd(), "outputs"),
        help="Directory to save outputs (frames and CSV).",
    )
    extract_grp.add_argument(
        "--limit-seconds",
        type=float,
        default=120.0,
        help="Process only the first N seconds of the video (default: 120).",
    )
    extract_grp.add_argument(
        "--min-gap",
        type=float,
        default=0.5,
        help="Minimum time gap in seconds between saved frames (default: 0.5).",
    )
    extract_grp.add_argument(
        "--top-percent",
        type=float,
        default=15.0,
        help=(
            "Select frames in the top X percent sharpness before gap filtering (default: 15)."
        ),
    )
    extract_grp.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional hard cap on number of frames to save.",
    )
    extract_grp.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved frames (0-100).",
    )
    extract_grp.add_argument(
        "--resize-max-width",
        type=int,
        default=1280,
        help="Resize frames for metric computation to this max width (speeds up processing).",
    )

    analyze_grp = parser.add_argument_group("LLM analysis")
    analyze_grp.add_argument(
        "--analyze-frames-dir",
        type=str,
        required=False,
        help="Directory containing sharp frames to analyze (e.g., outputs/.../frames)",
    )
    analyze_grp.add_argument(
        "--analysis-csv",
        type=str,
        required=False,
        help="Path to write SKU-level CSV (default: <frames_dir>/../sku_analysis.csv)",
    )
    analyze_grp.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="Model name via Portkey (default from OPENAI_MODEL or gpt-4o-mini)",
    )
    analyze_grp.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Analyze at most this many frames (useful for quick tests).",
    )
    analyze_grp.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Skip this many frames at the beginning of the directory.",
    )

    args = parser.parse_args()

    # Branch: analysis mode
    if args.analyze_frames_dir:
        frames_dir = os.path.abspath(args.analyze_frames_dir)
        if not os.path.isdir(frames_dir):
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        output_csv = (
            os.path.abspath(args.analysis_csv)
            if args.analysis_csv
            else os.path.join(os.path.dirname(frames_dir), "sku_analysis.csv")
        )
        print(f"Analyzing frames in: {frames_dir}")
        print(f"Writing SKU CSV to: {output_csv}")
        analyze_frames_dir(
            frames_dir=frames_dir,
            output_csv=output_csv,
            model=args.model,
            max_frames=args.max_frames,
            start_offset=args.start_offset,
        )
        return

    # Default: extraction mode (requires --video)
    if not args.video:
        parser.error("Either provide --video for extraction or --analyze-frames-dir for analysis.")

    video_path = os.path.abspath(args.video)
    out_dir_root = os.path.abspath(args.out_dir)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Create a run-specific subdirectory
    basename = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir_root, f"{basename}_first{int(args.limit_seconds)}s_{timestamp}")
    frames_dir = os.path.join(run_dir, "frames")
    ensure_dir(frames_dir)

    print(f"Reading video: {video_path}")
    print(f"Output directory: {run_dir}")

    extract_start = time.time()
    metrics, processed_indices, fps = iterate_frames(
        video_path=video_path,
        limit_seconds=float(args.limit_seconds),
        resize_max_width=int(args.resize_max_width) if args.resize_max_width else None,
    )
    if not metrics:
        print("No frames processed.", file=sys.stderr)
        return

    print(
        f"Computed metrics for {len(metrics)} frames at ~{fps:.2f} fps. Selecting candidates...",
        flush=True,
    )

    selected_indices_relative = select_candidates(
        metrics=metrics,
        top_percent=float(args.top_percent),
        min_gap_seconds=float(args.min_gap),
        max_candidates=args.max_candidates,
    )

    # Map relative indices (0..len(metrics)-1) to absolute frame indices
    selected_frame_numbers = [metrics[i].frame_index for i in selected_indices_relative]

    saved = save_selected_frames(
        video_path=video_path,
        output_dir=frames_dir,
        selected_frame_indices=selected_frame_numbers,
        jpeg_quality=int(args.jpeg_quality),
    )

    saved_map = {idx: path for idx, path in saved}
    csv_path = os.path.join(run_dir, "frame_metrics.csv")
    write_csv_summary(csv_path, metrics, selected_frame_numbers, saved_map)

    extract_elapsed = time.time() - extract_start
    print(
        f"Extraction complete: processed={len(metrics)}, selected={len(selected_frame_numbers)}, saved={len(saved)}. Took {extract_elapsed:.1f}s. Frames dir: {frames_dir}. CSV: {csv_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()


