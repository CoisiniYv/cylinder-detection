import argparse
import json

from api_common import DEFAULT_CONFIG, base_url, request_json


def parse_classes(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_payload(args: argparse.Namespace) -> dict:
    return {
        "model_name": args.model_name,
        "device_id": args.device_id,
        "gpu_device": args.gpu_device,
        "delay_ms": args.delay_ms,
        "qw_index": args.qw_index,
        "enable_four_side_crop": args.enable_four_side_crop,
        "crop_x": args.crop_x,
        "crop_y": args.crop_y,
        "crop_width": args.crop_width,
        "crop_height": args.crop_height,
        "is_qw": args.is_qw,
        "circle_x1": args.circle_x1,
        "circle_y1": args.circle_y1,
        "circle_x2": args.circle_x2,
        "circle_y2": args.circle_y2,
        "radius": args.radius,
        "enable_fourier_transform": args.enable_fourier_transform,
        "filter_width": args.filter_width,
        "attenuation_factor": args.attenuation_factor,
        "target_angle": args.target_angle,
        "angle_tolerance": args.angle_tolerance,
        "enable_denoising": args.enable_denoising,
        "denoise_h": args.denoise_h,
        "denoise_hColor": args.denoise_h_color,
        "denoise_search_window": args.denoise_search_window,
        "denoise_template_window": args.denoise_template_window,
        "slice_width": args.slice_width,
        "slice_height": args.slice_height,
        "slice_distance": args.slice_distance,
        "nms_threshold": args.nms_threshold,
        "conf_threshold": args.conf_threshold,
        "allowed_classes": parse_classes(args.allowed_classes),
        "pix_to_mm": args.pix_to_mm,
        "min_area_mm2": args.min_area_mm2,
        "min_diameter_mm": args.min_diameter_mm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a multi-face detection task.")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--ip", default=None, help="Override configured host.")
    parser.add_argument("--port", type=int, default=None, help="Override configured port.")

    parser.add_argument("--model-name", required=True, help="Model path relative to modelDir, or absolute path.")
    parser.add_argument("--device-id", default="CAM-001")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument("--qw-index", type=int, default=0)

    parser.add_argument("--enable-four-side-crop", action="store_true")
    parser.add_argument("--crop-x", type=int, default=0)
    parser.add_argument("--crop-y", type=int, default=0)
    parser.add_argument("--crop-width", type=int, default=0)
    parser.add_argument("--crop-height", type=int, default=0)

    parser.add_argument("--is-qw", action="store_true")
    parser.add_argument("--circle-x1", type=int, default=0)
    parser.add_argument("--circle-y1", type=int, default=0)
    parser.add_argument("--circle-x2", type=int, default=0)
    parser.add_argument("--circle-y2", type=int, default=0)
    parser.add_argument("--radius", type=int, default=0)

    parser.add_argument("--enable-fourier-transform", action="store_true")
    parser.add_argument("--filter-width", type=int, default=10)
    parser.add_argument("--attenuation-factor", type=float, default=0.0001)
    parser.add_argument("--target-angle", type=int, default=90)
    parser.add_argument("--angle-tolerance", type=int, default=10)
    parser.add_argument("--enable-denoising", action="store_true")
    parser.add_argument("--denoise-h", type=float, default=5.0)
    parser.add_argument("--denoise-h-color", type=float, default=8.0)
    parser.add_argument("--denoise-search-window", type=int, default=17)
    parser.add_argument("--denoise-template-window", type=int, default=11)

    parser.add_argument("--slice-width", type=int, default=640)
    parser.add_argument("--slice-height", type=int, default=640)
    parser.add_argument("--slice-distance", type=int, default=40)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--conf-threshold", type=float, default=0.20)
    parser.add_argument("--allowed-classes", default="0,2", help="Comma-separated class ids; empty means all classes.")

    parser.add_argument("--pix-to-mm", type=float, default=0.021)
    parser.add_argument("--min-area-mm2", type=float, default=0.0)
    parser.add_argument("--min-diameter-mm", type=float, default=0.0)
    args = parser.parse_args()

    url = f"{base_url(args.config, args.ip, args.port)}/api/control/add"
    payload = build_payload(args)
    print(f"[INFO] POST {url}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    request_json(url, method="POST", payload=payload, timeout=15.0)


if __name__ == "__main__":
    main()
