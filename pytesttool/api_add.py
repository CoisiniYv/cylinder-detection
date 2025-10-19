import os
import json
import argparse
import urllib.request
import urllib.error


def load_config(config_path):
    ip = "127.0.0.1"
    port = 9003
    model_dir = None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            ip = cfg.get("host") or cfg.get("ip") or ip
            port = int(cfg.get("analyzerPort", port))
            model_dir = cfg.get("modelDir")
    except Exception as e:
        print(f"[WARN] 读取配置失败，使用默认 {ip}:{port}，错误: {e}")
    return ip, port, model_dir


def build_payload(args, model_dir):
    # 默认推理参数，可根据需要调整或通过命令行覆盖
    trt_engine_file = args.trt_engine_file
    if not trt_engine_file and model_dir:
        # 尝试在 modelDir 中找一个 .engine 文件作为默认
        try:
            for name in os.listdir(model_dir):
                if name.lower().endswith('.engine'):
                    trt_engine_file = os.path.join(model_dir, name)
                    break
        except Exception:
            pass
    if not trt_engine_file:
        trt_engine_file = ""  # 允许为空，由服务端按默认处理或报错

    payload = {
        # 模型与图像预处理
        "trt_engine_file": trt_engine_file,
        "enable_swap_rb": True,
        # ROI（示例默认值，可通过命令行覆盖）
        "crop_x": args.crop_x,
        "crop_y": args.crop_y,
        "crop_width": args.crop_w,
        "crop_height": args.crop_h,
        # 线与条纹参数（服务端字段名）
        "line_x1": args.line_x1,
        "line_y1": args.line_y1,
        "line_x2": args.line_x2,
        "line_y2": args.line_y2,
        "stripe_radius": args.stripe_radius,
        # 条纹去除/去噪
        "filter_width": args.filter_width,
        "attenuation_factor": args.attenuation_factor,
        "target_angle": args.target_angle,
        "angle_tolerance": args.angle_tolerance,
        "enable_denoising": args.enable_denoising,
        "denoise_h": args.denoise_h,
        "denoise_hColor": args.denoise_hColor,
        "denoise_search_window": args.denoise_search_window,
        "denoise_template_window": args.denoise_template_window,
        # 切片相关
        "save_slices": args.save_slices,
        "slices_save_dir": args.slices_dir,
        "slice_width": args.slice_w,
        "slice_height": args.slice_h,
        "slice_distance": args.slice_distance,
        # 阈值（与服务端字段一致）
        "nms_threshold": args.nms_thresh,
        "conf_threshold": args.conf_thresh,
        # 输出与标签
        "result_image_path": args.result_img,
        "labels": args.labels,
        # 设备与采集节奏
        "device_id": args.device_id,
        "delay_ms": args.delay_ms,
    }
    return payload


def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            return body, None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}"
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description="调用 /api/control/add 下发检测任务（POST JSON）")
    parser.add_argument("-c", "--config", default=os.path.join(os.path.dirname(__file__), "..", "config.json"),
                        help="配置文件路径（默认：项目根目录下的 config.json）")
    parser.add_argument("--ip", default=None, help="覆盖配置文件中的 IP（可选）")
    parser.add_argument("--port", type=int, default=None, help="覆盖配置文件中的端口（可选）")

    # 设备与采集节奏
    parser.add_argument("--device_id", default="CAM-001", help="设备 ID")
    parser.add_argument("--delay_ms", type=int, default=200, help="采集延时（毫秒）")

    # 模型文件
    parser.add_argument("--trt_engine_file", default=None, help="TensorRT 引擎文件路径（默认：从 modelDir 自动寻找 .engine）")

    # ROI 参数
    parser.add_argument("--crop_x", type=int, default=0, help="ROI 左上角 X")
    parser.add_argument("--crop_y", type=int, default=57, help="ROI 左上角 Y")
    parser.add_argument("--crop_w", type=int, default=2856, help="ROI 宽度")
    parser.add_argument("--crop_h", type=int, default=2326, help="ROI 高度")

    # 线与条纹参数（对应服务端字段）
    parser.add_argument("--line_x1", type=int, default=2525, help="线起点 X1")
    parser.add_argument("--line_y1", type=int, default=1216, help="线起点 Y1")
    parser.add_argument("--line_x2", type=int, default=2525, help="线终点 X2")
    parser.add_argument("--line_y2", type=int, default=1216, help="线终点 Y2")
    parser.add_argument("--stripe_radius", type=int, default=250, help="条纹影响半径")

    # 条纹去除/去噪参数
    parser.add_argument("--filter_width", type=int, default=10, help="滤波窗口宽度")
    parser.add_argument("--attenuation_factor", type=float, default=0.0001, help="衰减系数")
    parser.add_argument("--target_angle", type=int, default=90, help="目标角度")
    parser.add_argument("--angle_tolerance", type=int, default=10, help="角度容差")
    parser.add_argument("--enable_denoising", type=int, default=1, help="启用去噪")
    parser.add_argument("--denoise_h", type=float, default=8.0, help="去噪参数 h")
    parser.add_argument("--denoise_hColor", type=float, default=8.0, help="去噪参数 hColor")
    parser.add_argument("--denoise_search_window", type=int, default=17, help="去噪搜索窗口")
    parser.add_argument("--denoise_template_window", type=int, default=11, help="去噪模板窗口")

    # 切片相关
    parser.add_argument("--slice_w", type=int, default=640, help="切片宽度")
    parser.add_argument("--slice_h", type=int, default=640, help="切片高度")
    parser.add_argument("--slice_distance", type=int, default=40, help="切片步进距离")
    parser.add_argument("--save_slices", type=int, default=0,help="保存切片")
    parser.add_argument("--slices_dir", default=os.path.join(os.path.dirname(__file__), "..", "output", "slices"),
                        help="切片保存目录")

    # 阈值（字段必须与服务端一致）
    parser.add_argument("--nms_thresh", dest="nms_thresh", type=float, default=0.45, help="NMS 阈值")
    parser.add_argument("--conf_thresh", dest="conf_thresh", type=float, default=0.20, help="置信度阈值")

    # 输出与标签
    parser.add_argument("--result_img", default=os.path.join(os.path.dirname(__file__), "..", "output"),
                        help="结果可视化图片保存目录或文件路径；若指定目录，服务端将自动生成不重复的文件名")
    parser.add_argument("--labels", nargs="*", default=[ "hh-Y","class_1","ox-Y","class_3","class_4","class_5", "class_6"], help="标签列表，例如：--labels good bad")

    args = parser.parse_args()

    ip, port, model_dir = load_config(args.config)
    if args.ip:
        ip = args.ip
    if args.port:
        port = args.port

    url = f"http://{ip}:{port}/api/control/add"
    payload = build_payload(args, model_dir)

    print("[INFO] POST", url)
    print("[INFO] Payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    body, err = post_json(url, payload)
    if err:
        print("[ERROR]", err)
    else:
        print("[RESP]", body)


if __name__ == "__main__":
    main()