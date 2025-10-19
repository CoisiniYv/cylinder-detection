import os
import json
import argparse
import urllib.request
import urllib.error


def load_config(config_path):
    ip = "127.0.0.1"
    port = 9003
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            ip = cfg.get("host") or cfg.get("ip") or ip
            port = int(cfg.get("analyzerPort", port))
    except Exception as e:
        print(f"[WARN] 读取配置失败，使用默认 {ip}:{port}，错误: {e}")
    return ip, port


def main():
    parser = argparse.ArgumentParser(description="调用 /api/control/cancel 取消任务接口")
    parser.add_argument("-c", "--config", default=os.path.join(os.path.dirname(__file__), "..", "config.json"),
                        help="配置文件路径（默认：项目根目录下的 config.json）")
    parser.add_argument("--ip", default=None, help="覆盖配置文件中的 IP（可选）")
    parser.add_argument("--port", type=int, default=None, help="覆盖配置文件中的端口（可选）")
    args = parser.parse_args()

    ip, port = load_config(args.config)
    if args.ip:
        ip = args.ip
    if args.port:
        port = args.port

    url = f"http://{ip}:{port}/api/control/cancel"
    print(f"[INFO] GET {url}")

    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            print("[RESP]", data)
    except urllib.error.HTTPError as e:
        print(f"[ERROR] HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}")
    except Exception as e:
        print(f"[ERROR] 请求失败: {e}")


if __name__ == "__main__":
    main()