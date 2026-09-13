import argparse

from api_common import DEFAULT_CONFIG, base_url, request_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the service health endpoint.")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--ip", default=None, help="Override configured host.")
    parser.add_argument("--port", type=int, default=None, help="Override configured port.")
    args = parser.parse_args()

    url = f"{base_url(args.config, args.ip, args.port)}/api/health"
    print(f"[INFO] GET {url}")
    request_json(url, timeout=5.0)


if __name__ == "__main__":
    main()
