import argparse
from xscope.dashboard import run_dashboard


def main():
    parser = argparse.ArgumentParser(description="xscope metrics visualization dashboard")
    parser.add_argument(
        "--dir",
        type=str,
        default="metrics",
        help="Path to the metrics directory containing experiment runs (default: 'metrics')",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run dashboard server on (default: 8080)",
    )
    args = parser.parse_args()
    run_dashboard(metrics_dir=args.dir, port=args.port, reload=False)

if __name__ in {"__main__", "__mp_main__"}:
    main()

