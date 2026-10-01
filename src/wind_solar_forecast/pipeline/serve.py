"""Service launcher utility for the FastAPI inference API and Streamlit dashboard."""

import argparse
import subprocess
import sys


def start_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Starts FastAPI service using uvicorn."""
    import uvicorn
    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


def start_dashboard(port: int = 8501) -> None:
    """Launches Streamlit analytics dashboard."""
    cmd = [sys.executable, "-m", "streamlit", "run", "dashboards/app.py", "--server.port", str(port)]
    subprocess.run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve API or Dashboard.")
    parser.add_argument("mode", choices=["api", "dashboard"], default="api", nargs="?")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.mode == "api":
        start_api(port=args.port)
    else:
        start_dashboard(port=args.port or 8501)


if __name__ == "__main__":
    main()
