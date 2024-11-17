import os
import subprocess
import threading
from pathlib import Path

BASE_DIR = Path(__file__).parent

SERVER_CONFIG = BASE_DIR / "server_config.yaml"
CLIENT_CONFIG = BASE_DIR / "client_config.yaml"
DATA_PATH = BASE_DIR / "runs/mnist"


def stream_output(process: subprocess.Popen, process_name: str) -> None:
    with process.stdout as stdout, process.stderr as stderr:
        for line in iter(stdout.readline, ""):
            line = line.strip()
            if line:
                print(line)
        for line in iter(stderr.readline, ""):
            line = line.strip()
            if line:
                print(line)


def run_server() -> subprocess.Popen:
    server_cmd = ["python", "-m", "nanofed.cli.server", str(SERVER_CONFIG)]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=BASE_DIR,
        env=env,
    )


def run_client(client_id: str) -> subprocess.Popen:
    client_cmd = [
        "python",
        "-m",
        "nanofed.cli.client",
        "--config",
        str(CLIENT_CONFIG),
        "--client_id",
        client_id,
        "--data_path",
        str(DATA_PATH),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        client_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=BASE_DIR,
        env=env,
    )


def main():
    print(
        f"Starting server with command: python -m nanofed.cli.server {SERVER_CONFIG}"  # noqa
    )
    server_process = run_server()

    server_thread = threading.Thread(
        target=stream_output, args=(server_process, "SERVER"), daemon=True
    )
    server_thread.start()

    client_processes = []
    client_threads = []
    for client_id in ["client1", "client2", "client3"]:
        print(
            f"Starting client {client_id} with python -m nanofed.cli.client --config {CLIENT_CONFIG} "  # noqa
            f"--client_id {client_id} --data_path {DATA_PATH}"
        )
        client_process = run_client(client_id)
        client_processes.append(client_process)

        client_thread = threading.Thread(
            target=stream_output,
            args=(client_process, client_id.upper()),
            daemon=True,
        )
        client_threads.append(client_thread)
        client_thread.start()

    print("Experiment running. Press Ctrl+C to stop.")

    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("Shutting down experiment...")
        for process in client_processes:
            process.terminate()
        server_process.terminate()

    for thread in client_threads:
        thread.join()
    server_thread.join()


if __name__ == "__main__":
    main()
