import subprocess
import time
import sys
import argparse
import os

def run_experiment(strategy_name, num_clients, num_rounds):
    utf8_env = os.environ.copy()
    utf8_env["PYTHONIOENCODING"] = "utf-8"
    utf8_env["PYTHONUTF8"]       = "1"

    print(f"\n{'='*52}")
    print(f"  LAUNCHING: {strategy_name.upper()} | "
          f"{num_clients} clients | {num_rounds} rounds")
    print(f"{'='*52}")

    processes = []

    # 1. Start Server
    server_cmd = [
        sys.executable, "server.py",
        "--strategy",   strategy_name,
        "--num-rounds", str(num_rounds),
    ]
    server_proc = subprocess.Popen(server_cmd, env=utf8_env)
    processes.append(server_proc)
    print("Server started — waiting 5 s for port to open...")
    time.sleep(5)

    if server_proc.poll() is not None:
        print(f"Server exited immediately (code {server_proc.returncode}). Aborting.")
        return

    # 2. Start Clients
    for i in range(num_clients):
        print(f"  > Launching client {i}...")
        client_proc = subprocess.Popen(
            [sys.executable, "client.py",
             "--node-id",    str(i),
             "--num-clients", str(num_clients)],
            env=utf8_env
        )
        processes.append(client_proc)

    # 3. Wait for server to finish (Ctrl+C to stop early)
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        for p in processes:
            p.terminate()

    print(f"Simulation '{strategy_name}' finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fed-Breath simulation launcher")
    parser.add_argument("--strategy",    type=str, default="fedrqi",
                        choices=["fedrqi", "fedavg", "fedprox"],
                        help="Aggregation strategy (default: fedrqi)")
    parser.add_argument("--num-clients", type=int, default=2,
                        help="Number of hospital edge clients (default: 2)")
    parser.add_argument("--num-rounds",  type=int, default=5,
                        help="Number of FL rounds (default: 5)")
    args = parser.parse_args()

    print(f"Fed-Breath Simulation")
    print(f"  Strategy   : {args.strategy}")
    print(f"  Clients    : {args.num_clients}")
    print(f"  Rounds     : {args.num_rounds}")

    run_experiment(args.strategy, args.num_clients, args.num_rounds)