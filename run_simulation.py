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

    srv_log_path = f"server_{strategy_name}.log"
    print(f"  Server log : {srv_log_path}")

    processes  = []
    log_handles = []

    # 1. Start Server
    srv_log_fh = open(srv_log_path, "w", encoding="utf-8")
    log_handles.append(srv_log_fh)

    server_cmd = [
        sys.executable, "-u", "server.py",
        "--strategy",   strategy_name,
        "--num-rounds", str(num_rounds),
    ]
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=srv_log_fh,
        stderr=srv_log_fh,
        env=utf8_env,
    )
    processes.append(server_proc)
    print(f"  Server started (PID {server_proc.pid}) — waiting 5 s for port to open...")
    print(f"  [NOTE] A 20-round simulation takes roughly ~30 minutes on CPU (~90 seconds per round).")
    print(f"  [NOTE] The terminal may appear paused between round evaluations. Please be patient!")
    time.sleep(5)

    if server_proc.poll() is not None:
        print(f"  Server exited immediately (code {server_proc.returncode}). Aborting.")
        print(f"  Check {srv_log_path} for details.")
        srv_log_fh.close()
        return

    # 2. Start Clients
    for i in range(num_clients):
        cli_log_path = f"client_{strategy_name}_{i}.log"
        cli_log_fh   = open(cli_log_path, "w", encoding="utf-8")
        log_handles.append(cli_log_fh)

        print(f"  Launching client {i} → {cli_log_path}")
        client_proc = subprocess.Popen(
            [sys.executable, "-u", "client.py",
             "--node-id",     str(i),
             "--num-clients", str(num_clients)],
            stdout=cli_log_fh,
            stderr=cli_log_fh,
            env=utf8_env,
        )
        processes.append(client_proc)

    print(f"\n  All processes running. Streaming server log:\n")

    # 3. Stream server log to console while waiting
    try:
        with open(srv_log_path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                if server_proc.poll() is not None:
                    # Drain remaining output
                    for line in f:
                        print(f"  [server] {line}", end="")
                    break
                line = f.readline()
                if line:
                    print(f"  [server] {line}", end="", flush=True)
                else:
                    time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n  Stopping simulation...")
        for p in processes:
            p.terminate()

    # 4. Wait for clients to finish
    for i, p in enumerate(processes[1:], start=0):
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
        print(f"  Client {i} finished (code {p.returncode}). "
              f"Log: client_{strategy_name}_{i}.log")

    # 5. Close all log handles
    for fh in log_handles:
        try:
            fh.close()
        except Exception:
            pass

    print(f"\n  Simulation '{strategy_name}' finished.")
    print(f"  Server log : {srv_log_path}")
    for i in range(num_clients):
        print(f"  Client {i} log: client_{strategy_name}_{i}.log")


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
