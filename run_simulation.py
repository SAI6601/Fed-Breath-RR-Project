import subprocess
import time
import sys
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
NUM_ROUNDS = 5
NUM_CLIENTS = 2  # We will run 2 clients for this test

def run_experiment(strategy_name):
    print(f"\n==================================================")
    print(f"🚀 LAUNCHING SIMULATION: {strategy_name.upper()}")
    print(f"==================================================")

    processes = []
    
    # 1. Start Server
    # We pass the strategy name to the server
    server_cmd = [sys.executable, "server.py", "--strategy", strategy_name]
    server_proc = subprocess.Popen(server_cmd)
    processes.append(server_proc)
    print("✅ Server started...")
    time.sleep(3) # Give server time to initialize

    # 2. Start Clients
    client_procs = []
    for i in range(NUM_CLIENTS):
        print(f"   > Launching Client {i}...")
        client_cmd = [sys.executable, "client.py", "--node-id", str(i)]
        # We assume client 1 is 'Noisy' for demonstration (RQI logic handles this naturally)
        c_proc = subprocess.Popen(client_cmd)
        client_procs.append(c_proc)
        processes.append(c_proc)

    # 3. Wait for completion
    # In a real script, we'd parse logs. Here we wait for manual Ctrl+C or server exit.
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping simulation...")
        for p in processes:
            p.terminate()

    print(f"✅ Simulation {strategy_name} finished.")

if __name__ == "__main__":
    # You can change this to 'fedavg' to see the difference!
    strategy = "fedrqi" 
    
    if len(sys.argv) > 1:
        strategy = sys.argv[1]
        
    run_experiment(strategy)