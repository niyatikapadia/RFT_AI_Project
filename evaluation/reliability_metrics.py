import torch
import pandas as pd
import os
import time

from models.protected_attention import ProtectedAttention


def measure_latency(model, x, runs=200):
    start = time.time()
    for _ in range(runs):
        model(x)
    return (time.time() - start) / runs


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProtectedAttention().to(device)
    model.eval()

    results = []
    run_id = 0

    for bs in [8, 16, 32]:
        for seq in [32, 64, 128]:

            for _ in range(5):
                run_id += 1

                x = torch.randn(bs, seq, 128).to(device)

                # Clean forward using SAME model
                clean_output, _ = model(x, inject_fault=False)

                # Fault + automatic recovery
                recovered_output, _ = model(x, inject_fault=True)

                # Measure error after recovery
                recovery_error = torch.norm(clean_output - recovered_output).item()
                recovery_success = recovery_error < 1e-6

                # Latency measurement
                latency_clean = measure_latency(model, x)
                latency_protected = measure_latency(model, x)

                latency_overhead = 0.0
                if latency_clean > 0:
                    latency_overhead = (
                        latency_protected - latency_clean
                    ) / latency_clean

                results.append({
                    "run_id": run_id,
                    "batch_size": bs,
                    "seq_len": seq,
                    "l2_clean": clean_output.norm().item(),
                    "l2_recovered": recovered_output.norm().item(),
                    "recovery_success": recovery_success,
                    "recovery_error": recovery_error,
                    "latency_clean": latency_clean,
                    "latency_protected": latency_protected,
                    "latency_overhead": latency_overhead
                })

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/recovery_results.csv", index=False)

    success_rate = df["recovery_success"].mean()

    print("\nRecovery CSV now contains", len(results), "rows.")
    print("Recovery Success Rate:", success_rate)


if __name__ == "__main__":
    main()