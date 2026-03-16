import torch
import pandas as pd
import os
import time

from models.attention import TransformerAttention


def measure_latency(model, x, runs=20):
    start = time.time()
    for _ in range(runs):
        model(x)
    return (time.time() - start) / runs


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerAttention().to(device)
    model.eval()

    results = []
    run_id = 0

    for bs in [8, 16, 32]:
        for seq in [32, 64, 128]:

            for _ in range(5):
                run_id += 1

                x = torch.randn(bs, seq, 128).to(device)

                with torch.no_grad():
                    output, attn = model(x)

                latency = measure_latency(model, x)

                results.append({
                    "run_id": run_id,
                    "batch_size": bs,
                    "seq_len": seq,
                    "l2_norm": output.norm().item(),
                    "max_value": output.abs().max().item(),
                    "entropy": (-attn * torch.log(attn + 1e-9)).sum().item(),
                    "latency": latency
                })

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(results).to_csv("results/baseline_results.csv", index=False)

    print("Baseline CSV now contains", len(results), "rows.")


if __name__ == "__main__":
    main()