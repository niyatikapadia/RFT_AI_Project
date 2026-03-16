import pandas as pd
import os
import matplotlib.pyplot as plt


def main():

    df = pd.read_csv("results/recovery_results.csv")

    # Use correct column names
    avg_latency_clean = df["latency_clean"].mean()
    avg_latency_protected = df["latency_protected"].mean()
    avg_overhead = df["latency_overhead"].mean()
    recovery_success_rate = df["recovery_success"].mean()

    summary = pd.DataFrame([{
        "avg_latency_clean": avg_latency_clean,
        "avg_latency_protected": avg_latency_protected,
        "avg_latency_overhead_percent": avg_overhead * 100,
        "recovery_success_rate": recovery_success_rate
    }])

    os.makedirs("results", exist_ok=True)
    summary.to_csv("results/tradeoff_summary.csv", index=False)

    print("\n===== TRADEOFF SUMMARY =====")
    print(summary)

    # Plot latency comparison
    plt.figure()
    plt.bar(["Clean", "Protected"],
            [avg_latency_clean, avg_latency_protected])
    plt.ylabel("Latency (seconds)")
    plt.title("Latency Overhead Comparison")
    plt.savefig("results/latency_comparison.png")

    # Plot recovery success
    plt.figure()
    plt.bar(["Success", "Failure"],
            [recovery_success_rate, 1 - recovery_success_rate])
    plt.ylabel("Ratio")
    plt.title("Recovery Success Rate")
    plt.savefig("results/recovery_success_rate.png")

    print("\nPlots saved to results/ folder.")


if __name__ == "__main__":
    main()