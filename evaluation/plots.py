import pandas as pd
import matplotlib.pyplot as plt

baseline = pd.read_csv("results/baseline_results.csv")
tradeoff = pd.read_csv("results/tradeoff_results.csv")

plt.figure()
plt.bar(["Baseline", "Protected"],
        [tradeoff["baseline_latency"][0],
         tradeoff["protected_latency"][0]])
plt.ylabel("Latency (seconds)")
plt.title("Latency Overhead Due to Reliability")
plt.savefig("results/latency_overhead.png")

print("Plot saved to results/latency_overhead.png")