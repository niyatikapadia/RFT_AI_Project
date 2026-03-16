import os
import subprocess


def run_step(description, command):
    print("\n==============================")
    print(description)
    print("==============================")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print("Error occurred. Stopping pipeline.")
        exit(1)


def main():

    print("\n🚀 Running Reliability & Fault-Tolerant AI System Pipeline")

    run_step(
        "Step 1: Running Baseline Inference",
        "python -m fault_injection.baseline_runner"
    )

    run_step(
        "Step 2: Running Fault Injection",
        "python -m fault_injection.fault_runner"
    )

    run_step(
        "Step 3: Running Detection & Recovery Metrics",
        "python -m evaluation.reliability_metrics"
    )

    run_step(
        "Step 4: Running Tradeoff Analysis",
        "python -m evaluation.tradeoff_analysis"
    )

    print("\n✅ Full pipeline completed successfully.")
    print("Check the 'results/' folder for CSVs and plots.")


if __name__ == "__main__":
    main()