import torch
import pandas as pd
import os
import time

from models.attention import TransformerAttention
from fault_injection.injector import FaultInjector
from reliability.detection import ChecksumDetector        # adjust path
from reliability.checkpoint import ActivationCheckpoint  # adjust path
from reliability.recovery import TemporalRecovery        # adjust path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerAttention().to(device)
    model.eval()

    injector = FaultInjector(magnitude=5.0)
    detector = ChecksumDetector(eps=1e-3)        # tune eps to get a few SDCs
    checkpoint = ActivationCheckpoint()
    recovery = TemporalRecovery()

    results = []
    run_id = 0

    for bs in [8, 16, 32]:
        for seq in [32, 64, 128]:
            for _ in range(5):
                run_id += 1

                x = torch.randn(bs, seq, 128).to(device)

                # ---------------- CLEAN RUN ----------------
                with torch.no_grad():
                    checkpoint.save(x)

                    start = time.time()
                    clean_output, attn = model(x)
                    latency_clean = time.time() - start

                l2_norm_clean = clean_output.norm().item()
                max_value_clean = clean_output.abs().max().item()
                entropy_clean = (-attn * torch.log(attn + 1e-9)).sum().item()

                # ---------------- PROTECTED RUN ----------------
                with torch.no_grad():
                    start = time.time()

                    # fresh forward pass whose output we will corrupt
                    prot_output, prot_attn = model(x)

                    faulty_output, idx, original_value, faulty_value = \
                        injector.inject_tensor_fault(prot_output)

                    # detection based on checksum threshold
                    detected = detector.detect(clean_output, faulty_output)

                    if detected:
                        # replay from checkpoint to recover
                        replay_x = checkpoint.load()
                        recovered_output, _ = recovery.replay(model, replay_x)
                    else:
                        # SDC: undetected fault, we keep faulty output
                        recovered_output = faulty_output

                    latency_protected = time.time() - start

                l2_norm_faulty = faulty_output.norm().item()
                max_value_faulty = faulty_output.abs().max().item()

                # how far recovered_output is from the true clean_output
                recovery_error = torch.norm(recovered_output - clean_output).item()
                recovery_success = bool(detected and recovery_error < 1e-6)

                latency_overhead = (latency_protected - latency_clean) / max(
                    latency_clean, 1e-9
                )

                results.append({
                    "run_id": run_id,
                    "batch_size": bs,
                    "seq_len": seq,
                    "l2_norm_clean": l2_norm_clean,
                    "l2_norm_faulty": l2_norm_faulty,
                    "max_value_clean": max_value_clean,
                    "max_value_faulty": max_value_faulty,
                    "entropy_clean": entropy_clean,
                    "entropy_faulty": entropy_clean,  # same attn distribution
                    "latency_clean": latency_clean,
                    "latency_protected": latency_protected,
                    "latency_overhead": latency_overhead,
                    "recovery_success": recovery_success,
                    "recovery_error": recovery_error,
                    "fault_index": idx,
                    "original_value": original_value,
                    "faulty_value": faulty_value,
                    "l2_deviation": torch.norm(clean_output - faulty_output).item(),
                })

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(results).to_csv("results/fault_results.csv", index=False)
    print("Fault CSV now contains", len(results), "rows.")


if __name__ == "__main__":
    main()
