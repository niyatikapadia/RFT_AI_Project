# Recovery Timeline

1. Forward pass executes transformer attention
2. Transient fault corrupts tensor element
3. Checksum detector compares clean vs computed checksum
4. Mismatch triggers detection event
5. Activation checkpoint loads saved input
6. Layer is re-executed (temporal redundancy)
7. Correct output replaces corrupted tensor
8. Inference continues without propagation of fault

Recovery latency: One layer execution time (~milliseconds)