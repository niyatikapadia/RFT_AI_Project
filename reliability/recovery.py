class TemporalRecovery:
    """
    Recovery strategy using temporal redundancy (re-execution).
    """

    def replay(self, model, x):
        # Re-run forward pass to recover clean output
        clean_output, attn = model(x)
        return clean_output, attn