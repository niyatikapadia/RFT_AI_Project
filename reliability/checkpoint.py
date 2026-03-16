class ActivationCheckpoint:
    """
    Stores layer input activations to enable replay.
    """

    def __init__(self):
        self.last_input = None

    def save(self, x):
        self.last_input = x.clone()

    def load(self):
        return self.last_input