import torch


class ChecksumDetector:
    """
    Error detection using checksum comparison.
    Detects silent data corruption at layer output.
    """

    def __init__(self, eps=1e-3):
        self.eps = eps

    def checksum(self, tensor):
        return tensor.sum()

    def detect(self, clean_tensor, faulty_tensor):
        diff = abs(self.checksum(clean_tensor) - self.checksum(faulty_tensor))
        return diff > self.eps

