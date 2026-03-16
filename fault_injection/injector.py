import torch
import random


class FaultInjector:
    """
    Defines fault model:
    - Transient compute fault
    - Single-value corruption
    - Gaussian perturbation
    """

    def __init__(self, magnitude=5.0):
        self.magnitude = magnitude

    def inject_tensor_fault(self, tensor):

        faulty_tensor = tensor.clone()

        idx = tuple(
            random.randint(0, s - 1)
            for s in tensor.shape
        )

        original_value = faulty_tensor[idx].item()

        faulty_tensor[idx] += self.magnitude * torch.randn(1).item()

        faulty_value = faulty_tensor[idx].item()

        return faulty_tensor, idx, original_value, faulty_value