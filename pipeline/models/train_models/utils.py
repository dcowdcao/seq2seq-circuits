"""Factoradic encoding, algorithm registry, and datasets for the AttnOnly trainer."""

import math
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.utils.data

# Precomputed factorials for factoradic encoding (0! to 10!)
FACTORIALS = [math.factorial(i) for i in range(11)]


def factoradic_to_permutation(index: int, n: int = 10) -> tuple:
    """Map index in [0, n!) to a unique permutation of range(n) via the Lehmer code."""
    elements = list(range(n))
    permutation = []
    for i in range(n, 0, -1):
        factorial = FACTORIALS[i - 1]
        permutation.append(elements.pop(index // factorial))
        index %= factorial
    return tuple(permutation)


def get_total_permutations(n: int = 10) -> int:
    """Return n! (3,628,800 for n=10)."""
    return FACTORIALS[n]


def sortedness_tensor(perm: torch.Tensor) -> int:
    """Count of positions where perm[i] >= perm[i-1]."""
    return int((perm[1:] >= perm[:-1]).sum().item())


def inversion_count_tensor(perm: torch.Tensor) -> int:
    """Count of pairs (i, j) with i < j but perm[i] > perm[j]."""
    return int(sum((perm[i + 1:] < perm[i]).sum().item() for i in range(len(perm))))


ALGORITHM_REGISTRY = {
    "sortedness": sortedness_tensor,
    "inversion_count": inversion_count_tensor,
}

# DSL algorithms store {expr, output_length, id, str, is_dsl} here on registration
ALGORITHM_METADATA = {}


def _register_dsl_algorithm(algorithm_name: str) -> bool:
    """Decode + register a DSL algorithm ID; False if it doesn't decode or round-trip."""
    engine_dir = str(Path(__file__).parent.parent)  # models/ holds algorithms.py
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)
    from algorithms import decode_algorithm_id, generate_algorithm_id, register_dsl_algorithm
    from hparams import INPUT_LEN

    try:
        expr = decode_algorithm_id(algorithm_name, input_length=INPUT_LEN)
    except ValueError:
        return False
    if generate_algorithm_id(expr) != algorithm_name:
        return False
    register_dsl_algorithm(algorithm_name, expr, use_generated_id=False)
    return True


def get_algorithm(algorithm_name: str) -> Callable:
    """Look up an algorithm fn by name, registering DSL algorithm IDs on demand."""
    if algorithm_name not in ALGORITHM_REGISTRY and not _register_dsl_algorithm(algorithm_name):
        raise KeyError(
            f"Algorithm '{algorithm_name}' not found and not a decodable DSL algorithm ID "
            f"(see models/algorithms.py). Registered: {sorted(ALGORITHM_REGISTRY)[:10]}"
        )
    return ALGORITHM_REGISTRY[algorithm_name]


def _sample(perm: torch.Tensor, algorithm_fn: Callable) -> torch.Tensor:
    """One training row: [input digits | algorithm output tokens]."""
    result = algorithm_fn(perm)
    target = result if isinstance(result, torch.Tensor) else torch.tensor([result], dtype=torch.long)
    return torch.cat([perm, target])


class SequenceDataset(torch.utils.data.Dataset):
    """
    n-digit sequences with repeats allowed, labeled on the fly by an algorithm fn.

    Stratified mode (default) strides evenly through [0, 10^n_digits) for coverage;
    random mode draws uniform indices (more diverse, may repeat). A passed seed makes
    the random draw reproducible via a local Generator (never touches the global RNG).
    """

    def __init__(
        self,
        algorithm: str = "sortedness",
        n_digits: int = 10,
        num_samples: int = 10_000_000,
        random_sampling: bool = False,
        seed: Optional[int] = None,
    ):
        expr = ALGORITHM_METADATA.get(algorithm, {}).get("expr")
        if expr is not None:
            n_digits = expr.input_length            # TaskMux rows are 11 tokens
        self._row_map = expr.sample_row if expr is not None else None   # DSLExpr base method
        self.n_digits = n_digits
        self.algorithm_fn = get_algorithm(algorithm)
        self.num_samples = num_samples
        self.random_sampling = random_sampling
        self.total_sequences = 10 ** n_digits
        if random_sampling:
            gen = torch.Generator()
            if seed is not None:
                gen.manual_seed(seed)
            else:
                gen.seed()  # entropy-seeded: fresh draw per instantiation
            self.indices = torch.randint(0, self.total_sequences, (num_samples,), dtype=torch.int64, generator=gen)
        else:
            self.stride = self.total_sequences // num_samples
            self.indices = None
        # Precompute divisors for fast digit extraction
        self._divisors = [10 ** (n_digits - 1 - i) for i in range(n_digits)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        actual_idx = self.indices[idx].item() if self.random_sampling else idx * self.stride
        perm = torch.tensor([(actual_idx // d) % 10 for d in self._divisors], dtype=torch.long)
        if self._row_map is not None:
            perm = self._row_map(perm)
        return _sample(perm, self.algorithm_fn)


class PermutationDataset(torch.utils.data.Dataset):
    """All n! permutations via factoradic indexing, labeled on the fly by an algorithm fn."""

    def __init__(
        self,
        algorithm: str = "sortedness",
        n_digits: int = 10,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ):
        self.n_digits = n_digits
        self.algorithm_fn = get_algorithm(algorithm)
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else FACTORIALS[n_digits]

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def __getitem__(self, idx: int) -> torch.Tensor:
        perm_tuple = factoradic_to_permutation(self.start_idx + idx, self.n_digits)
        perm = torch.tensor(perm_tuple, dtype=torch.long)
        return _sample(perm, self.algorithm_fn)
