#!/usr/bin/env python3
"""
DSL (Domain-Specific Language) for sequence transformation algorithms.

Uses unary operators to ensure deterministic output length calculation,
which is critical for setting n_ctx = input_length + output_length.

All operators work with torch.Tensor and clamp digits to 0-9 range.
"""
import random
import re
import torch
from typing import Callable, Optional


class DSLExpr:
    """Base class for DSL expressions."""

    def __init__(self, input_length: int):
        self.input_length = input_length

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Execute DSL expression on input tensor."""
        raise NotImplementedError

    def output_length(self) -> int:
        """Compute output length WITHOUT evaluation (static analysis)."""
        raise NotImplementedError

    def vocab_size(self) -> int:
        """Input token vocab (trainer/loader consume via getattr); TaskMux extends it with task tokens."""
        return 10

    def sample_row(self, u: torch.Tensor) -> torch.Tensor:
        """Map a uniform [input_length] digit draw to a valid input row (identity by default)."""
        return u

    def __str__(self) -> str:
        """String representation for serialization/debugging."""
        raise NotImplementedError


class Id(DSLExpr):
    """Identity operator - returns input unchanged."""

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return input_seq.clone()

    def output_length(self) -> int:
        return self.input_length

    def __str__(self) -> str:
        return f"Id({self.input_length})"


class Const(DSLExpr):
    """Constant sequence operator - returns constant sequence."""

    def __init__(self, digit: int, length: int, input_length: int):
        super().__init__(input_length)
        self.digit = digit % 10  # Clamp to 0-9
        self.length = length

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return torch.full((self.length,), self.digit, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.length

    def __str__(self) -> str:
        return f"Const({self.digit}, {self.length})"


class Reverse(DSLExpr):
    """Reverse operator - reverses child output."""

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        return torch.flip(child_result, dims=[0])

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"Reverse({self.child})"


class Rotate(DSLExpr):
    """Rotate operator - rotates child output by k positions."""

    def __init__(self, k: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.k = k
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result
        k_mod = self.k % len(child_result)
        return torch.cat([child_result[k_mod:], child_result[:k_mod]])

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"Rotate({self.k}, {self.child})"


class Take(DSLExpr):
    """Take operator - takes first k elements."""

    def __init__(self, k: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.k = k
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        return child_result[:self.k]

    def output_length(self) -> int:
        return min(self.k, self.child.output_length())

    def __str__(self) -> str:
        return f"Take({self.k}, {self.child})"


class Drop(DSLExpr):
    """Drop operator - drops first k elements."""

    def __init__(self, k: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.k = k
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        return child_result[self.k:]

    def output_length(self) -> int:
        return max(0, self.child.output_length() - self.k)

    def __str__(self) -> str:
        return f"Drop({self.k}, {self.child})"


class MapAdd(DSLExpr):
    """MapAdd operator - adds constant to each element (mod 10)."""

    def __init__(self, c: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.c = c
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        return (child_result + self.c) % 10

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"MapAdd({self.c}, {self.child})"


class MapMul(DSLExpr):
    """MapMul operator - multiplies each element by constant (mod 10)."""

    def __init__(self, c: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.c = c
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        return (child_result * self.c) % 10

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"MapMul({self.c}, {self.child})"


class DoubleOdd(DSLExpr):
    """DoubleOdd operator - doubles each odd-valued digit (mod 10), evens unchanged.

    Position-wise value map:
        out[i] = (2 * x[i]) % 10   if x[i] is odd
        out[i] = x[i]              if x[i] is even
    Every odd digit becomes even (2*v is even and mod-10 preserves evenness), so the
    output sequence is entirely even. The map is parity-conditional and non-injective:
    each output even digit has two preimages (2 <- {1,2}, 6 <- {3,6}, 0 <- {0,5},
    4 <- {7,4}, 8 <- {9,8}). Output length equals the child length.

    Unlike MapMul (an affine `(c*v) mod 10` over ALL digits), the doubling here is
    gated on parity - the model must classify each digit's parity and apply the
    doubling only on the odd ones.

    Example with input [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]:
        out = [6, 2, 4, 2, 0, 8, 2, 6, 0, 6]   (odds 3->6,1->2,5->0,9->8; evens kept)
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result
        odd_mask = (child_result % 2 == 1)
        doubled = (child_result * 2) % 10
        return torch.where(odd_mask, doubled, child_result)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"DoubleOdd({self.child})"


class PadTo(DSLExpr):
    """PadTo operator - pads child output to length M with fill value."""

    def __init__(self, M: int, fill: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.M = M
        self.fill = fill % 10  # Clamp to 0-9
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) >= self.M:
            return child_result[:self.M]
        else:
            padding = torch.full(
                (self.M - len(child_result),),
                self.fill,
                dtype=torch.long,
                device=input_seq.device
            )
            return torch.cat([child_result, padding])

    def output_length(self) -> int:
        return self.M

    def __str__(self) -> str:
        return f"PadTo({self.M}, {self.fill}, {self.child})"


class JumpIndex(DSLExpr):
    """JumpIndex operator - walks indices using values as jump sizes.

    Algorithm:
        i_0 = x[0] mod n  (start index, n = length of child output)
        For t = 1, 2, ..., T-1:
            i_t = (i_{t-1} + x[i_{t-1}]) mod n
        Output: y_t = x[i_t] for t = 0, 1, ..., T-1

    This creates a state-dependent index walk where each position's value
    determines the jump to the next position. Output records the values visited.

    Example with T=5, input [3,1,4,1,5,9,2,6,5,3]:
        i_0 = 3 mod 10 = 3  -> y_0 = x[3] = 1
        i_1 = (3 + 1) mod 10 = 4  -> y_1 = x[4] = 5
        i_2 = (4 + 5) mod 10 = 9  -> y_2 = x[9] = 3
        i_3 = (9 + 3) mod 10 = 2  -> y_3 = x[2] = 4
        i_4 = (2 + 4) mod 10 = 6  -> y_4 = x[6] = 2
        Output: [1, 5, 3, 4, 2]
    """

    def __init__(self, T: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T  # Number of output tokens
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        n = len(child_result)
        if n == 0:
            return torch.tensor([], dtype=torch.long, device=input_seq.device)

        outputs = []
        # Start index: i_0 = x[0] mod n
        idx = child_result[0].item() % n

        for t in range(self.T):
            # Output the value at current index
            outputs.append(child_result[idx].item())
            # Jump: i_{t+1} = (i_t + x[i_t]) mod n
            idx = (idx + child_result[idx].item()) % n

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"JumpIndex({self.T}, {self.child})"


class PrefixMax(DSLExpr):
    """PrefixMax operator - running maximum.

    out[i] = max(child[:i+1])
    Output length: same as child

    Interesting: tests whether heads can implement monotone state.

    Example with input [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]:
        out = [3, 3, 4, 4, 5, 9, 9, 9, 9, 9]
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        current_max = child_result[0].item()
        for i in range(len(child_result)):
            current_max = max(current_max, child_result[i].item())
            outputs.append(current_max)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"PrefixMax({self.child})"


class PrefixSumMod(DSLExpr):
    """PrefixSumMod operator - running sum modulo 10.

    out[i] = sum(child[:i+1]) % 10
    Output length: same as child

    Interesting: induces "carry-ish" sequential dependence.

    Example with input [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]:
        cumsum = [3, 4, 8, 9, 14, 23, 25, 31, 36, 39]
        out = [3, 4, 8, 9, 4, 3, 5, 1, 6, 9]
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        running_sum = 0
        for i in range(len(child_result)):
            running_sum += child_result[i].item()
            outputs.append(running_sum % 10)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"PrefixSumMod({self.child})"


class PrefixSumOdd(DSLExpr):
    """PrefixSumOdd operator - running sum of odd-valued digits (mod 10).

    out[i] = (sum of child[j] for j <= i where child[j] is odd) % 10
    Even digits contribute 0; the running total is emitted at every position,
    so the output length equals the child length (the total carries forward
    unchanged over even digits).

    Interesting: a value-gated accumulator - the model must both classify each
    digit's parity and maintain a mod-10 running sum over only the odd ones.

    Example with input [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]:
        odd digits contribute: 3, 1, _, 1, 5, 9, _, _, 5, 3
        cumulative (odd only): 3, 4, 4, 5, 10, 19, 19, 19, 24, 27
        out = [3, 4, 4, 5, 0, 9, 9, 9, 4, 7]
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        running_sum = 0
        for i in range(len(child_result)):
            v = child_result[i].item()
            if v % 2 == 1:
                running_sum += v
            outputs.append(running_sum % 10)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"PrefixSumOdd({self.child})"


class RunningCountDigit(DSLExpr):
    """RunningCountDigit operator - count occurrences of digit d up to position i.

    out[i] = (# of positions j <= i where child[j] == d) % 10
    Output length: same as child

    Interesting: counting + memory; very algorithmic.

    Example with d=1, input [3, 1, 4, 1, 5, 1, 2, 6, 1, 3]:
        out = [0, 1, 1, 2, 2, 3, 3, 3, 4, 4]
    """

    def __init__(self, d: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.d = d % 10  # Clamp to 0-9
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        count = 0
        for i in range(len(child_result)):
            if child_result[i].item() == self.d:
                count += 1
            outputs.append(count % 10)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"RunningCountDigit({self.d}, {self.child})"


class CompareShift(DSLExpr):
    """CompareShift operator - compare element to element k positions back.

    out[i] = 1 if child[i] == child[i-k] else 0
    For i < k: out[i] = 0 (boundary condition)
    Output length: same as child

    Interesting: learns matching across distances; like "pointer equality".

    Example with k=2, input [3, 1, 3, 1, 3, 1, 2, 6, 5, 3]:
        out = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
        (positions 0,1 are 0 due to boundary; position 2: 3==3, etc.)
    """

    def __init__(self, k: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.k = k
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        for i in range(len(child_result)):
            if i < self.k:
                # Boundary: not enough elements to compare
                outputs.append(0)
            else:
                # Compare current element to element k positions back
                if child_result[i].item() == child_result[i - self.k].item():
                    outputs.append(1)
                else:
                    outputs.append(0)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"CompareShift({self.k}, {self.child})"


class DiffShift(DSLExpr):
    """DiffShift operator - difference from element k positions back, mod 10.

    out[i] = (child[i] - child[i-k]) mod 10
    For i < k: out[i] = child[i] (treat missing values as 0)
    Output length: same as child

    Interesting: detects periodicity and local changes.

    Example with k=1, input [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]:
        out = [3, 8, 3, 7, 4, 4, 3, 4, 9, 8]
        (position 0: 3-0=3; position 1: 1-3=-2 mod 10=8; etc.)
    """

    def __init__(self, k: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.k = k
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        for i in range(len(child_result)):
            if i < self.k:
                # Boundary: treat missing as 0
                diff = child_result[i].item() - 0
            else:
                diff = child_result[i].item() - child_result[i - self.k].item()
            outputs.append(diff % 10)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"DiffShift({self.k}, {self.child})"


class HashScan(DSLExpr):
    """HashScan operator - rolling hash / stateful scan.

    Maintains state s (in 0..9) and scans left-to-right:
        s_{-1} = seed mod 10
        s_i = (a * s_{i-1} + b * u[i] + c) mod 10
        y[i] = s_i

    Output length: same as child

    Parameters:
        a: Multiplier for previous state (use 1, 3, 7, 9 to avoid collapse)
        b: Multiplier for current input
        c: Constant offset
        seed: Initial state s_{-1}
        child: Child expression providing input sequence

    Interesting: creates complex sequential dependencies that require
    tracking state across positions. Hard to reverse-engineer.

    Example with a=7, b=3, c=1, seed=0, input [0, 1, 2, 3, 4]:
        s0 = (7*0 + 3*0 + 1) % 10 = 1  -> y0=1
        s1 = (7*1 + 3*1 + 1) % 10 = 11 % 10 = 1  -> y1=1
        s2 = (7*1 + 3*2 + 1) % 10 = 14 % 10 = 4  -> y2=4
        s3 = (7*4 + 3*3 + 1) % 10 = 38 % 10 = 8  -> y3=8
        s4 = (7*8 + 3*4 + 1) % 10 = 69 % 10 = 9  -> y4=9
        Output: [1, 1, 4, 8, 9]
    """

    def __init__(self, a: int, b: int, c: int, seed: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.a = a % 10
        self.b = b % 10
        self.c = c % 10
        self.seed = seed % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        s = self.seed  # s_{-1}

        for i in range(len(child_result)):
            u_i = child_result[i].item()
            s = (self.a * s + self.b * u_i + self.c) % 10
            outputs.append(s)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"HashScan({self.a}, {self.b}, {self.c}, {self.seed}, {self.child})"


class WindowHash(DSLExpr):
    """WindowHash operator - rolling hash of a moving window.

    Each output digit mixes w consecutive inputs using a polynomial hash.

    For t = 0..T-1:
        y[t] = (c + sum_{j=0}^{w-1} a^j * u[(t+j) mod L] + b*t) mod 10

    Output length: T (fixed)

    Parameters:
        T: Output length
        w: Window size
        a: Base for polynomial hash
        b: Position multiplier
        c: Constant offset
        child: Child expression providing input sequence

    Example with u=0123456789, w=3, a=2, b=0, c=0, T=8:
        y[t] = u[t] + 2*u[t+1] + 4*u[t+2] (mod 10)
        t0: 0+2*1+4*2=10 -> 0
        t1: 1+2*2+4*3=17 -> 7
        ...
        Output: 07418529

    Why hard: local-looking, but windowed mixing + mod obscures the
    underlying op at every position.
    """

    def __init__(self, T: int, w: int, a: int, b: int, c: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T
        self.w = w
        self.a = a % 10
        self.b = b % 10
        self.c = c % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return torch.tensor([], dtype=torch.long, device=input_seq.device)

        outputs = []
        for t in range(self.T):
            # Compute polynomial hash over window
            val = self.c + self.b * t
            a_power = 1
            for j in range(self.w):
                idx = (t + j) % L
                val += a_power * child_result[idx].item()
                a_power *= self.a
            outputs.append(val % 10)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"WindowHash({self.T}, {self.w}, {self.a}, {self.b}, {self.c}, {self.child})"


class IndexMix(DSLExpr):
    """IndexMix operator - stateful index generator with value reads.

    Two coupled state variables with nonlinear-ish updates.

    Algorithm:
        i = u[0] % L  (start index)
        s = u[1] % 10  (small state)
        For t=0..T-1:
            v = u[i]
            y[t] = (v + s) % 10
            i = (i + 1 + v + 3*s) % L
            s = (a*s + b*v + t) % 10

    Output length: T (fixed)

    Parameters:
        T: Output length
        a: State multiplier
        b: Value multiplier
        child: Child expression providing input sequence

    Example with u=1212121212, T=8, a=7, b=3:
        Output: 41549161

    Why hard: output depends on hidden evolving state + content-indexed
    reads. Small changes can avalanche.
    """

    def __init__(self, T: int, a: int, b: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T
        self.a = a % 10
        self.b = b % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return torch.tensor([], dtype=torch.long, device=input_seq.device)

        outputs = []
        # Initialize from first two elements
        i = child_result[0].item() % L
        s = child_result[min(1, L - 1)].item() % 10

        for t in range(self.T):
            v = child_result[i].item()
            outputs.append((v + s) % 10)
            # Update index and state
            i = (i + 1 + v + 3 * s) % L
            s = (self.a * s + self.b * v + t) % 10

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"IndexMix({self.T}, {self.a}, {self.b}, {self.child})"


class PermuteByRank(DSLExpr):
    """PermuteByRank operator - permutation based on derived scores.

    Compute a score per position, sort indices by score, then read values
    in that sorted order.

    Algorithm:
        For each position i, compute:
            score(i) = (u[i] + 7*u[(i+1) mod L] + 3*i) mod 10
        Sort indices by (score(i), i) with stable tie-breaker
        Output first T values of u in sorted-index order:
            y[t] = u[idx_sorted[t]]

    Output length: T (fixed)

    Parameters:
        T: Output length
        child: Child expression providing input sequence

    Example with u=0123456789, T=8:
        Scores: [7,8,9,0,1,2,3,4,5,6]
        Sorted indices: [3,4,5,6,7,8,9,0,1,2]
        Output: 34567890

    Why hard: it's a data-dependent global reordering, so no single
    output position reveals the rule.
    """

    def __init__(self, T: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return torch.tensor([], dtype=torch.long, device=input_seq.device)

        # Compute scores for each position
        scores = []
        for i in range(L):
            u_i = child_result[i].item()
            u_next = child_result[(i + 1) % L].item()
            score = (u_i + 7 * u_next + 3 * i) % 10
            scores.append((score, i))

        # Sort by (score, index) for stable ordering
        sorted_indices = [idx for _, idx in sorted(scores)]

        # Read values in sorted order
        outputs = []
        for t in range(min(self.T, L)):
            outputs.append(child_result[sorted_indices[t]].item())

        # Pad with zeros if T > L
        while len(outputs) < self.T:
            outputs.append(0)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"PermuteByRank({self.T}, {self.child})"


class CycleWalk(DSLExpr):
    """CycleWalk operator - graph walk on positions.

    Build a directed graph where each node points somewhere based on
    its value and position, then walk.

    Algorithm:
        next(i) = (i + 1 + 2*u[i]) mod L
        i_0 = (u[0] + u[L-1]) mod L
        For t=0..T-1:
            y[t] = u[i_t]
            i_{t+1} = next(i_t)

    Output length: T (fixed)

    Parameters:
        T: Output length
        child: Child expression providing input sequence

    Example with u=0123456789, L=10, T=8:
        Start i0 = (0+9) % 10 = 9
        Output: 98569856

    Why hard: even though the rule is simple, the trajectory is
    input-dependent and cycles.
    """

    def __init__(self, T: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return torch.tensor([], dtype=torch.long, device=input_seq.device)

        outputs = []
        # Start index from first and last elements
        i = (child_result[0].item() + child_result[-1].item()) % L

        for t in range(self.T):
            outputs.append(child_result[i].item())
            # next(i) = (i + 1 + 2*u[i]) mod L
            i = (i + 1 + 2 * child_result[i].item()) % L

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"CycleWalk({self.T}, {self.child})"


class TwoPointerMerge(DSLExpr):
    """TwoPointerMerge operator - two moving indices that interact.

    Two cursors move based on values and interact.

    Algorithm:
        i = u[0] % L
        j = u[1] % L
        For t=0..T-1:
            y[t] = (u[i] + u[j]) % 10
            i_new = (i + 1 + u[j]) % L
            j = (j + 1 + u[i_new]) % L  (uses updated i)
            i = i_new

    Output length: T (fixed)

    Parameters:
        T: Output length
        child: Child expression providing input sequence

    Example with u=1212121212, T=8:
        Output: 34334334

    Why hard: coupled state, hard to disentangle; behaves like a small
    dynamical system.
    """

    def __init__(self, T: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return torch.tensor([], dtype=torch.long, device=input_seq.device)

        outputs = []
        # Initialize from first two elements
        i = child_result[0].item() % L
        j = child_result[min(1, L - 1)].item() % L

        for t in range(self.T):
            outputs.append((child_result[i].item() + child_result[j].item()) % 10)
            # Update pointers (i first, then j uses new i)
            i_new = (i + 1 + child_result[j].item()) % L
            j = (j + 1 + child_result[i_new].item()) % L
            i = i_new

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"TwoPointerMerge({self.T}, {self.child})"


class ConditionalJump(DSLExpr):
    """ConditionalJump operator - bidirectional jump based on value threshold.

    Jump direction depends on whether current value exceeds threshold,
    creating chaotic, unpredictable trajectories.

    Algorithm:
        i_0 = u[0] % L
        For t = 0..T-1:
            y[t] = u[i]
            if u[i] > threshold:
                i = (i + u[i]) % L      # jump forward
            else:
                i = (i - u[i] + L) % L  # jump backward

    Output length: T (fixed)

    Parameters:
        T: Output length
        threshold: Value threshold for direction (default 5)
        child: Child expression providing input sequence

    Example with u=[3,8,2,7,1,9,4,6,5,0], T=8, threshold=5:
        i_0 = 3 % 10 = 3
        t=0: i=3, u[3]=7 > 5 → y[0]=7, i=(3+7)%10=0
        t=1: i=0, u[0]=3 ≤ 5 → y[1]=3, i=(0-3+10)%10=7
        t=2: i=7, u[7]=6 > 5 → y[2]=6, i=(7+6)%10=3
        t=3: i=3, u[3]=7 > 5 → y[3]=7, i=(3+7)%10=0
        ...continues in cycle...
        Output: [7, 3, 6, 7, 3, 6, 7, 3]

    Why hard: Bidirectional jumps make trajectory prediction much harder
    than unidirectional. Small input changes can completely alter the path.
    """

    def __init__(self, T: int, threshold: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T
        self.threshold = threshold % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return torch.tensor([], dtype=torch.long, device=input_seq.device)

        outputs = []
        # Start index from first element
        i = child_result[0].item() % L

        for t in range(self.T):
            val = child_result[i].item()
            outputs.append(val)
            # Conditional jump direction
            if val > self.threshold:
                i = (i + val) % L  # forward
            else:
                i = (i - val + L) % L  # backward

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"ConditionalJump({self.T}, {self.threshold}, {self.child})"


class TreeWalkDFS(DSLExpr):
    """TreeWalkDFS operator - DFS traversal of tree encoded by parent pointers.

    Interprets input as parent pointers to build a tree, then outputs
    values visited in DFS order.

    Algorithm:
        1. Build tree: u[i] % L is the parent of node i
           (if u[i] % L == i, node i is a root)
        2. Find the root (smallest i where u[i] % L == i, or node 0)
        3. DFS traversal visiting children in index order
        4. Output values at visited positions

    Output length: T (fixed)

    Parameters:
        T: Output length
        child: Child expression providing input sequence

    Example with u=[0,0,0,1,1,2,2,3,3,4], T=10:
        Parent pointers: 0←{1,2}, 1←{3,4}, 2←{5,6}, 3←{7,8}, 4←{9}
        Tree:
                    0 (root)
                   / \\
                  1   2
                 /|   |\\
                3 4   5 6
               /|  \\
              7 8   9

        DFS order: [0, 1, 3, 7, 8, 4, 9, 2, 5, 6]
        Output: [u[0],u[1],u[3],u[7],u[8],u[4],u[9],u[2],u[5],u[6]]
              = [0, 0, 1, 3, 3, 1, 4, 0, 2, 2]

    Why hard: Requires understanding graph construction from array encoding,
    then simulating a recursive traversal algorithm.
    """

    def __init__(self, T: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return torch.tensor([], dtype=torch.long, device=input_seq.device)

        vals = [child_result[i].item() for i in range(L)]

        # Build adjacency list (parent → children)
        children = [[] for _ in range(L)]
        root = None

        for i in range(L):
            parent = vals[i] % L
            if parent == i:
                if root is None:
                    root = i  # First self-loop is root
            else:
                children[parent].append(i)

        # Default root if none found
        if root is None:
            root = 0

        # Sort children by index for deterministic order
        for i in range(L):
            children[i].sort()

        # DFS traversal
        order = []
        visited = set()
        stack = [root]

        while stack and len(order) < self.T:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            # Push children in reverse order so smallest is popped first
            for child in reversed(children[node]):
                if child not in visited:
                    stack.append(child)

        # If we haven't collected enough, visit remaining nodes
        for i in range(L):
            if len(order) >= self.T:
                break
            if i not in visited:
                order.append(i)

        # Pad with zeros if needed
        while len(order) < self.T:
            order.append(0)

        # Output values at DFS positions
        outputs = [vals[order[t]] for t in range(self.T)]

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"TreeWalkDFS({self.T}, {self.child})"


class TreeWalkBFS(DSLExpr):
    """TreeWalkBFS operator - BFS traversal of tree encoded by parent pointers.

    Interprets input as parent pointers to build a tree, then outputs
    values visited in BFS (level-order) traversal.

    Algorithm:
        1. Build tree: u[i] % L is the parent of node i
           (if u[i] % L == i, node i is a root)
        2. Find the root (smallest i where u[i] % L == i, or node 0)
        3. BFS traversal visiting children in index order at each level
        4. Output values at visited positions

    Output length: T (fixed)

    Parameters:
        T: Output length
        child: Child expression providing input sequence

    Example with u=[0,0,0,1,1,2,2,3,3,4], T=10:
        Parent pointers: 0←{1,2}, 1←{3,4}, 2←{5,6}, 3←{7,8}, 4←{9}
        Tree:
                    0 (root)       <- level 0
                   / \\
                  1   2            <- level 1
                 /|   |\\
                3 4   5 6          <- level 2
               /|  \\
              7 8   9              <- level 3

        BFS order: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        Output: [u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9]]
              = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4]

    Why hard: Requires understanding graph construction from array encoding,
    then simulating a queue-based traversal algorithm. Different from DFS,
    so the model must implement this specific traversal order.
    """

    def __init__(self, T: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        from collections import deque

        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return torch.tensor([], dtype=torch.long, device=input_seq.device)

        vals = [child_result[i].item() for i in range(L)]

        # Build adjacency list (parent → children)
        children = [[] for _ in range(L)]
        root = None

        for i in range(L):
            parent = vals[i] % L
            if parent == i:
                if root is None:
                    root = i  # First self-loop is root
            else:
                children[parent].append(i)

        # Default root if none found
        if root is None:
            root = 0

        # Sort children by index for deterministic order
        for i in range(L):
            children[i].sort()

        # BFS traversal
        order = []
        visited = set()
        queue = deque([root])

        while queue and len(order) < self.T:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            # Add children to queue in index order
            for child in children[node]:
                if child not in visited:
                    queue.append(child)

        # If we haven't collected enough, visit remaining nodes
        for i in range(L):
            if len(order) >= self.T:
                break
            if i not in visited:
                order.append(i)

        # Pad with zeros if needed
        while len(order) < self.T:
            order.append(0)

        # Output values at BFS positions
        outputs = [vals[order[t]] for t in range(self.T)]

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"TreeWalkBFS({self.T}, {self.child})"


class CellularAutomaton(DSLExpr):
    """CellularAutomaton operator - iterative parallel neighbor mixing.

    Apply a mod-10 neighbor rule over the array for `steps` iterations.
    Each iteration, every position updates simultaneously based on its
    left neighbor, itself, and its right neighbor (circular).

    Algorithm:
        For step = 1 to steps:
            For each i simultaneously:
                new[i] = (a * x[(i-1) % L] + b * x[i] + c * x[(i+1) % L]) % 10
            x = new
        Output: x (length L, same as child)

    Output length: same as child

    Parameters:
        steps: Number of iterations (1-3 recommended)
        a: Left neighbor coefficient
        b: Self coefficient
        c: Right neighbor coefficient
        child: Child expression providing input sequence

    Why hard: the model must infer the neighbor rule AND apply it for the iteration count.
    After 2+ steps, each output depends on a widening cone of input positions.
    """

    def __init__(self, steps: int, a: int, b: int, c: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.steps = steps
        self.a = a % 10
        self.b = b % 10
        self.c = c % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        x = self.child.eval(input_seq)
        L = len(x)
        if L == 0:
            return x

        vals = [x[i].item() for i in range(L)]

        for _ in range(self.steps):
            new_vals = []
            for i in range(L):
                left = vals[(i - 1) % L]
                center = vals[i]
                right = vals[(i + 1) % L]
                new_vals.append((self.a * left + self.b * center + self.c * right) % 10)
            vals = new_vals

        return torch.tensor(vals, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"CellularAutomaton({self.steps}, {self.a}, {self.b}, {self.c}, {self.child})"


class LFSR(DSLExpr):
    """LFSR operator - linear feedback shift register trace.

    Initialize a shift register from the first `width` values of the child,
    then generate T output values by tapping specific register positions
    and feeding back a linear combination.

    Algorithm:
        reg = [u[0], u[1], ..., u[width-1]]
        For t = 0 to T-1:
            y[t] = reg[0]
            feedback = (c1 * reg[tap1] + c2 * reg[tap2]) % 10
            shift reg left by 1, append feedback
        Output: y[0..T-1]

    Output length: T (fixed)

    Parameters:
        T: Output length
        width: Register width (2-5)
        tap1: First tap position
        tap2: Second tap position
        c1: First tap coefficient
        c2: Second tap coefficient
        child: Child expression providing input sequence
    """

    def __init__(self, T: int, width: int, tap1: int, tap2: int, c1: int, c2: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T
        self.width = width
        self.tap1 = tap1 % width
        self.tap2 = tap2 % width
        self.c1 = c1 % 10
        self.c2 = c2 % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return torch.tensor([], dtype=torch.long, device=input_seq.device)

        # Initialize register from child output
        reg = [child_result[i % L].item() for i in range(self.width)]

        outputs = []
        for t in range(self.T):
            outputs.append(reg[0])
            feedback = (self.c1 * reg[self.tap1] + self.c2 * reg[self.tap2]) % 10
            reg = reg[1:] + [feedback]

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"LFSR({self.T}, {self.width}, {self.tap1}, {self.tap2}, {self.c1}, {self.c2}, {self.child})"


class ParityInterleave(DSLExpr):
    """ParityInterleave operator - split even/odd positions, apply different transforms.

    Split input into even-indexed and odd-indexed subsequences,
    apply different affine transforms to each, then interleave back.

    Algorithm:
        For each position i:
            if i is even: y[i] = (a * u[i] + b) % 10
            if i is odd:  y[i] = (c * u[i] + d) % 10
        Output: y (length L, same as child)

    Output length: same as child

    Parameters:
        a: Even-position multiplier
        b: Even-position offset
        c: Odd-position multiplier
        d: Odd-position offset
        child: Child expression providing input sequence
    """

    def __init__(self, a: int, b: int, c: int, d: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.a = a % 10
        self.b = b % 10
        self.c = c % 10
        self.d = d % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        for i in range(len(child_result)):
            v = child_result[i].item()
            if i % 2 == 0:
                outputs.append((self.a * v + self.b) % 10)
            else:
                outputs.append((self.c * v + self.d) % 10)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"ParityInterleave({self.a}, {self.b}, {self.c}, {self.d}, {self.child})"


class ReductionTree(DSLExpr):
    """ReductionTree operator - binary tournament bottom-up aggregation.

    Build a binary reduction tree over the input. Each level combines
    adjacent pairs using a weighted sum. Output is the concatenation
    of all reduction levels (excluding the input), truncated to T.

    Algorithm:
        level = child output
        result = []
        While len(level) > 1:
            new_level = []
            for j in 0..len(level)-1 step 2:
                if j+1 < len(level):
                    new_level.append((a * level[j] + b * level[j+1]) % 10)
                else:
                    new_level.append(level[j])
            result.extend(new_level)
            level = new_level
        Output: first T values of result

    Output length: T (fixed)

    Parameters:
        T: Output length
        a: Left element coefficient
        b: Right element coefficient
        child: Child expression providing input sequence
    """

    def __init__(self, T: int, a: int, b: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T
        self.a = a % 10
        self.b = b % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return torch.tensor([], dtype=torch.long, device=input_seq.device)

        level = [child_result[i].item() for i in range(L)]
        result = []

        while len(level) > 1 and len(result) < self.T:
            new_level = []
            for j in range(0, len(level), 2):
                if j + 1 < len(level):
                    new_level.append((self.a * level[j] + self.b * level[j + 1]) % 10)
                else:
                    new_level.append(level[j])
            result.extend(new_level)
            level = new_level

        # Truncate or pad
        result = result[:self.T]
        while len(result) < self.T:
            result.append(0)

        return torch.tensor(result, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"ReductionTree({self.T}, {self.a}, {self.b}, {self.child})"


class ChainedIndex(DSLExpr):
    """ChainedIndex operator - multi-level pointer indirection.

    For each output position, start at that index and dereference
    through the input values `depth` times.

    Algorithm:
        For t = 0 to T-1:
            idx = t % L
            For d = 0 to depth-1:
                idx = u[idx] % L
            y[t] = u[idx]
        Output: y[0..T-1]

    Output length: T (fixed)

    Parameters:
        T: Output length
        depth: Number of indirection levels (1-3)
        child: Child expression providing input sequence
    """

    def __init__(self, T: int, depth: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T
        self.depth = depth
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return torch.tensor([], dtype=torch.long, device=input_seq.device)

        outputs = []
        for t in range(self.T):
            idx = t % L
            for _ in range(self.depth):
                idx = child_result[idx].item() % L
            outputs.append(child_result[idx].item())

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"ChainedIndex({self.T}, {self.depth}, {self.child})"


class SelfConvolve(DSLExpr):
    """SelfConvolve operator - circular autocorrelation.

    Each output position sums the products of all pairs separated
    by that lag. Creates quadratic dependencies on the input.

    Algorithm:
        For t = 0 to T-1:
            y[t] = (sum over i=0..L-1 of u[i] * u[(i+t) % L]) % 10
        Output: y[0..T-1]

    Output length: T (fixed)

    Parameters:
        T: Output length
        child: Child expression providing input sequence
    """

    def __init__(self, T: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return torch.tensor([], dtype=torch.long, device=input_seq.device)

        vals = [child_result[i].item() for i in range(L)]
        outputs = []

        for t in range(self.T):
            total = 0
            for i in range(L):
                total += vals[i] * vals[(i + t) % L]
            outputs.append(total % 10)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"SelfConvolve({self.T}, {self.child})"


class AccumulateReset(DSLExpr):
    """AccumulateReset operator - running sum with threshold reset.

    Like PrefixSumMod but resets when accumulator exceeds a threshold.

    Algorithm:
        acc = 0
        For i = 0 to L-1:
            acc = acc + u[i]
            if acc >= threshold:
                acc = acc % threshold
            y[i] = acc % 10
        Output: y (length L, same as child)

    Output length: same as child

    Parameters:
        threshold: Reset threshold (11-19 recommended for interesting behavior)
        child: Child expression providing input sequence
    """

    def __init__(self, threshold: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.threshold = threshold
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        acc = 0
        for i in range(len(child_result)):
            acc += child_result[i].item()
            if acc >= self.threshold:
                acc = acc % self.threshold
            outputs.append(acc % 10)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"AccumulateReset({self.threshold}, {self.child})"


class ScatterWrite(DSLExpr):
    """ScatterWrite operator - data-dependent write addressing.

    Scan input and write each value to a computed address in an output buffer.
    Later writes overwrite earlier ones. Unwritten positions remain 0.

    Algorithm:
        buf = [0] * T
        For i = 0 to L-1:
            addr = (a * u[i] + i) % T
            buf[addr] = u[i]
        Output: buf[0..T-1]

    Output length: T (fixed)

    Parameters:
        T: Output buffer size
        a: Address multiplier
        child: Child expression providing input sequence
    """

    def __init__(self, T: int, a: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T
        self.a = a % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)

        buf = [0] * self.T

        for i in range(L):
            v = child_result[i].item()
            addr = (self.a * v + i) % self.T
            buf[addr] = v

        return torch.tensor(buf, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"ScatterWrite({self.T}, {self.a}, {self.child})"


class InverseLookup(DSLExpr):
    """InverseLookup operator - value-to-position inversion.

    For each target digit d, find the first position in the input where
    that digit appears. Outputs positions, not values.

    Algorithm:
        For d = 0 to T-1:
            y[d] = min({i : u[i] == d}, default=9)
        Output: y[0..T-1]

    Output length: T (fixed)

    Parameters:
        T: Output length (number of digits to look up, typically 10)
        child: Child expression providing input sequence
    """

    def __init__(self, T: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)

        # Build first-occurrence map
        first_pos = {}
        for i in range(L):
            v = child_result[i].item()
            if v not in first_pos:
                first_pos[v] = i

        outputs = []
        for d in range(self.T):
            outputs.append(first_pos.get(d, 9) % 10)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"InverseLookup({self.T}, {self.child})"


class RunLengthEncode(DSLExpr):
    """RunLengthEncode operator - run-length encoding of child output.

    Scan child output left-to-right, grouping consecutive identical digits
    into (digit, count) pairs. Flatten pairs into a sequence, truncate/pad
    to length T.

    Algorithm:
        pairs = []
        For each run of consecutive identical digits d with length c:
            pairs.append(d)
            pairs.append(c % 10)
        Output: first T elements of pairs (pad with 0 if short)

    Output length: T (fixed)

    Parameters:
        T: Output length
        child: Child expression providing input sequence

    Example with input [3, 3, 3, 1, 1, 9, 2, 2, 2, 2], T=8:
        Runs: 3x3, 1x2, 9x1, 2x4
        Pairs flattened: [3, 3, 1, 2, 9, 1, 2, 4]
        Output: [3, 3, 1, 2, 9, 1, 2, 4]

    Why hard: Output structure depends on runs of identical values, which is
    a pattern-matching concept, not arithmetic. The position of each output
    element depends on how many runs preceded it. The model must compute
    both the "what" (digit) and "how many" (count) components.
    """

    def __init__(self, T: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return torch.tensor([0] * self.T, dtype=torch.long, device=input_seq.device)

        vals = [child_result[i].item() for i in range(L)]

        # Build run-length encoding
        pairs = []
        i = 0
        while i < L:
            digit = vals[i]
            count = 1
            while i + count < L and vals[i + count] == digit:
                count += 1
            pairs.append(digit)
            pairs.append(count % 10)
            i += count

        # Truncate or pad to T
        result = pairs[:self.T]
        while len(result) < self.T:
            result.append(0)

        return torch.tensor(result, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"RunLengthEncode({self.T}, {self.child})"


class ChunkSort(DSLExpr):
    """ChunkSort operator - sort a single contiguous chunk of the sequence.

    Leaves elements outside [i, j) unchanged. Sorts elements in the range
    [i, j) in ascending or descending order. The chunk size j-i must be
    at most 6.

    Algorithm:
        y = copy of child output
        y[i:j] = sorted(y[i:j], ascending=asc)
        Output: y (length same as child)

    Output length: same as child

    Parameters:
        i: Start index of chunk (inclusive)
        j: End index of chunk (exclusive), j-i <= 6
        asc: If True, sort ascending; if False, sort descending
        child: Child expression providing input sequence

    Example with i=2, j=6, asc=True, input [5, 8, 7, 3, 9, 1, 4, 2, 6, 0]:
        Chunk [7, 3, 9, 1] sorted ascending -> [1, 3, 7, 9]
        Output: [5, 8, 1, 3, 7, 9, 4, 2, 6, 0]

    Why hard: Sorting is comparison-based, not arithmetic: no (a*x + b) % 10
    formula reproduces it. The model must locate the specific contiguous
    region being reordered by value, while the rest is unchanged. The
    boundaries i,j add extra parameters.
    """

    def __init__(self, i: int, j: int, asc: bool, child: DSLExpr):
        super().__init__(child.input_length)
        self.i = i
        self.j = j
        self.asc = asc
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return child_result

        vals = [child_result[k].item() for k in range(L)]

        # Clamp indices to valid range
        si = max(0, min(self.i, L))
        sj = max(si, min(self.j, L))

        # Sort the chunk
        chunk = vals[si:sj]
        chunk.sort(reverse=not self.asc)
        vals[si:sj] = chunk

        return torch.tensor(vals, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"ChunkSort({self.i}, {self.j}, {1 if self.asc else 0}, {self.child})"


class SegmentReverse(DSLExpr):
    """SegmentReverse operator - reverse segments delimited by a boundary digit.

    Scans the child output and splits it into segments at every occurrence
    of the boundary digit. Each segment between boundaries is reversed
    independently; boundary digits stay in place.

    Algorithm:
        x = child.eval(input)
        Split x at positions where x[i] == boundary_digit
        Reverse the elements within each segment (between boundaries)
        Boundary digits remain at their original positions
        Output: reassembled sequence (same length as child)

    Output length: same as child

    Parameters:
        boundary_digit: The digit (0-9) that marks segment boundaries
        child: Child expression providing input sequence

    Example with boundary_digit=5, input [3, 8, 2, 5, 7, 1, 9, 5, 4, 6]:
        Segments: [3,8,2] | 5 | [7,1,9] | 5 | [4,6]
        Reverse each segment: [2,8,3] | 5 | [9,1,7] | 5 | [6,4]
        Output: [2, 8, 3, 5, 9, 1, 7, 5, 6, 4]

    Example with boundary_digit=0, input [0, 3, 7, 2, 0, 8, 1, 0, 4, 6]:
        Segments: [] | 0 | [3,7,2] | 0 | [8,1] | 0 | [4,6]
        Reverse each segment: [] | 0 | [2,7,3] | 0 | [1,8] | 0 | [6,4]
        Output: [0, 2, 7, 3, 0, 1, 8, 0, 6, 4]

    Why hard: the model must detect the boundary digit, then apply local
    reversals within variable-length segments. The segment structure depends
    on where the boundary digit appears in the input, creating input-dependent
    structure. Requires both pattern recognition and local reversal logic.
    """

    def __init__(self, boundary_digit: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.boundary_digit = boundary_digit % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return child_result

        vals = [child_result[i].item() for i in range(L)]
        output = [0] * L

        # Find segment boundaries and reverse within each segment
        seg_start = 0
        for i in range(L):
            if vals[i] == self.boundary_digit:
                # Reverse the segment [seg_start, i)
                segment = vals[seg_start:i]
                segment.reverse()
                for j, v in enumerate(segment):
                    output[seg_start + j] = v
                # Place boundary digit
                output[i] = self.boundary_digit
                seg_start = i + 1

        # Reverse the final segment after last boundary
        segment = vals[seg_start:L]
        segment.reverse()
        for j, v in enumerate(segment):
            output[seg_start + j] = v

        return torch.tensor(output, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"SegmentReverse({self.boundary_digit}, {self.child})"


class XORScan(DSLExpr):
    """XORScan operator - cumulative bitwise XOR.

    Computes the running XOR of all elements seen so far.
    Operates in GF(2) on each bit independently, unlike PrefixSumMod (mod-10 addition).

    out[i] = (x[0] XOR x[1] XOR ... XOR x[i]) % 10
    Output length: same as child

    Example with input [3, 5, 6, 1, 4, 2, 9, 7, 8, 0]:
        acc=3, acc=3^5=6, acc=6^6=0, acc=0^1=1, acc=1^4=5, ...
        out = [3, 6, 0, 1, 5, 7, 14%10=4, 4^7=3, 3^8=11%10=1, 1^0=1]
        out = [3, 6, 0, 1, 5, 7, 4, 3, 1, 1]

    Why different from PrefixSumMod: XOR of 3 and 5 is 6 (binary: 011^101=110),
    while (3+5)%10=8. They diverge for most inputs.
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        outputs = []
        acc = 0
        for val in child_result.tolist():
            acc ^= val
            outputs.append(acc % 10)
        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"XORScan({self.child})"


class ModRemainder(DSLExpr):
    """ModRemainder operator - rolling base-10 value modulo m.

    Treats the growing prefix as a base-10 number and outputs its remainder mod m.
    Works in a different algebraic structure than HashScan (which is mod 10).

    state_i = (10 * state_{i-1} + x[i]) % m
    out[i] = state_i % 10
    Output length: same as child

    Parameters:
        m: Modulus (should be coprime to 10, e.g. 7, 11, 13, 17)
        child: Child expression

    Example with m=7, input [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]:
        state: 1, 12%7=5, 53%7=4, 44%7=2, 25%7=4, 46%7=4, 47%7=5, 58%7=2, 29%7=1, 10%7=3
        out = [1, 5, 4, 2, 4, 4, 5, 2, 1, 3]

    Why hard: The modulus m != 10 means wrap-around occurs at different points than
    any mod-10 operation. HashScan(10,1,0,0) looks identical but is mod 10 throughout.
    """

    def __init__(self, m: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.m = m
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        outputs = []
        state = 0
        for val in child_result.tolist():
            state = (10 * state + val) % self.m
            outputs.append(state % 10)
        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"ModRemainder({self.m}, {self.child})"


class ZigzagCount(DSLExpr):
    """ZigzagCount operator - cumulative count of strict local extrema.

    At each position i, counts how many positions j in {1,...,i-1} are strict
    local extrema (peaks or valleys). Position j is a local extremum if:
        x[j] > x[j-1] AND x[j] > x[j+1]  (peak), or
        x[j] < x[j-1] AND x[j] < x[j+1]  (valley)

    out[i] = count of local extrema in positions {1,...,i-1}, mod 10
    Output length: same as child

    Note: out[0] = out[1] = 0 since no position has both neighbors yet.
    Each new element x[i] resolves whether x[i-1] is a local extremum.

    Example with input [1, 5, 2, 8, 3, 9, 4, 7, 2, 6]:
        pos 1 (5): peak? needs x[2]=2. Yes, 5>1 and 5>2. count=1 at pos 2.
        pos 2 (2): valley? needs x[3]=8. Yes, 2<5 and 2<8. count=2 at pos 3.
        ... (each alternating high/low is an extremum)
        out = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    Connects to permutation statistics (zigzag permutations, Euler numbers).
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        vals = child_result.tolist()
        L = len(vals)
        outputs = []
        count = 0
        for i in range(L):
            # Seeing x[i] resolves whether x[i-1] is a local extremum
            if i >= 2:
                prev = vals[i - 1]
                if (prev > vals[i - 2] and prev > vals[i]) or \
                   (prev < vals[i - 2] and prev < vals[i]):
                    count += 1
            outputs.append(count % 10)
        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"ZigzagCount({self.child})"


class RecordCount(DSLExpr):
    """RecordCount operator - cumulative count of left-to-right maxima.

    Counts how many times a new maximum has been set by position i.
    A new maximum (record) at position j means x[j] > x[k] for all k < j.

    out[i] = count of record positions in x[0:i+1], mod 10
    Output length: same as child

    Note: x[0] is always a record (first element). out[0] = 1.

    Example with input [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]:
        Records at: pos 0 (3), pos 2 (4), pos 4 (5), pos 5 (9)
        out = [1, 1, 2, 2, 3, 4, 4, 4, 4, 4]

    Different from PrefixMax: PrefixMax outputs the VALUE of the max,
    RecordCount outputs HOW MANY TIMES the max changed. These diverge
    whenever the same max value is not a new record.

    Connects to: expected value H(n) records in a random permutation
    (harmonic numbers), coupon collector problem.
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        vals = child_result.tolist()
        outputs = []
        count = 0
        current_max = -1
        for val in vals:
            if val > current_max:
                count += 1
                current_max = val
            outputs.append(count % 10)
        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"RecordCount({self.child})"


class RunningProduct(DSLExpr):
    """RunningProduct operator - running product modulo 10.

    out[i] = (product of child[:i+1]) % 10
    Output length: same as child

    The multiplicative analog of PrefixSumMod. A single 0 zeroes the product
    thereafter; a lone non-1 factor in an otherwise-all-1 sequence makes the
    output a clean step (locate-and-broadcast that factor).

    Example with input [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]:
        out = [3, 3, 2, 2, 0, 0, 0, 0, 0, 0]
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        prod = 1
        for i in range(len(child_result)):
            prod = (prod * child_result[i].item()) % 10
            outputs.append(prod)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"RunningProduct({self.child})"


class CountOdd(DSLExpr):
    """CountOdd operator - running count of odd-valued digits (mod 10).

    out[i] = (number of odd-valued digits in child[:i+1]) % 10
    Output length: same as child

    The count analog of PrefixSumOdd and the parity analog of RunningCountDigit.
    With exactly one odd digit the output is a 0->1 step detecting the lone odd.

    Example with input [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]:
        odd flags: 1,1,0,1,1,1,0,0,1,1
        out = [1, 2, 2, 3, 4, 5, 5, 5, 6, 7]
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        count = 0
        for i in range(len(child_result)):
            if child_result[i].item() % 2 == 1:
                count += 1
            outputs.append(count % 10)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"CountOdd({self.child})"


class RecordIndicator(DSLExpr):
    """RecordIndicator operator - 0/1 flag at each new left-to-right maximum.

    out[i] = 1 if child[i] > every earlier element (a new record), else 0.
    Output length: same as child

    The indicator analog of RecordCount (which emits the running record COUNT).
    x[0] is always a record, so out[0] = 1.

    Example with input [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]:
        records at pos 0,2,4,5 -> out = [1, 0, 1, 0, 1, 1, 0, 0, 0, 0]
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        current_max = -1
        for val in child_result.tolist():
            if val > current_max:
                outputs.append(1)
                current_max = val
            else:
                outputs.append(0)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"RecordIndicator({self.child})"


class NoveltyIndicator(DSLExpr):
    """NoveltyIndicator operator - 0/1 flag at the first occurrence of each value.

    out[i] = 1 if child[i] has NOT appeared in child[:i] (novel), else 0 (repeat).
    Output length: same as child

    With exactly one repeated value the output has a single 0 that localizes the
    repeat (a novelty / seen-before detector).

    Example with input [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]:
        out = [1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        seen = set()
        for val in child_result.tolist():
            if val in seen:
                outputs.append(0)
            else:
                outputs.append(1)
                seen.add(val)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"NoveltyIndicator({self.child})"


class FixedPointIndicator(DSLExpr):
    """FixedPointIndicator operator - 0/1 flag where value equals its index.

    out[i] = 1 if child[i] == i, else 0. Output length: same as child

    A value/position coincidence detector. With exactly one fixed point the
    output is a single 1.

    Example with input [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]:
        x[1]==1 only -> out = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = [1 if child_result[i].item() == i else 0
                   for i in range(len(child_result))]
        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"FixedPointIndicator({self.child})"


class ResetRamp(DSLExpr):
    """ResetRamp operator - positional distance-since-last-trigger ramp.

    out[i] = (i - last_index_j<=i_where_child[j]==d) % 10, with the 'last
    reset' position initialized to 0. A multi-reset positional sawtooth:
    0,1,2,... that resets to 0 at every occurrence of token d. Counts
    POSITIONS, not the token. Output length: same as child.

    With exactly one d it collapses to a single-reset ramp (distance since the
    lone event). Example with d=5, input [2,1,5,3,4,5,8,7,6,0]:
        resets at pos 2,5 -> out = [0,1,0,1,2,0,1,2,3,4]
    """

    def __init__(self, d: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.d = d % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        last = 0
        for i in range(len(child_result)):
            if child_result[i].item() == self.d:
                last = i
            outputs.append((i - last) % 10)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"ResetRamp({self.d}, {self.child})"


class TokenResetMax(DSLExpr):
    """TokenResetMax operator - running maximum that resets to 0 on token d.

    Running max, but every occurrence of token d resets the accumulator to 0
    (the d itself contributes 0). Output length: same as child.

    Excluding d (no reset fires) collapses it to a plain running maximum.
    Example with d=9, input [3,1,4,9,5,2,6,1,8,0]:
        out = [3,3,4,0,5,5,6,6,8,8]
    """

    def __init__(self, d: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.d = d % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        m = 0
        for v in child_result.tolist():
            if v == self.d:
                m = 0
            elif v > m:
                m = v
            outputs.append(m)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"TokenResetMax({self.d}, {self.child})"


class SignedWalk(DSLExpr):
    """SignedWalk operator - signed bidirectional counter (Dyck-style walk).

    +1 on the up-token, -1 on the down-token, unchanged on everything else;
    emits the running value (mod 10). Output length: same as child.

    Excluding the down-token gives a plain up-counter; excluding the up-token
    gives a decreasing counter. Example up=1,down=2, input [1,1,3,2,1,2,2,4,1,1]:
        running: 1,2,2,1,2,1,0,0,1,2 -> out = [1,2,2,1,2,1,0,0,1,2]
    """

    def __init__(self, up: int, down: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.up = up % 10
        self.down = down % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        c = 0
        for v in child_result.tolist():
            if v == self.up:
                c += 1
            elif v == self.down:
                c -= 1
            outputs.append(c % 10)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"SignedWalk({self.up}, {self.down}, {self.child})"


class SRLatch(DSLExpr):
    """SRLatch operator - 1-bit set/reset memory cell.

    The set-token drives the state to 1, the reset-token drives it to 0, every
    other token holds; emits the state (0/1). Output length: same as child.

    Excluding the reset-token collapses it to cum_or(==set): latch on at the
    first set. Example set=1,reset=2, input [3,1,4,1,2,1,5,2,1,0]:
        out = [0,1,1,1,0,1,1,0,1,1]
    """

    def __init__(self, set_tok: int, reset_tok: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.set_tok = set_tok % 10
        self.reset_tok = reset_tok % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        state = 0
        for v in child_result.tolist():
            if v == self.set_tok:
                state = 1
            elif v == self.reset_tok:
                state = 0
            outputs.append(state)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"SRLatch({self.set_tok}, {self.reset_tok}, {self.child})"


class SampleHold(DSLExpr):
    """SampleHold operator - threshold-gated sample-and-hold latch register.

    Holds the most-recent digit that passes a threshold gate (>=thresh for
    mode 'ge', <=thresh for mode 'le'); a non-qualifying digit holds the
    register's previous value (initial value 0). Output length: same as child.

    With exactly one qualifier the latch fires once -> a single step to that
    digit. Example mode='ge',thresh=5, input [2,1,7,3,4,2,8,1,0,3]:
        qualifiers 7,8 -> out = [0,0,7,7,7,7,8,8,8,8]
    """

    def __init__(self, mode: str, thresh: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.mode = 'ge' if mode == 'ge' else 'le'
        self.thresh = thresh % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        reg = 0
        for v in child_result.tolist():
            passes = (v >= self.thresh) if self.mode == 'ge' else (v <= self.thresh)
            if passes:
                reg = v
            outputs.append(reg)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"SampleHold('{self.mode}', {self.thresh}, {self.child})"


class Deplete(DSLExpr):
    """Deplete operator - deplete-to-floor countdown accumulator.

    Starts the accumulator at 'start', subtracts each digit, saturating at
    'floor' (never below it); emits the accumulator. A decreasing saturating
    accumulator. Output length: same as child.

    With a bounded total (sum <= start - floor) the floor never binds and it
    collapses to start - cumsum (a decreasing ramp). Example start=9,floor=0,
    input [2,1,3,0,4,1,2,5,3,1]:
        out = [7,6,3,3,0,0,0,0,0,0]
    """

    def __init__(self, start: int, floor: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.start = start % 10
        self.floor = floor % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result

        outputs = []
        acc = self.start
        for v in child_result.tolist():
            acc = max(self.floor, acc - v)
            outputs.append(acc % 10)

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"Deplete({self.start}, {self.floor}, {self.child})"


class AnchorSubtract(DSLExpr):
    """AnchorSubtract operator - broadcast subtract of a reference position.

    out[i] = (child[i] - child[a]) mod 10, where a is a fixed anchor position.
    Mixes one reference position into every output. Output length: same as child.

    Pinning the anchor value child[a]=c collapses it to a Caesar shift by -c.
    Example a=0, input [3,1,4,1,5,9,2,6,5,3] (anchor 3):
        out = [0, 8, 1, 8, 2, 6, 9, 3, 2, 0]
    """

    def __init__(self, a: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.a = a % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return child_result
        vals = child_result.tolist()
        anchor = vals[self.a % L]
        out = [(v - anchor) % 10 for v in vals]
        return torch.tensor(out, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"AnchorSubtract({self.a}, {self.child})"


class AnchorClamp(DSLExpr):
    """AnchorClamp operator - clamp every element against a reference position.

    out[i] = max(child[i], child[a]) for mode 'floor', or min(child[i], child[a])
    for mode 'ceil', where a is a fixed anchor position (the clamp bound is read
    from the data). Output length: same as child.

    Pinning child[a]=c collapses it to a constant floor/ceil clamp at c.
    Example mode='floor', a=0, input [3,1,4,1,5,9,2,6,5,3] (bound 3):
        out = [3, 3, 4, 3, 5, 9, 3, 6, 5, 3]
    """

    def __init__(self, mode: str, a: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.mode = 'floor' if mode == 'floor' else 'ceil'
        self.a = a % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return child_result
        vals = child_result.tolist()
        bound = vals[self.a % L]
        if self.mode == 'floor':
            out = [max(v, bound) for v in vals]
        else:
            out = [min(v, bound) for v in vals]
        return torch.tensor(out, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"AnchorClamp('{self.mode}', {self.a}, {self.child})"


class DataRotate(DSLExpr):
    """DataRotate operator - cyclic rotation by a data-dependent amount.

    out[i] = child[(i + child[a]) mod L], where the rotation amount is read from
    a fixed anchor position a. Output length: same as child.

    Pinning child[a]=c collapses it to a fixed cyclic left-rotation by c.
    Example a=0, input [3,1,4,1,5,9,2,6,5,3] (rotate by 3):
        out = [1, 5, 9, 2, 6, 5, 3, 3, 1, 4]
    """

    def __init__(self, a: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.a = a % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return child_result
        vals = child_result.tolist()
        r = vals[self.a % L] % L
        out = [vals[(i + r) % L] for i in range(L)]
        return torch.tensor(out, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"DataRotate({self.a}, {self.child})"


class DataReflect(DSLExpr):
    """DataReflect operator - reflection about a data-dependent pivot.

    out[i] = child[(2*child[a] - i) mod L], an involution whose mirror centre is
    read from a fixed anchor position a. Output length: same as child.

    Pinning child[a]=c collapses it to a fixed mirror about pivot c.
    Example a=0, input [3,1,4,1,5,9,2,6,5,3] (pivot 3):
        out = [2, 9, 5, 1, 4, 1, 3, 3, 5, 6]
    """

    def __init__(self, a: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.a = a % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return child_result
        vals = child_result.tolist()
        p = vals[self.a % L]
        out = [vals[(2 * p - i) % L] for i in range(L)]
        return torch.tensor(out, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"DataReflect({self.a}, {self.child})"


class StrideGather(DSLExpr):
    """StrideGather operator - strided gather with a data-dependent stride.

    out[i] = child[(i * child[a]) mod L], a multiplicative-index gather whose
    stride is read from a fixed anchor position a. Output length: same as child.

    Pinning child[a]=k collapses it to a fixed stride-k gather.
    Example a=0, input [3,1,4,1,5,9,2,6,5,3] (stride 3):
        out = [3, 1, 2, 3, 4, 9, 5, 1, 5, 6]
    """

    def __init__(self, a: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.a = a % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return child_result
        vals = child_result.tolist()
        k = vals[self.a % L]
        out = [vals[(i * k) % L] for i in range(L)]
        return torch.tensor(out, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"StrideGather({self.a}, {self.child})"


class PrefixReverseK(DSLExpr):
    """PrefixReverseK operator - reverse a data-dependent prefix length.

    out = reverse(child[:k]) ++ child[k:], where k = child[a] (clamped to [0,L])
    is read from a fixed anchor position a. Output length: same as child.

    Pinning child[a]=k collapses it to a fixed length-k prefix reversal.
    Example a=0, input [3,1,4,1,5,9,2,6,5,3] (k=3):
        out = [4, 1, 3, 1, 5, 9, 2, 6, 5, 3]
    """

    def __init__(self, a: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.a = a % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return child_result
        vals = child_result.tolist()
        k = max(0, min(L, vals[self.a % L]))
        out = vals[:k][::-1] + vals[k:]
        return torch.tensor(out, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"PrefixReverseK({self.a}, {self.child})"


class PrefixSortK(DSLExpr):
    """PrefixSortK operator - sort a data-dependent prefix length (ascending).

    out = sorted(child[:k]) ++ child[k:], where k = child[a] (clamped to [0,L])
    is read from a fixed anchor position a. A genuine ordering primitive (the
    output order depends on the values). Output length: same as child.

    Pinning child[a]=k collapses it to a fixed length-k prefix sort.
    Example a=0, input [3,1,4,1,5,9,2,6,5,3] (k=3):
        out = [1, 3, 4, 1, 5, 9, 2, 6, 5, 3]
    """

    def __init__(self, a: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.a = a % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return child_result
        vals = child_result.tolist()
        k = max(0, min(L, vals[self.a % L]))
        out = sorted(vals[:k]) + vals[k:]
        return torch.tensor(out, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"PrefixSortK({self.a}, {self.child})"


class CompareSwap(DSLExpr):
    """CompareSwap operator - conditional compare-and-swap of two fixed positions.

    A single sorting-network comparator on positions (i, j): if child[i] > child[j]
    swap their contents, else leave them; all other positions pass through.
    Output length: same as child.

    Forcing child[i] > child[j] on the domain collapses it to an unconditional
    swap of positions i and j. Example (i,j)=(0,9), input [7,1,4,1,5,9,2,6,5,0]:
        7 > 0 -> swap -> out = [0, 1, 4, 1, 5, 9, 2, 6, 5, 7]
    """

    def __init__(self, i: int, j: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.i = i % 10
        self.j = j % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return child_result
        out = child_result.tolist()
        i, j = self.i % L, self.j % L
        if out[i] > out[j]:
            out[i], out[j] = out[j], out[i]
        return torch.tensor(out, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"CompareSwap({self.i}, {self.j}, {self.child})"


class WindowMax(DSLExpr):
    """WindowMax operator - sliding (backward) window maximum.

    out[i] = max(child[max(0,i-w) .. i]) - the max over the current element and
    the w preceding ones (edge-replicated at the start). Output length: same as
    child.

    On a non-increasing input the window max is the earliest element, so the
    model collapses to a temporal shift-right-by-w (a delay line).
    Example w=1, input [3,1,4,1,5,9,2,6,5,3]:
        out = [3, 3, 4, 4, 5, 9, 9, 6, 6, 5]
    """

    def __init__(self, w: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.w = max(1, w % 10)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return child_result
        vals = child_result.tolist()
        out = [max(vals[max(0, i - self.w): i + 1]) for i in range(L)]
        return torch.tensor(out, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"WindowMax({self.w}, {self.child})"


class ParityRoute(DSLExpr):
    """ParityRoute operator - parity-routed +-1 map.

    out[i] = (child[i] + 1) % 10 if child[i] is even, else (child[i] - 1) % 10.
    The operation applied is routed by each digit's parity. Output length: same
    as child.

    Restricting to all-even (or all-odd) input exercises only one branch -> a
    single uniform +1 (or -1) map. Example input [3,1,4,1,5,9,2,6,5,3]:
        out = [2, 0, 5, 0, 4, 8, 3, 7, 4, 2]
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        if len(child_result) == 0:
            return child_result
        out = []
        for v in child_result.tolist():
            out.append((v + 1) % 10 if v % 2 == 0 else (v - 1) % 10)
        return torch.tensor(out, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"ParityRoute({self.child})"


class RippleCarry(DSLExpr):
    """RippleCarry operator - multi-digit ripple-carry add of k at the last digit.

    Treats the sequence as a number with the least-significant digit LAST; adds k
    to that digit and propagates the carry leftward. Output length: same as child.

    When the seeded digit can't overflow (no carry) it collapses to a purely
    local +k at the last position. Example k=1, input [3,1,4,1,5,9,2,6,5,9]:
        9+1 carries -> out = [3,1,4,1,5,9,2,6,6,0]
    """

    def __init__(self, k: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.k = k % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return child_result
        out = child_result.tolist()
        carry = self.k
        i = L - 1
        while carry and i >= 0:
            s = out[i] + carry
            out[i] = s % 10
            carry = s // 10
            i -= 1
        return torch.tensor(out, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"RippleCarry({self.k}, {self.child})"


class ArgMinMax(DSLExpr):
    """ArgMinMax operator - concatenated argmax and argmin indicators.

    Emits a length-2L output: the first L positions are the argmax indicator
    (1 at the first occurrence of the maximum value, else 0); the next L are the
    argmin indicator (1 at the first occurrence of the minimum value, else 0).
    Output length: 2 * child length.

    Scoring only one output window (K5) isolates the argmax- or argmin-indicator
    circuit. Example input [3,1,4,1,5,9,2,6,5,3] (max 9@5, min 1@1):
        out = [0,0,0,0,0,1,0,0,0,0,  0,1,0,0,0,0,0,0,0,0]
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        L = len(child_result)
        if L == 0:
            return child_result
        vals = child_result.tolist()
        imax = vals.index(max(vals))
        imin = vals.index(min(vals))
        out = [1 if i == imax else 0 for i in range(L)] + \
              [1 if i == imin else 0 for i in range(L)]
        return torch.tensor(out, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return 2 * self.child.output_length()

    def __str__(self) -> str:
        return f"ArgMinMax({self.child})"


class KMPFailure(DSLExpr):
    """KMPFailure operator - KMP failure function (longest proper prefix = suffix).

    out[i] = length of the longest proper prefix of x[0:i+1] that is also
             a suffix of x[0:i+1], mod 10.
    A "proper" prefix/suffix excludes the full string itself.
    Output length: same as child

    Example with input [3, 1, 3, 1, 3, 1, 2, 6, 5, 3]:
        out[0]=0 (no proper prefix)
        out[1]=0 ("31": no prefix of "3" equals suffix)
        out[2]=1 ("313": prefix "3" == suffix "3")
        out[3]=2 ("3131": prefix "31" == suffix "31")
        out[4]=3, out[5]=4 (growing match)
        out[6]=0 (mismatch at "2")
        out = [0, 0, 1, 2, 3, 4, 0, 0, 0, 1]

    For "3131313131" (period-2): out = [0,0,1,2,3,4,5,6,7,8]
    For "1234567890" (all distinct): out = [0,0,0,0,0,0,0,0,0,0]

    Uses the standard O(n) KMP failure function algorithm.
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        vals = child_result.tolist()
        L = len(vals)
        if L == 0:
            return child_result

        fail = [0] * L
        k = 0
        for i in range(1, L):
            while k > 0 and vals[i] != vals[k]:
                k = fail[k - 1]
            if vals[i] == vals[k]:
                k += 1
            fail[i] = k

        outputs = [f % 10 for f in fail]
        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"KMPFailure({self.child})"


class BWT(DSLExpr):
    """BWT operator - Burrows-Wheeler Transform.

    Sorts all cyclic rotations of the input lexicographically, then outputs
    the last character of each sorted rotation.

    Algorithm:
        1. Form all L cyclic rotations: rot[i] = x[i:] + x[:i]
        2. Sort the rotations lexicographically
        3. out[k] = last character of k-th sorted rotation = x[(sorted_start[k] - 1) % L]

    Output length: same as child (always L)

    Example with input [1, 2, 3, 1, 2, 3, 1, 2, 3, 1] (periodic "1231231231"):
        Sorted rotations: those starting with "11...", "12...", "13...",
        "21...", "22...", "23...", "31...", "32...", "33..."
        out = [3, 3, 3, 1, 1, 1, 1, 2, 2, 2]  (characters cluster by predecessor)

    Key property: identical characters cluster together in the output, since
    the BWT groups rotations by what precedes each symbol. Useful for compression
    (bzip2), reversible with the first-column array.

    Why hard: requires understanding global sort order of all rotations. The
    output is a permutation of the input, but with a non-local permutation rule.
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        vals = child_result.tolist()
        L = len(vals)
        if L == 0:
            return child_result

        # Sort rotation start indices by their rotation (lex order)
        sorted_starts = sorted(range(L), key=lambda i: vals[i:] + vals[:i])
        # Last char of rotation starting at i is vals[(i - 1 + L) % L]
        outputs = [vals[(i - 1 + L) % L] for i in sorted_starts]
        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"BWT({self.child})"


class LZ78Complexity(DSLExpr):
    """LZ78Complexity operator - cumulative LZ78 phrase count.

    Applies the LZ78 parsing algorithm and outputs the running phrase count.
    LZ78 greedily extends the current phrase until it is new (unseen before),
    then starts a fresh phrase.

    out[i] = number of complete LZ78 phrases emitted after parsing x[0:i+1], mod 10
    Output length: same as child

    Example with input [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (all same, maximally compressible):
        Phrases: "1" (new), then "11" (new), then "111" (new), then "1111" (new)
        out = [1, 1, 2, 2, 2, 3, 3, 3, 3, 4]  (logarithmic growth)

    Example with input [1, 2, 3, 4, 5, 6, 7, 8, 9, 0] (all distinct, incompressible):
        Every single character is a new phrase immediately.
        out = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]  (linear growth, mod 10)

    The growth rate of the phrase count measures compressibility:
    slow growth = repetitive/compressible, fast growth = random/incompressible.
    This is the Lempel-Ziv complexity, the accepted algorithmic measure of randomness.
    """

    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length)
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        vals = child_result.tolist()
        outputs = []
        dictionary = set()
        current_phrase = []
        count = 0
        for val in vals:
            current_phrase.append(val)
            phrase_key = tuple(current_phrase)
            if phrase_key not in dictionary:
                dictionary.add(phrase_key)
                count += 1
                current_phrase = []
            outputs.append(count % 10)
        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"LZ78Complexity({self.child})"


class VanEckExtension(DSLExpr):
    """VanEckExtension operator - continue Van Eck's sequence from input seed.

    Uses the input as the first L terms of a sequence, then generates T more
    terms using Van Eck's rule: each new term equals 0 if the previous term
    is appearing for the first time, otherwise equals the gap (distance) to
    the previous occurrence of that value.

    Algorithm:
        history = x[0..L-1]  (the input, positions 0..L-1)
        last_seen = {x[i]: i for i in range(L-1)}  (positions 0..L-2 only)
        prev_pos = L-1,  prev_val = x[L-1]
        For t = 0..T-1:
            if prev_val in last_seen:
                y[t] = (prev_pos - last_seen[prev_val]) % 10
            else:
                y[t] = 0
            last_seen[prev_val] = prev_pos
            prev_pos = L + t
            prev_val = y[t]

    Output length: T (fixed, less than input_length)

    Example with input [5, 7, 4, 8, 3, 9, 2, 0, 1, 6]:
        prev_val=6, not in last_seen(5,7,4,8,3,9,2,0,1) → y[0]=0
        prev_val=0, last seen at pos 7 → gap=10-7=3 → y[1]=3
        prev_val=3, last seen at pos 4 → gap=11-4=7 → y[2]=7
        out = [0, 3, 7, 1, 5, 4, 3, 5, 3, 2]

    Self-referential: each output depends on when values previously appeared,
    including in the generated outputs themselves. Requires full history tracking.
    """

    def __init__(self, T: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.T = T
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        seed = child_result.tolist()
        L = len(seed)
        if L == 0:
            return torch.tensor([], dtype=torch.long, device=input_seq.device)

        # Build last_seen from seed[0..L-2] (exclude last element - it's the "current prev")
        last_seen = {}
        for i in range(L - 1):
            last_seen[seed[i]] = i

        prev_pos = L - 1
        prev_val = seed[L - 1]

        outputs = []
        for t in range(self.T):
            if prev_val in last_seen:
                gap = (prev_pos - last_seen[prev_val]) % 10
            else:
                gap = 0
            outputs.append(gap)
            # Update: prev_val was at prev_pos
            last_seen[prev_val] = prev_pos
            prev_pos = L + t
            prev_val = gap

        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.T

    def __str__(self) -> str:
        return f"VanEckExtension({self.T}, {self.child})"


class SwitchedScan(DSLExpr):
    """SwitchedScan operator - state-value-gated switching between two linear recurrences.

    At each step, applies one of two linear update rules based on whether the
    current state is below or above a threshold. Unlike HashScan (fixed coefficients),
    the recurrence itself changes based on the running state.

    Algorithm:
        s = 0
        For each x[i]:
            if s < thresh:
                s = (a1 * s + b1 * x[i]) % 10   (low-state regime)
            else:
                s = (a2 * s + b2 * x[i]) % 10   (high-state regime)
            y[i] = s

    Output length: same as child

    Parameters:
        a1: State multiplier in low regime (s < thresh)
        b1: Input multiplier in low regime
        a2: State multiplier in high regime (s >= thresh)
        b2: Input multiplier in high regime
        thresh: Switching threshold (0-9)
        child: Child expression

    Example with a1=3, b1=1, a2=7, b2=3, thresh=5, input [1,2,3,4,5,6,7,8,9,0]:
        s=0<5: s=(3*0+1*1)%10=1; s=1<5: s=(3*1+1*2)%10=5;
        s=5>=5: s=(7*5+3*3)%10=4; s=4<5: s=(3*4+1*4)%10=6; ...
        out = [1, 5, 4, 6, 7, 7, 0, 8, 3, 9]

    Why hard: the regime-switching creates piecewise-linear dynamics that cannot
    be reproduced by any single HashScan (which has fixed coefficients).
    """

    def __init__(self, a1: int, b1: int, a2: int, b2: int, thresh: int, child: DSLExpr):
        super().__init__(child.input_length)
        self.a1 = a1 % 10
        self.b1 = b1 % 10
        self.a2 = a2 % 10
        self.b2 = b2 % 10
        self.thresh = thresh % 10
        self.child = child

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        child_result = self.child.eval(input_seq)
        s = 0
        outputs = []
        for val in child_result.tolist():
            if s < self.thresh:
                s = (self.a1 * s + self.b1 * val) % 10
            else:
                s = (self.a2 * s + self.b2 * val) % 10
            outputs.append(s)
        return torch.tensor(outputs, dtype=torch.long, device=input_seq.device)

    def output_length(self) -> int:
        return self.child.output_length()

    def __str__(self) -> str:
        return f"SwitchedScan({self.a1}, {self.b1}, {self.a2}, {self.b2}, {self.thresh}, {self.child})"




def _promoted_apply(fn, t):
    """Run a brainstorm reference algo on a tensor (refs may want a list or a tensor)."""
    try:
        y = fn(t.tolist())
    except (AttributeError, TypeError):
        y = fn(t)
    return y.tolist() if hasattr(y, "tolist") else list(y)

def _ref_sarank(x):                                # inverse suffix array (suffix ranks)
    n=len(x); order=sorted(range(n), key=lambda i: x[i:]); rank=[0]*n
    for r,i in enumerate(order): rank[i]=r
    return [rank[i]%10 for i in range(n)]


class Sarank(DSLExpr):
    """inverse suffix array (suffix ranks) (brainstorm-promoted)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_sarank, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Sarank({self.child})"


class TaskMux(DSLExpr):
    """Multi-task mux (N3-breadth rung 1): input = [task_token, payload x 10] (length 11).
    Task i uses token 10+i (above the digit vocab, so vocab_size = 10+N); output is
    members[i] applied to the payload. Members must map length 10 -> 10 so the output
    length is uniform. id grammar: ONE underscore-free token ``mux<N>.<m1>.<m2>...``
    where each <mi> is member i's algorithm_id with '_' replaced by '-'
    (e.g. ``mux3.rev-id.dblodd-id.psum-id``); see docs/reference/design.md."""

    def __init__(self, members: list[DSLExpr]):
        if len(members) < 2:
            raise ValueError(f"TaskMux needs >= 2 members (got {len(members)})")
        for i, m in enumerate(members):
            if m.input_length != 10 or m.output_length() != 10:
                raise ValueError(f"TaskMux member {i} ({m}) must map length 10 -> 10 "
                                 f"(got {m.input_length} -> {m.output_length()})")
        super().__init__(11)
        self.members = list(members)

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        if input_seq.shape[0] != self.input_length:
            raise ValueError(f"TaskMux input must have length {self.input_length} "
                             f"(got {input_seq.shape[0]})")
        task = int(input_seq[0]) - 10
        if not 0 <= task < len(self.members):
            raise ValueError(f"task token {int(input_seq[0])} out of range "
                             f"[10, {10 + len(self.members)})")
        return self.members[task].eval(input_seq[1:])

    def output_length(self) -> int:
        return 10

    def vocab_size(self) -> int:
        return 10 + len(self.members)

    def sample_row(self, u: torch.Tensor) -> torch.Tensor:
        """Digit 0 becomes the task token (near-uniform via mod N); payload unchanged."""
        row = u.clone()
        row[0] = 10 + (int(u[0]) % len(self.members))
        return row

    def __str__(self) -> str:
        return f"TaskMux([{', '.join(str(m) for m in self.members)}])"




def _ref_pmaxv9(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y, m = [], 0
    for v in x:
        if v != 9:
            m = max(m, v)
        y.append(m)
    return y

class Pmaxv9(DSLExpr):
    """pmaxv9_no9 (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_pmaxv9, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Pmaxv9({self.child})"



def _ref_dfa3t010020001(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    T = [[0, 1, 0], [0, 2, 0], [0, 0, 1]]
    s, y = 0, []
    for d in x:
        c = 0 if d <= 2 else (1 if d <= 6 else 2)
        s = T[s][c]
        y.append(s)
    return y

class SmallDFATrace(DSLExpr):
    """dfa3_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_dfa3t010020001, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"SmallDFATrace({self.child})"



def _ref_indecho(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    last, y = {}, []
    for i, v in enumerate(x):
        j = last.get(v)
        y.append(x[j + 1] if j is not None else v)
        last[v] = i
    return y

class Indecho(DSLExpr):
    """indecho_alldistinct (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_indecho, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Indecho({self.child})"



def _ref_cosetprod(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    E = [(0, 1, 2), (0, 2, 1), (1, 2, 0), (1, 0, 2), (2, 0, 1), (2, 1, 0)]
    C = [0, 1, 1, 0, 2, 2]
    idx = {e: i for i, e in enumerate(E)}
    cur = (0, 1, 2); out = []
    for d in x:
        e = E[d % 6]; cur = tuple(e[cur[i]] for i in range(3))
        out.append(C[idx[cur]])
    return out

class GroupCosetProductS3(DSLExpr):
    """cosetprod_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_cosetprod, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"GroupCosetProductS3({self.child})"



def _ref_kanc2(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    M = [3, 5, 7, 9, 1, 3, 7, 5, 9, 1]; B = [1, 2, 5, 0, 3, 8, 6, 4, 9, 7]
    return [(v + M[v % 2] * x[0] + B[v % 2]) % 10 for v in x]

class Kanc2(DSLExpr):
    """kanc2_class0 (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_kanc2, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Kanc2({self.child})"



def _ref_kanc4(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    M = [3, 5, 7, 9, 1, 3, 7, 5, 9, 1]; B = [1, 2, 5, 0, 3, 8, 6, 4, 9, 7]
    return [(v + M[v % 4] * x[0] + B[v % 4]) % 10 for v in x]

class Kanc4(DSLExpr):
    """kanc4_class0 (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_kanc4, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Kanc4({self.child})"



def _ref_kanc8(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    M = [3, 5, 7, 9, 1, 3, 7, 5, 9, 1]; B = [1, 2, 5, 0, 3, 8, 6, 4, 9, 7]
    return [(v + M[v % 8] * x[0] + B[v % 8]) % 10 for v in x]

class Kanc8(DSLExpr):
    """kanc8_class0 (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_kanc8, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Kanc8({self.child})"



def _ref_kanc10(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    M = [3, 5, 7, 9, 1, 3, 7, 5, 9, 1]; B = [1, 2, 5, 0, 3, 8, 6, 4, 9, 7]
    return [(v + M[v % 10] * x[0] + B[v % 10]) % 10 for v in x]

class Kanc10(DSLExpr):
    """kanc10_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_kanc10, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Kanc10({self.child})"



def _ref_sanc2(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    M = [3, 5, 7, 9, 1, 3, 7, 5, 9, 1]; B = [1, 2, 5, 0, 3, 8, 6, 4, 9, 7]
    y, par = [], 0
    for v in x:
        par ^= v % 2
        y.append((v + M[par] * x[0] + B[par]) % 10)
    return y

class Sanc2(DSLExpr):
    """sanc2_alleven (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_sanc2, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Sanc2({self.child})"



def _ref_tmap5(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [(v + 5) % 10 if v >= 5 else v for v in x]

class Tmap5(DSLExpr):
    """tmap5_low (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_tmap5, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Tmap5({self.child})"



def _ref_addfix(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    f, y = 0, []
    for i, v in enumerate(x):
        f += int(v == i)
        y.append((v + f) % 10)
    return y

class Addfix(DSLExpr):
    """addfix_onefp (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_addfix, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Addfix({self.child})"



def _ref_sjf1(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y, M = [x[0]], x[0]
    for i in range(1, 10):
        y.append(min(M, x[i])); M = max(M, x[i])
    return y

class Sjf1(DSLExpr):
    """sjf1_busy (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_sjf1, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Sjf1({self.child})"



def _ref_duelsum(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    e, o, y = 0, 0, []
    for v in x:
        if v % 2 == 0:
            e += v
        else:
            o += v
        y.append((e if e >= o else o) % 10)
    return y

class Duelsum(DSLExpr):
    """duel_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_duelsum, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Duelsum({self.child})"



def _ref_netrm(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y, M, m = [], x[0], x[0]
    for v in x:
        M = max(M, v); m = min(m, v)
        y.append(M if 2 * v >= M + m else m)
    return y

class Netrm(DSLExpr):
    """netrm_maxside (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_netrm, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Netrm({self.child})"



def _ref_convfix(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    s = [d & 1 for d in x]
    p = [(d >> 1) & 1 for d in x]
    z = [p[i] ^ s[i] ^ (s[i - 1] if i else 0) for i in range(10)]
    return [0] + [s[i - 1] ^ (z[i - 1] & z[i]) for i in range(1, 10)]

class Convfix(DSLExpr):
    """convfix_clean (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_convfix, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Convfix({self.child})"



def _ref_usum(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    seen, s, y = set(), 0, []
    for v in x:
        if v not in seen:
            seen.add(v); s = (s + v) % 10
        y.append(s)
    return y

class Usum(DSLExpr):
    """usum_binary (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_usum, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Usum({self.child})"



def _ref_perase0(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [0 if i == x[0] else x[i] for i in range(10)]

class Perase0(DSLExpr):
    """perase0_pin3 (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_perase0, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Perase0({self.child})"



def _ref_lagf3(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    k, y = 0, []
    for i, v in enumerate(x):
        k += int(v == 3)
        y.append(x[(i - k) % 10])
    return y

class Lagf3(DSLExpr):
    """lagf3_ge1three (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_lagf3, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Lagf3({self.child})"



def _ref_nearidx(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y, M, m, iM, im = [], x[0], x[0], 0, 0
    for i, v in enumerate(x):
        if v > M:
            M, iM = v, i
        if v < m:
            m, im = v, i
        y.append(iM if 2 * v >= M + m else im)
    return y

class Nearidx(DSLExpr):
    """nearidx_maxside (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_nearidx, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Nearidx({self.child})"



def _ref_tresetsum3(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    s, y = 0, []
    for v in x:
        s = 0 if v == 3 else (s + v) % 10
        y.append(s)
    return y

class Tresetsum3(DSLExpr):
    """ts3_one3 (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_tresetsum3, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Tresetsum3({self.child})"



def _ref_oepass0(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = list(x)
    for k in range(0, 9, 2):
        a, b = y[k], y[k + 1]
        y[k], y[k + 1] = min(a, b), max(a, b)
    return y

class Oepass0(DSLExpr):
    """oepass_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_oepass0, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Oepass0({self.child})"



def _ref_pge(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [next((x[j] for j in range(i - 1, -1, -1) if x[j] > x[i]), 0)
            for i in range(10)]

class Pge(DSLExpr):
    """pge_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_pge, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Pge({self.child})"



def _ref_cgpmax09(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    if x[0] <= x[9]:
        return list(x)
    y, m = [], 0
    for v in x:
        m = max(m, v)
        y.append(m)
    return y

class Cgpmax09(DSLExpr):
    """cgp_pmaxonly (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_cgpmax09, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Cgpmax09({self.child})"



def _ref_lanesum(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [sum(x[j] for j in range(i + 1) if j % 2 == i % 2) % 10 for i in range(10)]

class Lanesum(DSLExpr):
    """lanesum_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_lanesum, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Lanesum({self.child})"



def _ref_addpmax(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    out, m = [], 0
    for v in x:
        out.append((v + m) % 10)
        m = max(m, v)
    return out

class Addpmax(DSLExpr):
    """addpmax_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_addpmax, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Addpmax({self.child})"



def _ref_maxrun(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y, r, best, prev = [], 0, 0, None
    for v in x:
        r = r + 1 if v == prev else 1
        best = max(best, r)
        y.append(min(best, 9))
        prev = v
    return y

class Maxrun(DSLExpr):
    """maxrun_onepair (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_maxrun, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Maxrun({self.child})"



def _ref_dualtie(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = [x[0]]
    for i in range(1, 10):
        pre = x[:i]
        y.append(0 if x[i] == max(pre) or x[i] == min(pre) else x[i])
    return y

class Dualtie(DSLExpr):
    """dualtie_notie (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_dualtie, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Dualtie({self.child})"



def _ref_cap59(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y, c = [], 0
    for v in x:
        c += int(v >= 5)
        y.append(9 if c >= 5 else v)
    return y

class Cap59(DSLExpr):
    """cap59_nofire (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_cap59, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Cap59({self.child})"



def _ref_cgrev09(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return x[::-1] if x[0] > x[9] else x[:]

class Cgrev09(DSLExpr):
    """cgrev09_revonly (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_cgrev09, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Cgrev09({self.child})"



def _ref_dupmax(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    seen, best, y = {}, 0, []
    for v in x:
        seen[v] = seen.get(v, 0) + 1
        if seen[v] >= 2:
            best = max(best, v)
        y.append(best)
    return y

class Dupmax(DSLExpr):
    """dupmax_onerepeat (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_dupmax, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Dupmax({self.child})"



def _ref_rmaj5(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    c, y = 0, []
    for i, v in enumerate(x):
        c += int(v >= 5)
        y.append(1 if 2 * c > i + 1 else 0)
    return y

class Rmaj5(DSLExpr):
    """rmaj5_monodec (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_rmaj5, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Rmaj5({self.child})"



def _ref_majmask(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y, e, o = [], 0, 0
    for v in x:
        e += int(v % 2 == 0)
        o += int(v % 2 == 1)
        minority = (v % 2 == 0 and e < o) or (v % 2 == 1 and o < e)
        y.append(0 if minority else v)
    return y

class Majmask(DSLExpr):
    """majmask_nominority (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_majmask, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Majmask({self.child})"



def _ref_bcast0(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [x[0]] * 10

class AnchorBroadcast(DSLExpr):
    """bcast0_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_bcast0, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"AnchorBroadcast({self.child})"



def _ref_zpeak(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = []
    for i, v in enumerate(x):
        left = x[i - 1] if i > 0 else -1
        right = x[i + 1] if i < 9 else -1
        y.append(0 if v > left and v > right else v)
    return y

class Zpeak(DSLExpr):
    """zpeak_monotone (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_zpeak, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Zpeak({self.child})"



def _ref_mulanc0(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    a = x[0]
    return [(a * v) % 10 for v in x]

class Mulanc0(DSLExpr):
    """mulanc0_caesar3 (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_mulanc0, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Mulanc0({self.child})"



def _ref_recdist(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    out, cur, last = [], -1, 0
    for i, v in enumerate(x):
        if v > cur:
            cur, last = v, i
        out.append(i - last)
    return out

class Recdist(DSLExpr):
    """recdist_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_recdist, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Recdist({self.child})"



def _ref_pmrep(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    def step(v):
        return v + 1 if v % 2 == 0 else v - 1
    y = [step(x[0])]
    for i in range(1, 10):
        y.append(x[i] if x[i] == x[i - 1] else step(x[i]))
    return y

class Pmrep(DSLExpr):
    """pmrep_norepeat (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_pmrep, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Pmrep({self.child})"



def _ref_mulpair(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [x[0]] + [(x[i - 1] * x[i]) % 10 for i in range(1, 10)]

class Mulpair(DSLExpr):
    """mulpair_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_mulpair, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Mulpair({self.child})"



def _ref_prevsegmax3(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    prev, cur, y = 0, 0, []
    for v in x:
        if v == 3:
            prev, cur = cur, 0
            y.append(prev)
        else:
            cur = max(cur, v)
            y.append(prev)
    return y

class PrevSegmentMax(DSLExpr):
    """prevsegmax_onethree (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_prevsegmax3, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"PrevSegmentMax({self.child})"



def _ref_occcnt(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [sum(1 for j in range(i) if x[j] == x[i]) for i in range(10)]

class Occcnt(DSLExpr):
    """occcnt_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_occcnt, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Occcnt({self.child})"



def _ref_segdist3(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    seen, y = set(), []
    for v in x:
        if v == 3:
            seen = set()
            y.append(0)
        else:
            seen.add(v)
            y.append(len(seen))
    return y

class SegmentDistinct(DSLExpr):
    """segdist3_onethree (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_segdist3, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"SegmentDistinct({self.child})"



def _ref_dclamp(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = [x[0]]
    for i in range(1, 10):
        lo, hi = min(x[:i]), max(x[:i])
        y.append(min(hi, max(lo, x[i])))
    return y

class Dclamp(DSLExpr):
    """dclamp_pinband (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_dclamp, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Dclamp({self.child})"



def _ref_firstuniq(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = []
    for i in range(10):
        pre = x[:i + 1]
        cnt = {v: pre.count(v) for v in pre}
        ans = 0
        for v in pre:
            if cnt[v] == 1:
                ans = v; break
        y.append(ans)
    return y

class FirstUnique(DSLExpr):
    """fu_alldistinct (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_firstuniq, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"FirstUnique({self.child})"



def _ref_runmed(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [sorted(x[:i + 1])[i // 2] for i in range(10)]

class Runmed(DSLExpr):
    """runmed_sorted (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_runmed, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Runmed({self.child})"



def _ref_cntge3(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    c, y = 0, []
    for v in x:
        c += int(v >= 3)
        y.append(c % 10)
    return y

class Cntge3(DSLExpr):
    """cntge3_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_cntge3, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Cntge3({self.child})"



def _ref_twosum10(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    seen, y = set(), []
    for v in x:
        y.append(1 if ((10 - v) % 10) in seen else 0)
        seen.add(v)
    return y

class Twosum10(DSLExpr):
    """ts_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_twosum10, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Twosum10({self.child})"



def _ref_holdodd(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    h, y = 0, []
    for v in x:
        if v % 2 == 1: h = v
        y.append(h)
    return y

class Holdodd(DSLExpr):
    """ho_holdge5equiv (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_holdodd, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Holdodd({self.child})"



def _ref_addlastodd(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    h, y = 0, []
    for v in x:
        if v % 2 == 1: h = v
        y.append((v + h) % 10)
    return y

class Addlastodd(DSLExpr):
    """alo_alleven (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_addlastodd, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Addlastodd({self.child})"



def _ref_bmcand(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    cand, score, y = None, 0, []
    for v in x:
        if score == 0:
            cand, score = v, 1
        elif v == cand:
            score += 1
        else:
            score -= 1
        y.append(cand)
    return y

class Bmcand(DSLExpr):
    """bm_alldistinct (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_bmcand, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Bmcand({self.child})"



def _ref_hold2ge5(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    q, y = [], []
    for v in x:
        if v >= 5: q.append(v)
        y.append(q[-2] if len(q) >= 2 else 0)
    return y

class Hold2ge5(DSLExpr):
    """h2_ge2qual (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_hold2ge5, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Hold2ge5({self.child})"



def _ref_sjf2(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    room, y = [], []
    for v in x:
        room.append(v)
        if len(room) >= 3:
            room.sort()
            y.append(room.pop(0))
        else:
            y.append(0)
    return y

class Sjf2(DSLExpr):
    """sjf2_jam (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_sjf2, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Sjf2({self.child})"



def _ref_gapge5(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    pos, y = [], []
    for i, v in enumerate(x):
        if v >= 5: pos.append(i)
        y.append(pos[-1] - pos[-2] if len(pos) >= 2 else 0)
    return y

class Gapge5(DSLExpr):
    """gapge5_ge2high (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_gapge5, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Gapge5({self.child})"



def _ref_riseerase(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [x[0]] + [0 if x[i] > x[i - 1] else x[i] for i in range(1, 10)]

class Riseerase(DSLExpr):
    """riseerase_norise (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_riseerase, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Riseerase({self.child})"



def _ref_ebud3(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    c, y = 0, []
    for v in x:
        if v >= 5:
            c += 1
            y.append(0 if c <= 3 else v)
        else:
            y.append(v)
    return y

class Ebud3(DSLExpr):
    """ebud3_lowbudget (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_ebud3, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Ebud3({self.child})"



def _ref_tieswap(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = [x[0]]
    for i in range(1, 10):
        pre = x[:i]; mx, mn = max(pre), min(pre)
        if x[i] == mx: y.append(mn)
        elif x[i] == mn: y.append(mx)
        else: y.append(x[i])
    return y

class Tieswap(DSLExpr):
    """tieswap_notie (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_tieswap, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Tieswap({self.child})"



def _ref_risemax(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = [x[0]]
    for i in range(1, 10):
        if x[i] > x[i - 1]: y.append(max(x[:i + 1]))
        else: y.append(x[i])
    return y

class Risemax(DSLExpr):
    """risemax_norise (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_risemax, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Risemax({self.child})"



def _ref_delc2(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [0 if i >= 2 and x[i - 2] >= 5 else x[i] for i in range(10)]

class Delc2(DSLExpr):
    """delc2_alllow (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_delc2, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Delc2({self.child})"



def _ref_riskeep(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [0] + [x[i] if x[i] > x[i - 1] else 0 for i in range(1, 10)]

class Riskeep(DSLExpr):
    """riskeep_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_riskeep, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Riskeep({self.child})"



def _ref_mixtie(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = [x[0]]
    for i in range(1, 10):
        pre = x[:i]; mx, mn = max(pre), min(pre)
        if x[i] == mx: y.append(0)
        elif x[i] == mn: y.append(mx)
        else: y.append(x[i])
    return y

class Mixtie(DSLExpr):
    """mixtie_notie (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_mixtie, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Mixtie({self.child})"



def _ref_pmaxv89(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    veto = [8, 9]
    y, m = [], 0
    for v in x:
        if v not in veto: m = max(m, v)
        y.append(m)
    return y

class Pmaxv89(DSLExpr):
    """pmaxv89_noveto (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_pmaxv89, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Pmaxv89({self.child})"



def _ref_pmaxv789(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    veto = [7, 8, 9]
    y, m = [], 0
    for v in x:
        if v not in veto: m = max(m, v)
        y.append(m)
    return y

class Pmaxv789(DSLExpr):
    """pmaxv789_noveto (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_pmaxv789, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Pmaxv789({self.child})"



def _ref_togsum0(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    e, o, sel, y = 0, 0, 0, []
    for v in x:
        if v % 2 == 0: e += v
        else: o += v
        if v == 0: sel ^= 1
        y.append((o if sel else e) % 10)
    return y

class Togsum0(DSLExpr):
    """tg_nozero (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_togsum0, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Togsum0({self.child})"



def _ref_signsum0(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    s, sign, y = 0, 1, []
    for v in x:
        if v == 0: sign *= -1
        s = (s + sign * v) % 10
        y.append(s)
    return y

class Signsum0(DSLExpr):
    """ss_nozero (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_signsum0, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Signsum0({self.child})"



def _ref_med3(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [x[i] if i in (0, 9) else sorted([x[i - 1], x[i], x[i + 1]])[1] for i in range(10)]

class Median3(DSLExpr):
    """med3_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_med3, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Median3({self.child})"



def _ref_maj3(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    from collections import Counter
    y = []
    for i in range(10):
        win = [x[j] for j in (i - 1, i, i + 1) if 0 <= j < 10]
        best = [v for v, c in Counter(win).items() if c >= 2]
        y.append(best[0] if best else x[i])
    return y

class Majority3(DSLExpr):
    """maj3_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_maj3, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Majority3({self.child})"



def _ref_spile4(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = []
    for i in range(10):
        v = x[i] - (4 if x[i] >= 4 else 0)
        if i > 0 and x[i - 1] >= 4: v += 1
        if i < 9 and x[i + 1] >= 4: v += 1
        y.append(v)
    return y

class Spile4(DSLExpr):
    """spile4_low (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_spile4, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Spile4({self.child})"



def _ref_spile5(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = []
    for i in range(10):
        v = x[i] - (5 if x[i] >= 5 else 0)
        if i > 0 and x[i - 1] >= 5: v += 1
        if i < 9 and x[i + 1] >= 5: v += 1
        y.append(v)
    return y

class Spile5(DSLExpr):
    """spile5_low (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_spile5, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Spile5({self.child})"



def _ref_spile6(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = []
    for i in range(10):
        v = x[i] - (6 if x[i] >= 6 else 0)
        if i > 0 and x[i - 1] >= 6: v += 1
        if i < 9 and x[i + 1] >= 6: v += 1
        y.append(v)
    return y

class Spile6(DSLExpr):
    """spile6_low (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_spile6, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Spile6({self.child})"



def _ref_cyca(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = []
    for i in range(10):
        succ = (x[i] + 1) % 10
        nb = ([x[i - 1]] if i > 0 else []) + ([x[i + 1]] if i < 9 else [])
        y.append(succ if succ in nb else x[i])
    return y

class Cyca(DSLExpr):
    """cyca_evens (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_cyca, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Cyca({self.child})"



def _ref_r184t4(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    car = [v >= 4 for v in x]
    y = list(x); used = [False] * 10
    for i in range(9):
        if car[i] and not car[i + 1] and not used[i] and not used[i + 1]:
            y[i], y[i + 1] = x[i + 1], x[i]
            used[i] = used[i + 1] = True
    return y

class Rule184Thr4(DSLExpr):
    """r184t4_allhigh (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_r184t4, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Rule184Thr4({self.child})"



def _ref_r184t5(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    car = [v >= 5 for v in x]
    y = list(x); used = [False] * 10
    for i in range(9):
        if car[i] and not car[i + 1] and not used[i] and not used[i + 1]:
            y[i], y[i + 1] = x[i + 1], x[i]
            used[i] = used[i + 1] = True
    return y

class Rule184Thr5(DSLExpr):
    """r184t5_allhigh (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_r184t5, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Rule184Thr5({self.child})"



def _ref_r184t6(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    car = [v >= 6 for v in x]
    y = list(x); used = [False] * 10
    for i in range(9):
        if car[i] and not car[i + 1] and not used[i] and not used[i + 1]:
            y[i], y[i + 1] = x[i + 1], x[i]
            used[i] = used[i + 1] = True
    return y

class Rule184Thr6(DSLExpr):
    """r184t6_allhigh (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_r184t6, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Rule184Thr6({self.child})"



def _ref_ewma2(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y, prev = [], 0
    for v in x:
        cur = (v + prev // 2) % 10
        y.append(cur); prev = cur
    return y

class Ewma2(DSLExpr):
    """leaky_bin01 (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_ewma2, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Ewma2({self.child})"



def _ref_zcross5(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y, c = [], 0
    for i in range(10):
        if i > 0 and (x[i] >= 5) != (x[i - 1] >= 5): c += 1
        y.append(c % 10)
    return y

class Zcross5(DSLExpr):
    """zc_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_zcross5, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Zcross5({self.child})"



def _ref_heapmax(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = []
    for i in range(10):
        cand = [x[i]]
        if 2 * i + 1 < 10: cand.append(x[2 * i + 1])
        if 2 * i + 2 < 10: cand.append(x[2 * i + 2])
        y.append(max(cand))
    return y

class Heapmax(DSLExpr):
    """heapmax_desc (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_heapmax, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Heapmax({self.child})"



def _ref_vig2(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    key = [3, 1]
    return [(x[i] + key[i % 2]) % 10 for i in range(10)]

class Vig2(DSLExpr):
    """vig2_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_vig2, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Vig2({self.child})"



def _ref_vig3(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    key = [3, 1, 4]
    return [(x[i] + key[i % 3]) % 10 for i in range(10)]

class Vig3(DSLExpr):
    """vig3_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_vig3, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Vig3({self.child})"



def _ref_vig5(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    key = [3, 1, 4, 1, 5]
    return [(x[i] + key[i % 5]) % 10 for i in range(10)]

class Vig5(DSLExpr):
    """vig5_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_vig5, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Vig5({self.child})"



def _ref_hamk3(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    pat = [1, 2, 3]; y = []
    for i in range(10):
        c = 0
        for k in range(3):
            j = i - 2 + k
            if j >= 0 and x[j] == pat[k]: c += 1
        y.append(c)
    return y

class Hamk3(DSLExpr):
    """hamk3_alph123 (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_hamk3, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Hamk3({self.child})"



def _ref_altrun(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y, best, cur, lastdir = [], 1, 1, 0
    for i in range(10):
        if i == 0:
            cur, lastdir = 1, 0
        else:
            d = x[i] - x[i - 1]; s = (d > 0) - (d < 0)
            if s == 0:
                cur, lastdir = 1, 0
            elif lastdir == 0 or s == lastdir:
                cur, lastdir = 2, s
            else:
                cur, lastdir = cur + 1, s
        best = max(best, cur)
        y.append(best)
    return y

class Altrun(DSLExpr):
    """altrun_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_altrun, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def vocab_size(self) -> int:
        return 11                          # longest alternating run over 10 positions is 10
    def __str__(self) -> str:
        return f"Altrun({self.child})"



def _ref_blkpar(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = list(x)
    for k in range(0, 10, 2):
        if (x[k] + x[k + 1]) % 2 == 1: y[k], y[k + 1] = x[k + 1], x[k]
    return y

class BlockSwapParity(DSLExpr):
    """blkpar_even (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_blkpar, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"BlockSwapParity({self.child})"



def _ref_blkoe2(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = list(x)
    for k in range(0, 10, 2):
        if y[k] > y[k + 1]: y[k], y[k + 1] = y[k + 1], y[k]
    for k in range(1, 9, 2):
        if y[k] > y[k + 1]: y[k], y[k + 1] = y[k + 1], y[k]
    return y

class BrickSort2(DSLExpr):
    """blkoe2_asc (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_blkoe2, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"BrickSort2({self.child})"



def _ref_blkadd(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = list(x)
    for k in range(0, 10, 2):
        y[k + 1] = (x[k] + x[k + 1]) % 10
    return y

class FeistelAdd(DSLExpr):
    """blkadd_ctrl0 (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_blkadd, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"FeistelAdd({self.child})"



def _ref_blkmax(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = list(x)
    for k in range(0, 10, 2):
        m = max(x[k], x[k + 1]); y[k] = y[k + 1] = m
    return y

class BlockMax2(DSLExpr):
    """blkmax_asc (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_blkmax, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"BlockMax2({self.child})"



def _ref_errdiff2(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    import math
    e, y = 0, []
    for v in x:
        q = 2 * math.floor((v + e) / 2 + 0.5)
        e = (v + e) - q
        y.append(q % 10)
    return y

class ErrorDiffuse2(DSLExpr):
    """errdiff2_grid (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_errdiff2, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"ErrorDiffuse2({self.child})"



def _ref_errdiff3(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    import math
    e, y = 0, []
    for v in x:
        q = 3 * math.floor((v + e) / 3 + 0.5)
        e = (v + e) - q
        y.append(q % 10)
    return y

class ErrorDiffuse3(DSLExpr):
    """errdiff3_grid (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_errdiff3, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"ErrorDiffuse3({self.child})"



def _ref_errdiff5(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    import math
    e, y = 0, []
    for v in x:
        q = 5 * math.floor((v + e) / 5 + 0.5)
        e = (v + e) - q
        y.append(q % 10)
    return y

class ErrorDiffuse5(DSLExpr):
    """errdiff5_grid (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_errdiff5, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"ErrorDiffuse5({self.child})"



def _ref_boxsum2(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [sum(x[max(0, i - 2 + 1):i + 1]) % 10 for i in range(10)]

class BoxSum2(DSLExpr):
    """boxsum2_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_boxsum2, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"BoxSum2({self.child})"



def _ref_boxsum3(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [sum(x[max(0, i - 3 + 1):i + 1]) % 10 for i in range(10)]

class BoxSum3(DSLExpr):
    """boxsum3_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_boxsum3, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"BoxSum3({self.child})"



def _ref_boxsum4(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [sum(x[max(0, i - 4 + 1):i + 1]) % 10 for i in range(10)]

class BoxSum4(DSLExpr):
    """boxsum4_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_boxsum4, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"BoxSum4({self.child})"



def _ref_boxsum5(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [sum(x[max(0, i - 5 + 1):i + 1]) % 10 for i in range(10)]

class BoxSum5(DSLExpr):
    """boxsum5_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_boxsum5, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"BoxSum5({self.child})"



def _ref_schmitt(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    s, y = 9, []
    for v in x:
        if v <= 3: s = 0
        elif v >= 6: s = 9
        y.append(s)
    return y

class Schmitt(DSLExpr):
    """schmitt_noband (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_schmitt, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Schmitt({self.child})"



def _ref_railfence2(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    rails = [[] for _ in range(2)]
    r, d = 0, 1
    for i in range(10):
        rails[r].append(i)
        if r == 0: d = 1
        elif r == 2 - 1: d = -1
        r += d
    order = [i for rail in rails for i in rail]
    return [x[order[j]] for j in range(10)]

class RailFence2(DSLExpr):
    """railfence2_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_railfence2, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"RailFence2({self.child})"



def _ref_railfence3(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    rails = [[] for _ in range(3)]
    r, d = 0, 1
    for i in range(10):
        rails[r].append(i)
        if r == 0: d = 1
        elif r == 3 - 1: d = -1
        r += d
    order = [i for rail in rails for i in rail]
    return [x[order[j]] for j in range(10)]

class RailFence3(DSLExpr):
    """railfence3_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_railfence3, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"RailFence3({self.child})"



def _ref_railfence4(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    rails = [[] for _ in range(4)]
    r, d = 0, 1
    for i in range(10):
        rails[r].append(i)
        if r == 0: d = 1
        elif r == 4 - 1: d = -1
        r += d
    order = [i for rail in rails for i in rail]
    return [x[order[j]] for j in range(10)]

class RailFence4(DSLExpr):
    """railfence4_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_railfence4, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"RailFence4({self.child})"



def _ref_qlen3(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    q, y = 0, []
    for v in x:
        q = max(0, min(9, q + (1 if v >= 3 else -1)))
        y.append(q)
    return y

class QueueWalk3(DSLExpr):
    """qlen3_high (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_qlen3, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"QueueWalk3({self.child})"



def _ref_qlen4(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    q, y = 0, []
    for v in x:
        q = max(0, min(9, q + (1 if v >= 4 else -1)))
        y.append(q)
    return y

class QueueWalk4(DSLExpr):
    """qlen4_high (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_qlen4, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"QueueWalk4({self.child})"



def _ref_qlen5(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    q, y = 0, []
    for v in x:
        q = max(0, min(9, q + (1 if v >= 5 else -1)))
        y.append(q)
    return y

class QueueWalk5(DSLExpr):
    """qlen5_high (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_qlen5, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"QueueWalk5({self.child})"



def _ref_qlen6(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    q, y = 0, []
    for v in x:
        q = max(0, min(9, q + (1 if v >= 6 else -1)))
        y.append(q)
    return y

class QueueWalk6(DSLExpr):
    """qlen6_high (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_qlen6, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"QueueWalk6({self.child})"



def _ref_bpath(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    b = []
    for i in range(10):
        if i == 0: b.append(x[0])
        elif i == 1: b.append(max(x[1], b[0]))
        else: b.append(max(x[i], min(b[i - 1], b[i - 2])))
    return b

class MinBottleneck(DSLExpr):
    """bpath_asc (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_bpath, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"MinBottleneck({self.child})"



def _ref_wpath(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    c = []
    for i in range(10):
        if i == 0: c.append(x[0])
        elif i == 1: c.append(min(x[1], c[0]))
        else: c.append(min(x[i], max(c[i - 1], c[i - 2])))
    return c

class MaxCapacity(DSLExpr):
    """wpath_desc (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_wpath, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"MaxCapacity({self.child})"



def _ref_hsub(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    def sub(i):
        m = x[i]
        for c in (2 * i + 1, 2 * i + 2):
            if c < 10: m = max(m, sub(c))
        return m
    return [sub(i) for i in range(10)]

class HeapSubtreeMax(DSLExpr):
    """hsub_desc (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_hsub, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"HeapSubtreeMax({self.child})"



def _ref_incrun(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y, best, cur = [], 1, 1
    for i in range(10):
        if i > 0 and x[i] > x[i - 1]: cur += 1
        elif i > 0: cur = 1
        best = max(best, cur); y.append(best)
    return y

class IncRunMax(DSLExpr):
    """incrun_bin01 (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_incrun, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"IncRunMax({self.child})"



def _ref_haar(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = list(x)
    for k in range(0, 10, 2):
        a, b = x[k], x[k + 1]
        y[k], y[k + 1] = (a + b) % 10, (a - b) % 10
    return y

class HaarStep(DSLExpr):
    """haar_zerodetail (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_haar, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"HaarStep({self.child})"



def _ref_runmad(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    import math
    s, y = 0, []
    for i, v in enumerate(x):
        s += v; m = math.floor(s / (i + 1) + 0.5)
        y.append(abs(v - m))
    return y

class RunMAD(DSLExpr):
    """runmad_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_runmad, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"RunMAD({self.child})"



def _ref_contfrac(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    qm2, qm1, y = 0, 1, []
    for a in x:
        q = (a * qm1 + qm2) % 10
        y.append(q); qm2, qm1 = qm1, q
    return y

class ContFrac(DSLExpr):
    """contfrac_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_contfrac, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"ContFrac({self.child})"



def _ref_s3prod(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    E = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (2, 1, 0)]
    idx = {e: i for i, e in enumerate(E)}
    cur = (0, 1, 2); out = []
    for d in x:
        e = E[d % 6]
        cur = tuple(e[cur[i]] for i in range(3))
        out.append(idx[cur])
    return out

class GroupProductS3(DSLExpr):
    """s3prod_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_s3prod, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"GroupProductS3({self.child})"



def _ref_d4prod(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    E = [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2), (0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0)]
    idx = {e: i for i, e in enumerate(E)}
    cur = (0, 1, 2, 3); out = []
    for d in x:
        e = E[d % 8]
        cur = tuple(e[cur[i]] for i in range(4))
        out.append(idx[cur])
    return out

class GroupProductD4(DSLExpr):
    """d4prod_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_d4prod, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"GroupProductD4({self.child})"



def _ref_q8prod(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    T = [[0, 1, 2, 3, 4, 5, 6, 7], [1, 4, 3, 6, 5, 0, 7, 2], [2, 7, 4, 1, 6, 3, 0, 5], [3, 2, 5, 4, 7, 6, 1, 0], [4, 5, 6, 7, 0, 1, 2, 3], [5, 0, 7, 2, 1, 4, 3, 6], [6, 3, 0, 5, 2, 7, 4, 1], [7, 6, 1, 0, 3, 2, 5, 4]]
    cur = 0; out = []
    for d in x:
        cur = T[cur][d % 8]; out.append(cur)
    return out

class GroupProductQ8(DSLExpr):
    """q8prod_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_q8prod, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"GroupProductQ8({self.child})"



def _ref_d5prod(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    E = [(0, 1, 2, 3, 4), (1, 2, 3, 4, 0), (2, 3, 4, 0, 1), (3, 4, 0, 1, 2), (4, 0, 1, 2, 3), (0, 4, 3, 2, 1), (1, 0, 4, 3, 2), (2, 1, 0, 4, 3), (3, 2, 1, 0, 4), (4, 3, 2, 1, 0)]
    idx = {e: i for i, e in enumerate(E)}
    cur = (0, 1, 2, 3, 4); out = []
    for d in x:
        e = E[d % 10]
        cur = tuple(e[cur[i]] for i in range(5))
        out.append(idx[cur])
    return out

class GroupProductD5(DSLExpr):
    """d5prod_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_d5prod, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"GroupProductD5({self.child})"



def _ref_d4char(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    E = [(0, 1, 2, 3), (1, 2, 3, 0), (0, 3, 2, 1), (3, 0, 1, 2), (2, 1, 0, 3), (3, 2, 1, 0), (2, 3, 0, 1), (1, 0, 3, 2)]
    CHI = [2, 0, 0, 0, 0, 0, 8, 0]
    idx = {e: i for i, e in enumerate(E)}
    cur = (0, 1, 2, 3); out = []
    for d in x:
        e = E[d % 8]; cur = tuple(e[cur[i]] for i in range(4))
        out.append(CHI[idx[cur]])
    return out

class D4Character(DSLExpr):
    """d4char_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_d4char, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"D4Character({self.child})"



def _ref_d5act(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    E = [(0, 1, 2, 3, 4), (1, 2, 3, 4, 0), (2, 3, 4, 0, 1), (3, 4, 0, 1, 2), (4, 0, 1, 2, 3), (0, 4, 3, 2, 1), (1, 0, 4, 3, 2), (2, 1, 0, 4, 3), (3, 2, 1, 0, 4), (4, 3, 2, 1, 0)]
    cur = (0, 1, 2, 3, 4); out = []
    for d in x:
        out.append(cur[d % 5])
        e = E[d % 10]; cur = tuple(e[cur[i]] for i in range(5))
    return out

class GroupActionD5(DSLExpr):
    """d5act_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_d5act, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"GroupActionD5({self.child})"



def _ref_rangeadd(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    hi = lo = x[0]; out = []
    for v in x:
        hi = max(hi, v); lo = min(lo, v)
        out.append((v + hi - lo) % 10)
    return out

class RunningRangeAdd(DSLExpr):
    """rangeadd_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_rangeadd, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"RunningRangeAdd({self.child})"



def _ref_medadd(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    out = []
    for i in range(len(x)):
        s = sorted(x[:i + 1]); m = s[i // 2]
        out.append((x[i] + m) % 10)
    return out

class RunningMedianAdd(DSLExpr):
    """medadd_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_medadd, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"RunningMedianAdd({self.child})"



def _ref_tribscan(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    a = b = c = 0; out = []
    for v in x:
        q = (v + a + b + c) % 10; out.append(q)
        a, b, c = q, a, b
    return out

class ThirdOrderScan(DSLExpr):
    """tribscan_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_tribscan, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"ThirdOrderScan({self.child})"



def _ref_rotor3(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    u = v = w = 0; out = []
    for d in x:
        u = (u + w + d) % 10; v = (v + u) % 10; w = (w + v) % 10
        out.append(u)
    return out

class LinearRotor3(DSLExpr):
    """rotor3_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_rotor3, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"LinearRotor3({self.child})"



def _ref_modecntadd(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    from collections import Counter
    seen = Counter(); out = []
    for v in x:
        seen[v] += 1; out.append((v + max(seen.values())) % 10)
    return out

class RunningModeCountAdd(DSLExpr):
    """modecntadd_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_modecntadd, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"RunningModeCountAdd({self.child})"



def _ref_oscpos(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    th = p = 0; out = []
    for v in x:
        th = (th + p) % 10; p = (p - th + v) % 10
        out.append(th)
    return out

class Oscpos(DSLExpr):
    """oscpos_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_oscpos, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Oscpos({self.child})"



def _ref_oscmom(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    th = p = 0; out = []
    for v in x:
        th = (th + p) % 10; p = (p - th + v) % 10
        out.append(p)
    return out

class Oscmom(DSLExpr):
    """oscmom_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_oscmom, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Oscmom({self.child})"



def _ref_refrac(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    H = 5; last = None; out = []
    for i, v in enumerate(x):
        gap = (i - last) if last is not None else 99
        silenced = last is not None and (v + gap < H)
        out.append(0 if silenced else v)
        if v >= 7: last = i
    return out

class RefractoryVeto(DSLExpr):
    """refrac_alllow (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_refrac, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"RefractoryVeto({self.child})"



def _ref_mexhat(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    out = []
    for i in range(len(x)):
        if 0 < i < len(x) - 1 and x[i - 1] + x[i + 1] - 2 * x[i] >= 4:
            out.append(0)
        else:
            out.append(x[i])
    return out

class MexicanHatVeto(DSLExpr):
    """mexhat_monotone (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_mexhat, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"MexicanHatVeto({self.child})"



def _ref_maxfreq(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    from collections import Counter
    seen = Counter(); out = []
    for v in x:
        seen[v] += 1; out.append(max(seen.values()))
    return out

class MaxFrequency(DSLExpr):
    """maxfreq_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_maxfreq, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"MaxFrequency({self.child})"



def _ref_ducci(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [abs(x[i] - x[(i + 1) % 10]) for i in range(10)]

class DucciStep(DSLExpr):
    """ducci_binary (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_ducci, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"DucciStep({self.child})"



def _ref_pgcd(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    import math
    return [x[0]] + [math.gcd(x[i], x[i - 1]) for i in range(1, 10)]

class PairGcd(DSLExpr):
    """pgcd_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_pgcd, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"PairGcd({self.child})"



def _ref_fibscan(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    a = b = 0; out = []
    for v in x:
        q = (v + a + b) % 10; out.append(q); a, b = q, a
    return out

class SecondOrderScan(DSLExpr):
    """fibscan_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_fibscan, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"SecondOrderScan({self.child})"



def _ref_meanint(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    s = 0; out = []
    for i, v in enumerate(x):
        s += v; out.append(1 if s % (i + 1) == 0 else 0)
    return out

class NivenByCount(DSLExpr):
    """meanint_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_meanint, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"NivenByCount({self.child})"



def _ref_euclstep(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    out = [x[0]]
    for i in range(1, 10):
        out.append(x[i - 1] % x[i] if x[i] != 0 else x[i - 1])
    return out

class EuclidStep(DSLExpr):
    """euclstep_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_euclstep, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"EuclidStep({self.child})"



def _ref_paircarry(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    return [0] + [1 if x[i] + x[i - 1] >= 10 else 0 for i in range(1, 10)]

class PairCarry(DSLExpr):
    """paircarry_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_paircarry, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"PairCarry({self.child})"



def _ref_invcount(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    c = 0; out = []
    for i in range(10):
        c += sum(1 for j in range(i) if x[j] > x[i]); out.append(c % 10)
    return out

class InversionCount(DSLExpr):
    """invcount_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_invcount, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"InversionCount({self.child})"



def _ref_majindex(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    m = 0; out = []
    for i in range(10):
        if i > 0 and x[i - 1] > x[i]: m += i
        out.append(m % 10)
    return out

class MajorIndex(DSLExpr):
    """majindex_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_majindex, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"MajorIndex({self.child})"



def _ref_nimhist(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    from collections import Counter
    seen = Counter(); out = []
    for v in x:
        seen[v] += 1
        r = 0
        for k in range(10): r ^= seen[k]
        out.append(r % 10)
    return out

class NimHistogram(DSLExpr):
    """nimhist_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_nimhist, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"NimHistogram({self.child})"



def _ref_graviclump(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    m = round(sum(x) / len(x))
    return [v + (1 if m > v else (-1 if m < v else 0)) for v in x]

class MeanRelax(DSLExpr):
    """graviclump_binary (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_graviclump, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"MeanRelax({self.child})"



def _ref_entrain(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    t = x[0]; out = []
    for v in x:
        d = (t - v) % 10
        out.append(v if d == 0 else ((v + 1) % 10 if d <= 5 else (v - 1) % 10))
    return out

class AnchorNudge(DSLExpr):
    """entrain_pin3 (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_entrain, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"AnchorNudge({self.child})"



def _ref_phaselock(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    import math
    sc = sum(math.cos(2 * math.pi * v / 10) for v in x)
    ss = sum(math.sin(2 * math.pi * v / 10) for v in x)
    psi = math.atan2(ss, sc)
    out = []
    for v in x:
        a = 2 * math.pi * v / 10
        d = (psi - a + math.pi) % (2 * math.pi) - math.pi
        out.append(v if abs(d) < 1e-9 else ((v + 1) % 10 if d > 0 else (v - 1) % 10))
    return out

class PhaseNudge(DSLExpr):
    """phaselock_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_phaselock, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"PhaseNudge({self.child})"



def _ref_heatx(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = list(x); mx = max(x); mn = min(x)
    if mx != mn:
        y[x.index(mx)] = mx - 1; y[x.index(mn)] = mn + 1
    return y

class HeatExchange(DSLExpr):
    """heatx_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_heatx, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"HeatExchange({self.child})"



def _ref_e2sym(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    e2 = s1 = 0; out = []
    for v in x:
        e2 = (e2 + v * s1) % 10; s1 = (s1 + v) % 10; out.append(e2)
    return out

class ElemSym2(DSLExpr):
    """e2sym_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_e2sym, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"ElemSym2({self.child})"



def _ref_runmex(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    seen = set(); out = []
    for v in x:
        seen.add(v); m = 0
        while m in seen: m += 1
        out.append(min(m, 9))
    return out

class RunningMex(DSLExpr):
    """runmex_binary (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_runmex, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"RunningMex({self.child})"



def _ref_paladd(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    return [(v[i] + v[9 - i]) % 10 for i in range(10)]

class MirrorAdd(DSLExpr):
    """paladd_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_paladd, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"MirrorAdd({self.child})"



def _ref_palmax(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    return [max(v[i], v[9 - i]) for i in range(10)]

class MirrorMax(DSLExpr):
    """palmax_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_palmax, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"MirrorMax({self.child})"



def _ref_palmin(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    return [min(v[i], v[9 - i]) for i in range(10)]

class MirrorMin(DSLExpr):
    """palmin_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_palmin, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"MirrorMin({self.child})"



def _ref_rankadd(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    return [(v[i] + sum(1 for j in range(i) if v[j] < v[i])) % 10 for i in range(10)]

class OnlineRankAdd(DSLExpr):
    """rankadd_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_rankadd, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"OnlineRankAdd({self.child})"



def _ref_signdiff(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    n = len(v); out = []
    for i in range(n):
        l = v[i - 1] if i > 0 else v[i]
        r = v[i + 1] if i < n - 1 else v[i]
        lap = l + r - 2 * v[i]
        s = (lap > 0) - (lap < 0)
        out.append(min(9, max(0, v[i] + s)))
    return out

class SignDiffuse(DSLExpr):
    """signdiff_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_signdiff, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"SignDiffuse({self.child})"



def _ref_dblge5(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    return [(2 * d) % 10 if d >= 5 else d for d in v]

class DoubleGE(DSLExpr):
    """dblge5_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_dblge5, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"DoubleGE({self.child})"



def _ref_midadd(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    out = []; mx = mn = v[0]
    for i, d in enumerate(v):
        mx = max(mx, d); mn = min(mn, d)
        out.append((d + (mx + mn) // 2) % 10)
    return out

class MidrangeAdd(DSLExpr):
    """midadd_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_midadd, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"MidrangeAdd({self.child})"



def _ref_seedsum(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    return [(v[i] + (v[i - 1] if i >= 1 else 0) + (v[i - 3] if i >= 3 else 0)) % 10 for i in range(10)]

class SpacedSeedSum(DSLExpr):
    """seedsum_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_seedsum, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"SpacedSeedSum({self.child})"



def _ref_argposadd(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    out = []; m = -1; ap = 0
    for i, d in enumerate(v):
        if d > m: m = d; ap = i
        out.append((d + ap) % 10)
    return out

class ArgPosAdd(DSLExpr):
    """argposadd_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_argposadd, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"ArgPosAdd({self.child})"



def _ref_lanemax(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    out = []; m = [-1, -1]
    for i, d in enumerate(v):
        m[i % 2] = max(m[i % 2], d); out.append(m[i % 2])
    return out

class LaneRunningMax(DSLExpr):
    """lanemax_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_lanemax, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"LaneRunningMax({self.child})"



def _ref_pancake(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    m = max(v); k = v.index(m)
    return v[:k + 1][::-1] + v[k + 1:]

class PancakeFlip(DSLExpr):
    """pancake_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_pancake, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"PancakeFlip({self.child})"



def _ref_bitmaj(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    out = []; cnt = [0, 0, 0, 0]
    for i, d in enumerate(v):
        for b in range(4): cnt[b] += (d >> b) & 1
        n = i + 1; val = 0
        for b in range(4):
            if cnt[b] * 2 > n: val += (1 << b)
        out.append(val % 10)
    return out

class BitPlaneMajority(DSLExpr):
    """bitmaj_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_bitmaj, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"BitPlaneMajority({self.child})"



def _ref_skipn5(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    out = []; skip = False
    for d in v:
        if skip: out.append(0); skip = False
        else: out.append(d); skip = (d >= 5)
    return out

class PredicatedSkip(DSLExpr):
    """skipn5_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_skipn5, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"PredicatedSkip({self.child})"



def _ref_bittv(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    out = []; s = 0; prev = None
    for i, d in enumerate(v):
        if i > 0: s = (s + bin(d ^ prev).count('1')) % 10
        out.append(s); prev = d
    return out

class BitTransitionCount(DSLExpr):
    """bittv_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_bittv, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"BitTransitionCount({self.child})"



def _ref_vm1(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    out = []; acc = 0
    for d in v:
        if d <= 4: acc = d
        elif d == 5: acc = (acc + 1) % 10
        elif d == 6: acc = (acc - 1) % 10
        elif d == 7: acc = (2 * acc) % 10
        elif d == 8: acc = (9 - acc) % 10
        out.append(acc % 10)
    return out

class OpcodeMachine(DSLExpr):
    """vm1_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_vm1, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"OpcodeMachine({self.child})"



def _ref_tourney(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    x = [int(v) for v in x]
    slots = [(x[i], i) for i in range(10)] + [(-1, -1)] * 6
    depth = [0] * 10
    cur = slots
    while len(cur) > 1:
        nxt = []
        for k in range(0, len(cur), 2):
            av, ai = cur[k]; bv, bi = cur[k + 1]
            if bv < 0:
                win = cur[k]
            elif av < 0:
                win = cur[k + 1]
            else:
                win = cur[k] if (av > bv or (av == bv and ai < bi)) else cur[k + 1]
                depth[win[1]] += 1
            nxt.append(win)
        cur = nxt
    return depth

class KnockoutDepth(DSLExpr):
    """tourney_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_tourney, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"KnockoutDepth({self.child})"



def _ref_envmax(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    out = []; m = 0
    for d in v:
        m = max(d, m - 1, 0); out.append(m)
    return out

class EnvelopeMax(DSLExpr):
    """envmax_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_envmax, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"EnvelopeMax({self.child})"



def _ref_cmin(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    c = [0, 0, 0, 0]; out = []
    for d in x:
        d = int(d); a = d % 4; b = (d // 4) % 4
        c[a] += 1; c[b] += 1
        out.append(min(c[a], c[b]) % 10)
    return out

class CountMinSketch(DSLExpr):
    """cmin_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_cmin, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"CountMinSketch({self.child})"



def _ref_bittrans(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    x = [int(v) for v in x]; L = len(x)
    t = [(x[pos] >> plane) & 1 for plane in range(4) for pos in range(L)]
    return [(t[4*i] + 2*t[4*i+1] + 4*t[4*i+2] + 8*t[4*i+3]) % 10 for i in range(L)]

class BitTransposeGather(DSLExpr):
    """bittrans_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_bittrans, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"BitTransposeGather({self.child})"



def _ref_intern(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    d = {}; out = []
    for val in v:
        if val not in d: d[val] = (len(d) + 1) % 10
        out.append(d[val])
    return out

class InternRank(DSLExpr):
    """intern_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_intern, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"InternRank({self.child})"



def _ref_dupxor(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    from collections import Counter
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    c = Counter(v)
    return [(d ^ 1) if c[d] % 2 == 1 else d for d in v]

class MultiplicityParityXor(DSLExpr):
    """dupxor_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_dupxor, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"MultiplicityParityXor({self.child})"



def _ref_nbrdeg(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    return [sum(1 for j in range(len(v)) if j != i and abs(v[i] - v[j]) <= 1) for i in range(len(v))]

class ProximityDegree(DSLExpr):
    """nbrdeg_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_nbrdeg, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"ProximityDegree({self.child})"



def _ref_cfreq(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    from collections import Counter
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    c = Counter(v)
    return [sum(1 for xj in v if c[xj] > c[xi]) for xi in v]

class CumFreqRank(DSLExpr):
    """cfreq_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_cfreq, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"CumFreqRank({self.child})"



def _ref_dsub(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    d, last, out = [1], {}, []
    for i, val in enumerate(v):
        nxt = 2 * d[-1] - (d[last[val]] if val in last else 0)
        last[val] = i
        d.append(nxt)
        out.append(nxt % 10)
    return out

class DistinctSubseqCount(DSLExpr):
    """dsub_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_dsub, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"DistinctSubseqCount({self.child})"



def _ref_codonskip(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y, c = [], 0
    for v in x:
        if v == 0:
            y.append(0)
        else:
            c += 1
            y.append((c - 1) % 3 + 1)
    return y

class CodonSkip(DSLExpr):
    """codonskip_inframe (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_codonskip, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"CodonSkip({self.child})"



def _ref_fraglen(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = list(x); n = len(v)
    cuts = [0] + [i for i in range(1, n) if v[i] >= 5] + [n]
    y = [0] * n
    for a, b in zip(cuts, cuts[1:]):
        for i in range(a, b):
            y[i] = min(b - a, 9)
    return y

class FragLen(DSLExpr):
    """fraglen_onecut (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_fraglen, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"FragLen({self.child})"



def _ref_divzeta(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    div = lambda a, d: (d == 0) if a == 0 else (d % a == 0)
    return [sum(1 for j in range(i) if div(v[j], v[i])) for i in range(len(v))]

class DivisorZeta(DSLExpr):
    """divzeta_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_divzeta, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"DivisorZeta({self.child})"



def _ref_mobinv(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    y = [v[0]]
    for i in range(1, len(v)):
        y.append((v[i] - sum(y[d] for d in range(1, i) if i % d == 0)) % 10)
    return y

class MobiusInvert(DSLExpr):
    """mobinv_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_mobinv, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"MobiusInvert({self.child})"



def _ref_factnorm(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    out, c = [], 0
    for i in range(len(x)):
        r = min(i + 2, 10)
        t = x[i] + c
        out.append(t % r); c = t // r
    return out

class FactoradicNormalize(DSLExpr):
    """factnorm_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_factnorm, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"FactoradicNormalize({self.child})"



def _ref_nonadj4(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = []
    for i in range(len(x)):
        w = x[max(0, i - 3):i + 1]
        best = 0
        for m in range(1 << len(w)):
            if m & (m << 1):
                continue
            best = max(best, sum(w[k] for k in range(len(w)) if m >> k & 1))
        y.append(best % 10)
    return y

class NonAdjacentMax4(DSLExpr):
    """nonadj4_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_nonadj4, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"NonAdjacentMax4({self.child})"



def _ref_hkconf(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    out = []
    for a in v:
        nb = [b for b in v if abs(a - b) <= 2]
        out.append(int(math.floor(sum(nb) / len(nb) + 0.5)))
    return out

class BoundedConfidenceMean(DSLExpr):
    """hkconf_range_le2 (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_hkconf, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"BoundedConfidenceMean({self.child})"



def _ref_baksnep(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = x.tolist() if hasattr(x, 'tolist') else list(x)
    m, M = min(v), max(v)
    i = v.index(m)
    return [M if abs(j - i) <= 1 else d for j, d in enumerate(v)]

class ExtremalReplace(DSLExpr):
    """baksnep_no_extreme_tie (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_baksnep, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"ExtremalReplace({self.child})"



def _ref_hsifte(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y = list(x)
    for p, ch in [(0, [1, 2]), (3, [7, 8]), (4, [9])]:
        best = max(ch, key=lambda c: (y[c], -c))
        if y[best] > y[p]:
            y[p], y[best] = y[best], y[p]
    return y

class Hsifte(DSLExpr):
    """hsifte_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_hsifte, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Hsifte({self.child})"



def _ref_lprobe(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    occ, out = [False] * 10, []
    for v in x:
        s = int(v)
        while occ[s]:
            s = (s + 1) % 10
        occ[s] = True
        out.append(s)
    return out

class LinearProbe(DSLExpr):
    """lprobe_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_lprobe, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"LinearProbe({self.child})"



def _ref_dutch(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = [int(t) for t in x]
    return [d for d in v if d <= 2] + [d for d in v if 3 <= d <= 6] + [d for d in v if d >= 7]

class FlagPartition(DSLExpr):
    """dutch_whole (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_dutch, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"FlagPartition({self.child})"



def _ref_slew2(x, R=2):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    y, p = [], None
    for i, v in enumerate(x):
        p = v if i == 0 else min(max(v, p - R), p + R)
        y.append(p)
    return y

class Slew2(DSLExpr):
    """slew2_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_slew2, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Slew2({self.child})"



def _ref_rms3(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    win = lambda i: (x[max(0, i - 1)], x[i], x[min(9, i + 1)])
    return [int(math.floor(math.sqrt(sum(v * v for v in win(i)) / 3) + 0.5)) for i in range(10)]

class Rms3(DSLExpr):
    """rms3_full (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_rms3, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Rms3({self.child})"



def _ref_spn2(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    S = [5, 2, 1, 9, 6, 0, 4, 8, 7, 3]
    y = list(x)
    for _ in range(2):
        t = sum(y) % 10
        y = [S[(d + t) % 10] for d in y]
    return y

class Spn2(DSLExpr):
    """spn2_transparent (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_spn2, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Spn2({self.child})"



def _ref_xpar(x):
    import math
    from collections import Counter, defaultdict, deque, OrderedDict
    from itertools import groupby, accumulate, product, combinations, permutations
    from functools import reduce
    gcd = math.gcd
    v = list(x)
    r = [sum(v[5 * i:5 * i + 5]) % 10 for i in range(2)]
    c = [(v[j] + v[5 + j]) % 10 for j in range(5)]
    y = list(v)
    if any(r) and any(c):
        i = 0 if r[0] else 1
        j = next(k for k in range(5) if c[k])
        y[5 * i + j] = (y[5 * i + j] - r[i]) % 10
    return y

class Xpar(DSLExpr):
    """xpar_codeword (integrated)."""
    def __init__(self, child: DSLExpr):
        super().__init__(child.input_length); self.child = child
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        y = _promoted_apply(_ref_xpar, self.child.eval(input_seq))
        return torch.tensor(y, dtype=torch.long, device=input_seq.device)
    def output_length(self) -> int:
        return self.child.output_length()
    def __str__(self) -> str:
        return f"Xpar({self.child})"

def generate_algorithm_id(expr: DSLExpr) -> str:
    """
    Generate unique, human-readable ID from DSL expression structure.

    Uses compact operator codes and recursive traversal.
    Examples:
        Id(10) → "id"
        Reverse(Id(10)) → "rev_id"
        Take(5, MapAdd(2, Id(10))) → "take5_madd2_id"
        PadTo(7, 0, Reverse(Take(3, Id(10)))) → "pad7f0_rev_take3_id"

    Returns deterministic ID that encodes full AST structure.
    """
    def encode(node):
        op_name = node.__class__.__name__

        if op_name == 'Id':
            return 'id'
        elif op_name == 'Const':
            return f'c{node.digit}L{node.length}'
        elif op_name == 'Reverse':
            return f'rev_{encode(node.child)}'
        elif op_name == 'Rotate':
            return f'rot{node.k}_{encode(node.child)}'
        elif op_name == 'Take':
            return f'take{node.k}_{encode(node.child)}'
        elif op_name == 'Drop':
            return f'drop{node.k}_{encode(node.child)}'
        elif op_name == 'MapAdd':
            return f'madd{node.c}_{encode(node.child)}'
        elif op_name == 'MapMul':
            return f'mmul{node.c}_{encode(node.child)}'
        elif op_name == 'DoubleOdd':
            return f'dblodd_{encode(node.child)}'
        elif op_name == 'PadTo':
            return f'pad{node.M}f{node.fill}_{encode(node.child)}'
        elif op_name == 'JumpIndex':
            return f'jump{node.T}_{encode(node.child)}'
        elif op_name == 'PrefixMax':
            return f'pmax_{encode(node.child)}'
        elif op_name == 'PrefixSumMod':
            return f'psum_{encode(node.child)}'
        elif op_name == 'PrefixSumOdd':
            return f'psumodd_{encode(node.child)}'
        elif op_name == 'RunningProduct':
            return f'prod_{encode(node.child)}'
        elif op_name == 'CountOdd':
            return f'cntodd_{encode(node.child)}'
        elif op_name == 'RecordIndicator':
            return f'recordind_{encode(node.child)}'
        elif op_name == 'NoveltyIndicator':
            return f'novel_{encode(node.child)}'
        elif op_name == 'FixedPointIndicator':
            return f'fixedpoint_{encode(node.child)}'
        elif op_name == 'ResetRamp':
            return f'reset{node.d}_{encode(node.child)}'
        elif op_name == 'TokenResetMax':
            return f'tresetmax{node.d}_{encode(node.child)}'
        elif op_name == 'SignedWalk':
            return f'walk{node.up}{node.down}_{encode(node.child)}'
        elif op_name == 'SRLatch':
            return f'sr{node.set_tok}{node.reset_tok}_{encode(node.child)}'
        elif op_name == 'SampleHold':
            return f'hold{node.mode}{node.thresh}_{encode(node.child)}'
        elif op_name == 'Deplete':
            return f'dep{node.start}f{node.floor}_{encode(node.child)}'
        elif op_name == 'AnchorSubtract':
            return f'anc{node.a}_{encode(node.child)}'
        elif op_name == 'AnchorClamp':
            return f"{'fclamp' if node.mode == 'floor' else 'cclamp'}{node.a}_{encode(node.child)}"
        elif op_name == 'DataRotate':
            return f'drot{node.a}_{encode(node.child)}'
        elif op_name == 'DataReflect':
            return f'refl{node.a}_{encode(node.child)}'
        elif op_name == 'StrideGather':
            return f'stride{node.a}_{encode(node.child)}'
        elif op_name == 'PrefixReverseK':
            return f'pref{node.a}_{encode(node.child)}'
        elif op_name == 'PrefixSortK':
            return f'psort{node.a}_{encode(node.child)}'
        elif op_name == 'CompareSwap':
            return f'cmpswap{node.i}p{node.j}_{encode(node.child)}'
        elif op_name == 'WindowMax':
            return f'bmax{node.w}_{encode(node.child)}'
        elif op_name == 'ParityRoute':
            return f'pmstep_{encode(node.child)}'
        elif op_name == 'RippleCarry':
            return f'carryk{node.k}_{encode(node.child)}'
        elif op_name == 'ArgMinMax':
            return f'argminmax_{encode(node.child)}'
        elif op_name == 'RunningCountDigit':
            return f'rcnt{node.d}_{encode(node.child)}'
        elif op_name == 'CompareShift':
            return f'cmpsh{node.k}_{encode(node.child)}'
        elif op_name == 'DiffShift':
            return f'diffsh{node.k}_{encode(node.child)}'
        elif op_name == 'HashScan':
            return f'hscan{node.a}b{node.b}c{node.c}s{node.seed}_{encode(node.child)}'
        elif op_name == 'WindowHash':
            return f'whash{node.T}w{node.w}a{node.a}b{node.b}c{node.c}_{encode(node.child)}'
        elif op_name == 'IndexMix':
            return f'imix{node.T}a{node.a}b{node.b}_{encode(node.child)}'
        elif op_name == 'PermuteByRank':
            return f'prank{node.T}_{encode(node.child)}'
        elif op_name == 'CycleWalk':
            return f'cwalk{node.T}_{encode(node.child)}'
        elif op_name == 'TwoPointerMerge':
            return f'tpmerge{node.T}_{encode(node.child)}'
        elif op_name == 'ConditionalJump':
            return f'cjump{node.T}t{node.threshold}_{encode(node.child)}'
        elif op_name == 'TreeWalkDFS':
            return f'tdfs{node.T}_{encode(node.child)}'
        elif op_name == 'TreeWalkBFS':
            return f'tbfs{node.T}_{encode(node.child)}'
        elif op_name == 'CellularAutomaton':
            return f'ca{node.steps}a{node.a}b{node.b}c{node.c}_{encode(node.child)}'
        elif op_name == 'LFSR':
            return f'lfsr{node.T}w{node.width}t{node.tap1}t{node.tap2}c{node.c1}c{node.c2}_{encode(node.child)}'
        elif op_name == 'ParityInterleave':
            return f'parity{node.a}e{node.b}o{node.c}f{node.d}_{encode(node.child)}'
        elif op_name == 'ReductionTree':
            return f'rtree{node.T}a{node.a}b{node.b}_{encode(node.child)}'
        elif op_name == 'ChainedIndex':
            return f'chain{node.T}d{node.depth}_{encode(node.child)}'
        elif op_name == 'SelfConvolve':
            return f'sconv{node.T}_{encode(node.child)}'
        elif op_name == 'AccumulateReset':
            return f'areset{node.threshold}_{encode(node.child)}'
        elif op_name == 'ScatterWrite':
            return f'scatter{node.T}a{node.a}_{encode(node.child)}'
        elif op_name == 'InverseLookup':
            return f'invlook{node.T}_{encode(node.child)}'
        elif op_name == 'RunLengthEncode':
            return f'rle{node.T}_{encode(node.child)}'
        elif op_name == 'ChunkSort':
            return f'csort{node.i}to{node.j}{"a" if node.asc else "d"}_{encode(node.child)}'
        elif op_name == 'SegmentReverse':
            return f'segrev{node.boundary_digit}_{encode(node.child)}'
        elif op_name == 'XORScan':
            return f'xorscan_{encode(node.child)}'
        elif op_name == 'ModRemainder':
            return f'modrem{node.m}_{encode(node.child)}'
        elif op_name == 'ZigzagCount':
            return f'zigzag_{encode(node.child)}'
        elif op_name == 'RecordCount':
            return f'reccount_{encode(node.child)}'
        elif op_name == 'KMPFailure':
            return f'kmpfail_{encode(node.child)}'
        elif op_name == 'BWT':
            return f'bwt_{encode(node.child)}'
        elif op_name == 'LZ78Complexity':
            return f'lz78_{encode(node.child)}'
        elif op_name == 'VanEckExtension':
            return f'vaneck{node.T}_{encode(node.child)}'
        elif op_name == 'SwitchedScan':
            return f'swscan{node.a1}e{node.b1}f{node.a2}g{node.b2}t{node.thresh}_{encode(node.child)}'
        elif op_name == 'Sarank':
            return f'sarank_{encode(node.child)}'
        elif op_name == 'TaskMux':
            ids = [encode(m) for m in node.members]
            for mid in ids:
                if not re.fullmatch(r'[a-z0-9_]+', mid):
                    raise ValueError(f"TaskMux member id {mid!r} not encodable "
                                     "(the mux grammar reserves '.' and '-')")
            return 'mux' + str(len(ids)) + ''.join('.' + mid.replace('_', '-') for mid in ids)
        elif op_name == 'Pmaxv9':
            return f'pmaxv9_{encode(node.child)}'
        elif op_name == 'SmallDFATrace':
            return f'dfa3t010020001_{encode(node.child)}'
        elif op_name == 'Indecho':
            return f'indecho_{encode(node.child)}'
        elif op_name == 'GroupCosetProductS3':
            return f'cosetprod_{encode(node.child)}'
        elif op_name == 'Kanc2':
            return f'kanc2_{encode(node.child)}'
        elif op_name == 'Kanc4':
            return f'kanc4_{encode(node.child)}'
        elif op_name == 'Kanc8':
            return f'kanc8_{encode(node.child)}'
        elif op_name == 'Kanc10':
            return f'kanc10_{encode(node.child)}'
        elif op_name == 'Sanc2':
            return f'sanc2_{encode(node.child)}'
        elif op_name == 'Tmap5':
            return f'tmap5_{encode(node.child)}'
        elif op_name == 'Addfix':
            return f'addfix_{encode(node.child)}'
        elif op_name == 'Sjf1':
            return f'sjf1_{encode(node.child)}'
        elif op_name == 'Duelsum':
            return f'duelsum_{encode(node.child)}'
        elif op_name == 'Netrm':
            return f'netrm_{encode(node.child)}'
        elif op_name == 'Convfix':
            return f'convfix_{encode(node.child)}'
        elif op_name == 'Usum':
            return f'usum_{encode(node.child)}'
        elif op_name == 'Perase0':
            return f'perase0_{encode(node.child)}'
        elif op_name == 'Lagf3':
            return f'lagf3_{encode(node.child)}'
        elif op_name == 'Nearidx':
            return f'nearidx_{encode(node.child)}'
        elif op_name == 'Tresetsum3':
            return f'tresetsum3_{encode(node.child)}'
        elif op_name == 'Oepass0':
            return f'oepass0_{encode(node.child)}'
        elif op_name == 'Pge':
            return f'pge_{encode(node.child)}'
        elif op_name == 'Cgpmax09':
            return f'cgpmax09_{encode(node.child)}'
        elif op_name == 'Lanesum':
            return f'lanesum_{encode(node.child)}'
        elif op_name == 'Addpmax':
            return f'addpmax_{encode(node.child)}'
        elif op_name == 'Maxrun':
            return f'maxrun_{encode(node.child)}'
        elif op_name == 'Dualtie':
            return f'dualtie_{encode(node.child)}'
        elif op_name == 'Cap59':
            return f'cap59_{encode(node.child)}'
        elif op_name == 'Cgrev09':
            return f'cgrev09_{encode(node.child)}'
        elif op_name == 'Dupmax':
            return f'dupmax_{encode(node.child)}'
        elif op_name == 'Rmaj5':
            return f'rmaj5_{encode(node.child)}'
        elif op_name == 'Majmask':
            return f'majmask_{encode(node.child)}'
        elif op_name == 'AnchorBroadcast':
            return f'bcast0_{encode(node.child)}'
        elif op_name == 'Zpeak':
            return f'zpeak_{encode(node.child)}'
        elif op_name == 'Mulanc0':
            return f'mulanc0_{encode(node.child)}'
        elif op_name == 'Recdist':
            return f'recdist_{encode(node.child)}'
        elif op_name == 'Pmrep':
            return f'pmrep_{encode(node.child)}'
        elif op_name == 'Mulpair':
            return f'mulpair_{encode(node.child)}'
        elif op_name == 'PrevSegmentMax':
            return f'prevsegmax3_{encode(node.child)}'
        elif op_name == 'Occcnt':
            return f'occcnt_{encode(node.child)}'
        elif op_name == 'SegmentDistinct':
            return f'segdist3_{encode(node.child)}'
        elif op_name == 'Dclamp':
            return f'dclamp_{encode(node.child)}'
        elif op_name == 'FirstUnique':
            return f'firstuniq_{encode(node.child)}'
        elif op_name == 'Runmed':
            return f'runmed_{encode(node.child)}'
        elif op_name == 'Cntge3':
            return f'cntge3_{encode(node.child)}'
        elif op_name == 'Twosum10':
            return f'twosum10_{encode(node.child)}'
        elif op_name == 'Holdodd':
            return f'holdodd_{encode(node.child)}'
        elif op_name == 'Addlastodd':
            return f'addlastodd_{encode(node.child)}'
        elif op_name == 'Bmcand':
            return f'bmcand_{encode(node.child)}'
        elif op_name == 'Hold2ge5':
            return f'hold2ge5_{encode(node.child)}'
        elif op_name == 'Sjf2':
            return f'sjf2_{encode(node.child)}'
        elif op_name == 'Gapge5':
            return f'gapge5_{encode(node.child)}'
        elif op_name == 'Riseerase':
            return f'riseerase_{encode(node.child)}'
        elif op_name == 'Ebud3':
            return f'ebud3_{encode(node.child)}'
        elif op_name == 'Tieswap':
            return f'tieswap_{encode(node.child)}'
        elif op_name == 'Risemax':
            return f'risemax_{encode(node.child)}'
        elif op_name == 'Delc2':
            return f'delc2_{encode(node.child)}'
        elif op_name == 'Riskeep':
            return f'riskeep_{encode(node.child)}'
        elif op_name == 'Mixtie':
            return f'mixtie_{encode(node.child)}'
        elif op_name == 'Pmaxv89':
            return f'pmaxv89_{encode(node.child)}'
        elif op_name == 'Pmaxv789':
            return f'pmaxv789_{encode(node.child)}'
        elif op_name == 'Togsum0':
            return f'togsum0_{encode(node.child)}'
        elif op_name == 'Signsum0':
            return f'signsum0_{encode(node.child)}'
        elif op_name == 'Median3':
            return f'med3_{encode(node.child)}'
        elif op_name == 'Majority3':
            return f'maj3_{encode(node.child)}'
        elif op_name == 'Spile4':
            return f'spile4_{encode(node.child)}'
        elif op_name == 'Spile5':
            return f'spile5_{encode(node.child)}'
        elif op_name == 'Spile6':
            return f'spile6_{encode(node.child)}'
        elif op_name == 'Cyca':
            return f'cyca_{encode(node.child)}'
        elif op_name == 'Rule184Thr4':
            return f'r184t4_{encode(node.child)}'
        elif op_name == 'Rule184Thr5':
            return f'r184t5_{encode(node.child)}'
        elif op_name == 'Rule184Thr6':
            return f'r184t6_{encode(node.child)}'
        elif op_name == 'Ewma2':
            return f'ewma2_{encode(node.child)}'
        elif op_name == 'Zcross5':
            return f'zcross5_{encode(node.child)}'
        elif op_name == 'Heapmax':
            return f'heapmax_{encode(node.child)}'
        elif op_name == 'Vig2':
            return f'vig2_{encode(node.child)}'
        elif op_name == 'Vig3':
            return f'vig3_{encode(node.child)}'
        elif op_name == 'Vig5':
            return f'vig5_{encode(node.child)}'
        elif op_name == 'Hamk3':
            return f'hamk3_{encode(node.child)}'
        elif op_name == 'Altrun':
            return f'altrun_{encode(node.child)}'
        elif op_name == 'BlockSwapParity':
            return f'blkpar_{encode(node.child)}'
        elif op_name == 'BrickSort2':
            return f'blkoe2_{encode(node.child)}'
        elif op_name == 'FeistelAdd':
            return f'blkadd_{encode(node.child)}'
        elif op_name == 'BlockMax2':
            return f'blkmax_{encode(node.child)}'
        elif op_name == 'ErrorDiffuse2':
            return f'errdiff2_{encode(node.child)}'
        elif op_name == 'ErrorDiffuse3':
            return f'errdiff3_{encode(node.child)}'
        elif op_name == 'ErrorDiffuse5':
            return f'errdiff5_{encode(node.child)}'
        elif op_name == 'BoxSum2':
            return f'boxsum2_{encode(node.child)}'
        elif op_name == 'BoxSum3':
            return f'boxsum3_{encode(node.child)}'
        elif op_name == 'BoxSum4':
            return f'boxsum4_{encode(node.child)}'
        elif op_name == 'BoxSum5':
            return f'boxsum5_{encode(node.child)}'
        elif op_name == 'Schmitt':
            return f'schmitt_{encode(node.child)}'
        elif op_name == 'RailFence2':
            return f'railfence2_{encode(node.child)}'
        elif op_name == 'RailFence3':
            return f'railfence3_{encode(node.child)}'
        elif op_name == 'RailFence4':
            return f'railfence4_{encode(node.child)}'
        elif op_name == 'QueueWalk3':
            return f'qlen3_{encode(node.child)}'
        elif op_name == 'QueueWalk4':
            return f'qlen4_{encode(node.child)}'
        elif op_name == 'QueueWalk5':
            return f'qlen5_{encode(node.child)}'
        elif op_name == 'QueueWalk6':
            return f'qlen6_{encode(node.child)}'
        elif op_name == 'MinBottleneck':
            return f'bpath_{encode(node.child)}'
        elif op_name == 'MaxCapacity':
            return f'wpath_{encode(node.child)}'
        elif op_name == 'HeapSubtreeMax':
            return f'hsub_{encode(node.child)}'
        elif op_name == 'IncRunMax':
            return f'incrun_{encode(node.child)}'
        elif op_name == 'HaarStep':
            return f'haar_{encode(node.child)}'
        elif op_name == 'RunMAD':
            return f'runmad_{encode(node.child)}'
        elif op_name == 'ContFrac':
            return f'contfrac_{encode(node.child)}'
        elif op_name == 'GroupProductS3':
            return f's3prod_{encode(node.child)}'
        elif op_name == 'GroupProductD4':
            return f'd4prod_{encode(node.child)}'
        elif op_name == 'GroupProductQ8':
            return f'q8prod_{encode(node.child)}'
        elif op_name == 'GroupProductD5':
            return f'd5prod_{encode(node.child)}'
        elif op_name == 'D4Character':
            return f'd4char_{encode(node.child)}'
        elif op_name == 'GroupActionD5':
            return f'd5act_{encode(node.child)}'
        elif op_name == 'RunningRangeAdd':
            return f'rangeadd_{encode(node.child)}'
        elif op_name == 'RunningMedianAdd':
            return f'medadd_{encode(node.child)}'
        elif op_name == 'ThirdOrderScan':
            return f'tribscan_{encode(node.child)}'
        elif op_name == 'LinearRotor3':
            return f'rotor3_{encode(node.child)}'
        elif op_name == 'RunningModeCountAdd':
            return f'modecntadd_{encode(node.child)}'
        elif op_name == 'Oscpos':
            return f'oscpos_{encode(node.child)}'
        elif op_name == 'Oscmom':
            return f'oscmom_{encode(node.child)}'
        elif op_name == 'RefractoryVeto':
            return f'refrac_{encode(node.child)}'
        elif op_name == 'MexicanHatVeto':
            return f'mexhat_{encode(node.child)}'
        elif op_name == 'MaxFrequency':
            return f'maxfreq_{encode(node.child)}'
        elif op_name == 'DucciStep':
            return f'ducci_{encode(node.child)}'
        elif op_name == 'PairGcd':
            return f'pgcd_{encode(node.child)}'
        elif op_name == 'SecondOrderScan':
            return f'fibscan_{encode(node.child)}'
        elif op_name == 'NivenByCount':
            return f'meanint_{encode(node.child)}'
        elif op_name == 'EuclidStep':
            return f'euclstep_{encode(node.child)}'
        elif op_name == 'PairCarry':
            return f'paircarry_{encode(node.child)}'
        elif op_name == 'InversionCount':
            return f'invcount_{encode(node.child)}'
        elif op_name == 'MajorIndex':
            return f'majindex_{encode(node.child)}'
        elif op_name == 'NimHistogram':
            return f'nimhist_{encode(node.child)}'
        elif op_name == 'MeanRelax':
            return f'graviclump_{encode(node.child)}'
        elif op_name == 'AnchorNudge':
            return f'entrain_{encode(node.child)}'
        elif op_name == 'PhaseNudge':
            return f'phaselock_{encode(node.child)}'
        elif op_name == 'HeatExchange':
            return f'heatx_{encode(node.child)}'
        elif op_name == 'ElemSym2':
            return f'e2sym_{encode(node.child)}'
        elif op_name == 'RunningMex':
            return f'runmex_{encode(node.child)}'
        elif op_name == 'MirrorAdd':
            return f'paladd_{encode(node.child)}'
        elif op_name == 'MirrorMax':
            return f'palmax_{encode(node.child)}'
        elif op_name == 'MirrorMin':
            return f'palmin_{encode(node.child)}'
        elif op_name == 'OnlineRankAdd':
            return f'rankadd_{encode(node.child)}'
        elif op_name == 'SignDiffuse':
            return f'signdiff_{encode(node.child)}'
        elif op_name == 'DoubleGE':
            return f'dblge5_{encode(node.child)}'
        elif op_name == 'MidrangeAdd':
            return f'midadd_{encode(node.child)}'
        elif op_name == 'SpacedSeedSum':
            return f'seedsum_{encode(node.child)}'
        elif op_name == 'ArgPosAdd':
            return f'argposadd_{encode(node.child)}'
        elif op_name == 'LaneRunningMax':
            return f'lanemax_{encode(node.child)}'
        elif op_name == 'PancakeFlip':
            return f'pancake_{encode(node.child)}'
        elif op_name == 'BitPlaneMajority':
            return f'bitmaj_{encode(node.child)}'
        elif op_name == 'PredicatedSkip':
            return f'skipn5_{encode(node.child)}'
        elif op_name == 'BitTransitionCount':
            return f'bittv_{encode(node.child)}'
        elif op_name == 'OpcodeMachine':
            return f'vm1_{encode(node.child)}'
        elif op_name == 'KnockoutDepth':
            return f'tourney_{encode(node.child)}'
        elif op_name == 'EnvelopeMax':
            return f'envmax_{encode(node.child)}'
        elif op_name == 'CountMinSketch':
            return f'cmin_{encode(node.child)}'
        elif op_name == 'BitTransposeGather':
            return f'bittrans_{encode(node.child)}'
        elif op_name == 'InternRank':
            return f'intern_{encode(node.child)}'
        elif op_name == 'MultiplicityParityXor':
            return f'dupxor_{encode(node.child)}'
        elif op_name == 'ProximityDegree':
            return f'nbrdeg_{encode(node.child)}'
        elif op_name == 'CumFreqRank':
            return f'cfreq_{encode(node.child)}'
        elif op_name == 'DistinctSubseqCount':
            return f'dsub_{encode(node.child)}'
        elif op_name == 'CodonSkip':
            return f'codonskip_{encode(node.child)}'
        elif op_name == 'FragLen':
            return f'fraglen_{encode(node.child)}'
        elif op_name == 'DivisorZeta':
            return f'divzeta_{encode(node.child)}'
        elif op_name == 'MobiusInvert':
            return f'mobinv_{encode(node.child)}'
        elif op_name == 'FactoradicNormalize':
            return f'factnorm_{encode(node.child)}'
        elif op_name == 'NonAdjacentMax4':
            return f'nonadj4_{encode(node.child)}'
        elif op_name == 'BoundedConfidenceMean':
            return f'hkconf_{encode(node.child)}'
        elif op_name == 'ExtremalReplace':
            return f'baksnep_{encode(node.child)}'
        elif op_name == 'Hsifte':
            return f'hsifte_{encode(node.child)}'
        elif op_name == 'LinearProbe':
            return f'lprobe_{encode(node.child)}'
        elif op_name == 'FlagPartition':
            return f'dutch_{encode(node.child)}'
        elif op_name == 'Slew2':
            return f'slew2_{encode(node.child)}'
        elif op_name == 'Rms3':
            return f'rms3_{encode(node.child)}'
        elif op_name == 'Spn2':
            return f'spn2_{encode(node.child)}'
        elif op_name == 'Xpar':
            return f'xpar_{encode(node.child)}'
        else:
            raise ValueError(f"Unknown operator: {op_name}")

    return encode(expr)


def base_algorithm_id(name: str) -> str:
    """The DSL algorithm_id behind a model name (strips a trailing ``__<variant>``)."""
    return name.split('__', 1)[0]


def decode_algorithm_id(algo_id: str, input_length: int = 10) -> DSLExpr:
    """
    Decode algorithm ID back into DSL expression.

    This is the inverse of generate_algorithm_id().

    Examples:
        "id" → Id(10)
        "rev_id" → Reverse(Id(10))
        "take5_madd2_id" → Take(5, MapAdd(2, Id(10)))
        "pad7f0_rev_take3_id" → PadTo(7, 0, Reverse(Take(3, Id(10))))

    Args:
        algo_id: Algorithm ID string
        input_length: Input sequence length (for Id nodes)

    Returns:
        DSLExpr tree
    """
    # Split by underscore to get operation sequence (outermost to innermost)
    parts = algo_id.split('_')

    def parse_op(part: str):
        """Parse single operation string into (op_name, params)."""
        if part == 'id':
            return ('Id', {})
        elif part == 'rev':
            return ('Reverse', {})
        elif m := re.match(r'c(\d)L(\d+)', part):
            return ('Const', {'digit': int(m.group(1)), 'length': int(m.group(2))})
        elif m := re.match(r'rot(-?\d+)', part):
            return ('Rotate', {'k': int(m.group(1))})
        elif m := re.match(r'take(\d+)', part):
            return ('Take', {'k': int(m.group(1))})
        elif m := re.match(r'drop(\d+)', part):
            return ('Drop', {'k': int(m.group(1))})
        elif m := re.match(r'madd(\d+)', part):
            return ('MapAdd', {'c': int(m.group(1))})
        elif m := re.match(r'mmul(\d+)', part):
            return ('MapMul', {'c': int(m.group(1))})
        elif part == 'dblodd':
            return ('DoubleOdd', {})
        elif m := re.match(r'pad(\d+)f(\d)', part):
            return ('PadTo', {'M': int(m.group(1)), 'fill': int(m.group(2))})
        elif m := re.match(r'jump(\d+)', part):
            return ('JumpIndex', {'T': int(m.group(1))})
        elif part == 'pmax':
            return ('PrefixMax', {})
        elif part == 'psumodd':
            return ('PrefixSumOdd', {})
        elif part == 'psum':
            return ('PrefixSumMod', {})
        elif part == 'prod':
            return ('RunningProduct', {})
        elif part == 'cntodd':
            return ('CountOdd', {})
        elif part == 'recordind':
            return ('RecordIndicator', {})
        elif part == 'novel':
            return ('NoveltyIndicator', {})
        elif part == 'fixedpoint':
            return ('FixedPointIndicator', {})
        elif m := re.match(r'reset(\d)', part):
            return ('ResetRamp', {'d': int(m.group(1))})
        elif m := re.match(r'tresetmax(\d)', part):
            return ('TokenResetMax', {'d': int(m.group(1))})
        elif m := re.match(r'walk(\d)(\d)', part):
            return ('SignedWalk', {'up': int(m.group(1)), 'down': int(m.group(2))})
        elif m := re.match(r'sr(\d)(\d)', part):
            return ('SRLatch', {'set_tok': int(m.group(1)), 'reset_tok': int(m.group(2))})
        elif m := re.match(r'hold(ge|le)(\d)', part):
            return ('SampleHold', {'mode': m.group(1), 'thresh': int(m.group(2))})
        elif m := re.match(r'dep(\d)f(\d)', part):
            return ('Deplete', {'start': int(m.group(1)), 'floor': int(m.group(2))})
        elif m := re.match(r'anc(\d)', part):
            return ('AnchorSubtract', {'a': int(m.group(1))})
        elif m := re.match(r'(f|c)clamp(\d)', part):
            return ('AnchorClamp', {'mode': 'floor' if m.group(1) == 'f' else 'ceil', 'a': int(m.group(2))})
        elif m := re.match(r'drot(\d)', part):
            return ('DataRotate', {'a': int(m.group(1))})
        elif m := re.match(r'refl(\d)', part):
            return ('DataReflect', {'a': int(m.group(1))})
        elif m := re.match(r'stride(\d)', part):
            return ('StrideGather', {'a': int(m.group(1))})
        elif m := re.match(r'pref(\d)', part):
            return ('PrefixReverseK', {'a': int(m.group(1))})
        elif m := re.match(r'psort(\d)', part):
            return ('PrefixSortK', {'a': int(m.group(1))})
        elif m := re.match(r'cmpswap(\d)p(\d)', part):
            return ('CompareSwap', {'i': int(m.group(1)), 'j': int(m.group(2))})
        elif m := re.match(r'bmax(\d)', part):
            return ('WindowMax', {'w': int(m.group(1))})
        elif part == 'pmstep':
            return ('ParityRoute', {})
        elif m := re.match(r'carryk(\d)', part):
            return ('RippleCarry', {'k': int(m.group(1))})
        elif part == 'argminmax':
            return ('ArgMinMax', {})
        elif m := re.match(r'rcnt(\d)', part):
            return ('RunningCountDigit', {'d': int(m.group(1))})
        elif m := re.match(r'cmpsh(\d+)', part):
            return ('CompareShift', {'k': int(m.group(1))})
        elif m := re.match(r'diffsh(\d+)', part):
            return ('DiffShift', {'k': int(m.group(1))})
        elif m := re.match(r'hscan(\d)b(\d)c(\d)s(\d)', part):
            return ('HashScan', {'a': int(m.group(1)), 'b': int(m.group(2)), 'c': int(m.group(3)), 'seed': int(m.group(4))})
        elif m := re.match(r'whash(\d+)w(\d+)a(\d)b(\d)c(\d)', part):
            return ('WindowHash', {'T': int(m.group(1)), 'w': int(m.group(2)), 'a': int(m.group(3)), 'b': int(m.group(4)), 'c': int(m.group(5))})
        elif m := re.match(r'imix(\d+)a(\d)b(\d)', part):
            return ('IndexMix', {'T': int(m.group(1)), 'a': int(m.group(2)), 'b': int(m.group(3))})
        elif m := re.match(r'prank(\d+)', part):
            return ('PermuteByRank', {'T': int(m.group(1))})
        elif m := re.match(r'cwalk(\d+)', part):
            return ('CycleWalk', {'T': int(m.group(1))})
        elif m := re.match(r'tpmerge(\d+)', part):
            return ('TwoPointerMerge', {'T': int(m.group(1))})
        elif m := re.match(r'cjump(\d+)t(\d)', part):
            return ('ConditionalJump', {'T': int(m.group(1)), 'threshold': int(m.group(2))})
        elif m := re.match(r'tdfs(\d+)', part):
            return ('TreeWalkDFS', {'T': int(m.group(1))})
        elif m := re.match(r'tbfs(\d+)', part):
            return ('TreeWalkBFS', {'T': int(m.group(1))})
        elif m := re.match(r'ca(\d+)a(\d)b(\d)c(\d)', part):
            return ('CellularAutomaton', {'steps': int(m.group(1)), 'a': int(m.group(2)), 'b': int(m.group(3)), 'c': int(m.group(4))})
        elif m := re.match(r'lfsr(\d+)w(\d+)t(\d+)t(\d+)c(\d)c(\d)', part):
            return ('LFSR', {'T': int(m.group(1)), 'width': int(m.group(2)), 'tap1': int(m.group(3)), 'tap2': int(m.group(4)), 'c1': int(m.group(5)), 'c2': int(m.group(6))})
        elif m := re.match(r'parity(\d)e(\d)o(\d)f(\d)', part):
            return ('ParityInterleave', {'a': int(m.group(1)), 'b': int(m.group(2)), 'c': int(m.group(3)), 'd': int(m.group(4))})
        elif m := re.match(r'rtree(\d+)a(\d)b(\d)', part):
            return ('ReductionTree', {'T': int(m.group(1)), 'a': int(m.group(2)), 'b': int(m.group(3))})
        elif m := re.match(r'chain(\d+)d(\d+)', part):
            return ('ChainedIndex', {'T': int(m.group(1)), 'depth': int(m.group(2))})
        elif m := re.match(r'sconv(\d+)', part):
            return ('SelfConvolve', {'T': int(m.group(1))})
        elif m := re.match(r'areset(\d+)', part):
            return ('AccumulateReset', {'threshold': int(m.group(1))})
        elif m := re.match(r'scatter(\d+)a(\d)', part):
            return ('ScatterWrite', {'T': int(m.group(1)), 'a': int(m.group(2))})
        elif m := re.match(r'invlook(\d+)', part):
            return ('InverseLookup', {'T': int(m.group(1))})
        elif m := re.match(r'rle(\d+)', part):
            return ('RunLengthEncode', {'T': int(m.group(1))})
        elif m := re.match(r'csort(\d+)to(\d+)([ad])', part):
            return ('ChunkSort', {'i': int(m.group(1)), 'j': int(m.group(2)), 'asc': m.group(3) == 'a'})
        elif m := re.match(r'segrev(\d+)', part):
            return ('SegmentReverse', {'boundary_digit': int(m.group(1))})
        elif part == 'xorscan':
            return ('XORScan', {})
        elif m := re.match(r'modrem(\d+)', part):
            return ('ModRemainder', {'m': int(m.group(1))})
        elif part == 'zigzag':
            return ('ZigzagCount', {})
        elif part == 'reccount':
            return ('RecordCount', {})
        elif part == 'kmpfail':
            return ('KMPFailure', {})
        elif part == 'bwt':
            return ('BWT', {})
        elif part == 'lz78':
            return ('LZ78Complexity', {})
        elif m := re.match(r'vaneck(\d+)', part):
            return ('VanEckExtension', {'T': int(m.group(1))})
        elif m := re.match(r'swscan(\d)e(\d)f(\d)g(\d)t(\d)', part):
            return ('SwitchedScan', {'a1': int(m.group(1)), 'b1': int(m.group(2)),
                                     'a2': int(m.group(3)), 'b2': int(m.group(4)),
                                     'thresh': int(m.group(5))})
        elif part == 'sarank':
            return ('Sarank', {})
        elif m := re.match(r'mux(\d+)\.(.+)$', part):
            member_ids = [p.replace('-', '_') for p in m.group(2).split('.')]
            if len(member_ids) != int(m.group(1)):
                raise ValueError(f"TaskMux arity {m.group(1)} != "
                                 f"{len(member_ids)} members in {part!r}")
            return ('TaskMux', {'member_ids': member_ids})
        elif part == 'pmaxv9':
            return ('Pmaxv9', {})
        elif part == 'dfa3t010020001':
            return ('SmallDFATrace', {})
        elif part == 'indecho':
            return ('Indecho', {})
        elif part == 'cosetprod':
            return ('GroupCosetProductS3', {})
        elif part == 'kanc2':
            return ('Kanc2', {})
        elif part == 'kanc4':
            return ('Kanc4', {})
        elif part == 'kanc8':
            return ('Kanc8', {})
        elif part == 'kanc10':
            return ('Kanc10', {})
        elif part == 'sanc2':
            return ('Sanc2', {})
        elif part == 'tmap5':
            return ('Tmap5', {})
        elif part == 'addfix':
            return ('Addfix', {})
        elif part == 'sjf1':
            return ('Sjf1', {})
        elif part == 'duelsum':
            return ('Duelsum', {})
        elif part == 'netrm':
            return ('Netrm', {})
        elif part == 'convfix':
            return ('Convfix', {})
        elif part == 'usum':
            return ('Usum', {})
        elif part == 'perase0':
            return ('Perase0', {})
        elif part == 'lagf3':
            return ('Lagf3', {})
        elif part == 'nearidx':
            return ('Nearidx', {})
        elif part == 'tresetsum3':
            return ('Tresetsum3', {})
        elif part == 'oepass0':
            return ('Oepass0', {})
        elif part == 'pge':
            return ('Pge', {})
        elif part == 'cgpmax09':
            return ('Cgpmax09', {})
        elif part == 'lanesum':
            return ('Lanesum', {})
        elif part == 'addpmax':
            return ('Addpmax', {})
        elif part == 'maxrun':
            return ('Maxrun', {})
        elif part == 'dualtie':
            return ('Dualtie', {})
        elif part == 'cap59':
            return ('Cap59', {})
        elif part == 'cgrev09':
            return ('Cgrev09', {})
        elif part == 'dupmax':
            return ('Dupmax', {})
        elif part == 'rmaj5':
            return ('Rmaj5', {})
        elif part == 'majmask':
            return ('Majmask', {})
        elif part == 'bcast0':
            return ('AnchorBroadcast', {})
        elif part == 'zpeak':
            return ('Zpeak', {})
        elif part == 'mulanc0':
            return ('Mulanc0', {})
        elif part == 'recdist':
            return ('Recdist', {})
        elif part == 'pmrep':
            return ('Pmrep', {})
        elif part == 'mulpair':
            return ('Mulpair', {})
        elif part == 'prevsegmax3':
            return ('PrevSegmentMax', {})
        elif part == 'occcnt':
            return ('Occcnt', {})
        elif part == 'segdist3':
            return ('SegmentDistinct', {})
        elif part == 'dclamp':
            return ('Dclamp', {})
        elif part == 'firstuniq':
            return ('FirstUnique', {})
        elif part == 'runmed':
            return ('Runmed', {})
        elif part == 'cntge3':
            return ('Cntge3', {})
        elif part == 'twosum10':
            return ('Twosum10', {})
        elif part == 'holdodd':
            return ('Holdodd', {})
        elif part == 'addlastodd':
            return ('Addlastodd', {})
        elif part == 'bmcand':
            return ('Bmcand', {})
        elif part == 'hold2ge5':
            return ('Hold2ge5', {})
        elif part == 'sjf2':
            return ('Sjf2', {})
        elif part == 'gapge5':
            return ('Gapge5', {})
        elif part == 'riseerase':
            return ('Riseerase', {})
        elif part == 'ebud3':
            return ('Ebud3', {})
        elif part == 'tieswap':
            return ('Tieswap', {})
        elif part == 'risemax':
            return ('Risemax', {})
        elif part == 'delc2':
            return ('Delc2', {})
        elif part == 'riskeep':
            return ('Riskeep', {})
        elif part == 'mixtie':
            return ('Mixtie', {})
        elif part == 'pmaxv89':
            return ('Pmaxv89', {})
        elif part == 'pmaxv789':
            return ('Pmaxv789', {})
        elif part == 'togsum0':
            return ('Togsum0', {})
        elif part == 'signsum0':
            return ('Signsum0', {})
        elif part == 'med3':
            return ('Median3', {})
        elif part == 'maj3':
            return ('Majority3', {})
        elif part == 'spile4':
            return ('Spile4', {})
        elif part == 'spile5':
            return ('Spile5', {})
        elif part == 'spile6':
            return ('Spile6', {})
        elif part == 'cyca':
            return ('Cyca', {})
        elif part == 'r184t4':
            return ('Rule184Thr4', {})
        elif part == 'r184t5':
            return ('Rule184Thr5', {})
        elif part == 'r184t6':
            return ('Rule184Thr6', {})
        elif part == 'ewma2':
            return ('Ewma2', {})
        elif part == 'zcross5':
            return ('Zcross5', {})
        elif part == 'heapmax':
            return ('Heapmax', {})
        elif part == 'vig2':
            return ('Vig2', {})
        elif part == 'vig3':
            return ('Vig3', {})
        elif part == 'vig5':
            return ('Vig5', {})
        elif part == 'hamk3':
            return ('Hamk3', {})
        elif part == 'altrun':
            return ('Altrun', {})
        elif part == 'blkpar':
            return ('BlockSwapParity', {})
        elif part == 'blkoe2':
            return ('BrickSort2', {})
        elif part == 'blkadd':
            return ('FeistelAdd', {})
        elif part == 'blkmax':
            return ('BlockMax2', {})
        elif part == 'errdiff2':
            return ('ErrorDiffuse2', {})
        elif part == 'errdiff3':
            return ('ErrorDiffuse3', {})
        elif part == 'errdiff5':
            return ('ErrorDiffuse5', {})
        elif part == 'boxsum2':
            return ('BoxSum2', {})
        elif part == 'boxsum3':
            return ('BoxSum3', {})
        elif part == 'boxsum4':
            return ('BoxSum4', {})
        elif part == 'boxsum5':
            return ('BoxSum5', {})
        elif part == 'schmitt':
            return ('Schmitt', {})
        elif part == 'railfence2':
            return ('RailFence2', {})
        elif part == 'railfence3':
            return ('RailFence3', {})
        elif part == 'railfence4':
            return ('RailFence4', {})
        elif part == 'qlen3':
            return ('QueueWalk3', {})
        elif part == 'qlen4':
            return ('QueueWalk4', {})
        elif part == 'qlen5':
            return ('QueueWalk5', {})
        elif part == 'qlen6':
            return ('QueueWalk6', {})
        elif part == 'bpath':
            return ('MinBottleneck', {})
        elif part == 'wpath':
            return ('MaxCapacity', {})
        elif part == 'hsub':
            return ('HeapSubtreeMax', {})
        elif part == 'incrun':
            return ('IncRunMax', {})
        elif part == 'haar':
            return ('HaarStep', {})
        elif part == 'runmad':
            return ('RunMAD', {})
        elif part == 'contfrac':
            return ('ContFrac', {})
        elif part == 's3prod':
            return ('GroupProductS3', {})
        elif part == 'd4prod':
            return ('GroupProductD4', {})
        elif part == 'q8prod':
            return ('GroupProductQ8', {})
        elif part == 'd5prod':
            return ('GroupProductD5', {})
        elif part == 'd4char':
            return ('D4Character', {})
        elif part == 'd5act':
            return ('GroupActionD5', {})
        elif part == 'rangeadd':
            return ('RunningRangeAdd', {})
        elif part == 'medadd':
            return ('RunningMedianAdd', {})
        elif part == 'tribscan':
            return ('ThirdOrderScan', {})
        elif part == 'rotor3':
            return ('LinearRotor3', {})
        elif part == 'modecntadd':
            return ('RunningModeCountAdd', {})
        elif part == 'oscpos':
            return ('Oscpos', {})
        elif part == 'oscmom':
            return ('Oscmom', {})
        elif part == 'refrac':
            return ('RefractoryVeto', {})
        elif part == 'mexhat':
            return ('MexicanHatVeto', {})
        elif part == 'maxfreq':
            return ('MaxFrequency', {})
        elif part == 'ducci':
            return ('DucciStep', {})
        elif part == 'pgcd':
            return ('PairGcd', {})
        elif part == 'fibscan':
            return ('SecondOrderScan', {})
        elif part == 'meanint':
            return ('NivenByCount', {})
        elif part == 'euclstep':
            return ('EuclidStep', {})
        elif part == 'paircarry':
            return ('PairCarry', {})
        elif part == 'invcount':
            return ('InversionCount', {})
        elif part == 'majindex':
            return ('MajorIndex', {})
        elif part == 'nimhist':
            return ('NimHistogram', {})
        elif part == 'graviclump':
            return ('MeanRelax', {})
        elif part == 'entrain':
            return ('AnchorNudge', {})
        elif part == 'phaselock':
            return ('PhaseNudge', {})
        elif part == 'heatx':
            return ('HeatExchange', {})
        elif part == 'e2sym':
            return ('ElemSym2', {})
        elif part == 'runmex':
            return ('RunningMex', {})
        elif part == 'paladd':
            return ('MirrorAdd', {})
        elif part == 'palmax':
            return ('MirrorMax', {})
        elif part == 'palmin':
            return ('MirrorMin', {})
        elif part == 'rankadd':
            return ('OnlineRankAdd', {})
        elif part == 'signdiff':
            return ('SignDiffuse', {})
        elif part == 'dblge5':
            return ('DoubleGE', {})
        elif part == 'midadd':
            return ('MidrangeAdd', {})
        elif part == 'seedsum':
            return ('SpacedSeedSum', {})
        elif part == 'argposadd':
            return ('ArgPosAdd', {})
        elif part == 'lanemax':
            return ('LaneRunningMax', {})
        elif part == 'pancake':
            return ('PancakeFlip', {})
        elif part == 'bitmaj':
            return ('BitPlaneMajority', {})
        elif part == 'skipn5':
            return ('PredicatedSkip', {})
        elif part == 'bittv':
            return ('BitTransitionCount', {})
        elif part == 'vm1':
            return ('OpcodeMachine', {})
        elif part == 'tourney':
            return ('KnockoutDepth', {})
        elif part == 'envmax':
            return ('EnvelopeMax', {})
        elif part == 'cmin':
            return ('CountMinSketch', {})
        elif part == 'bittrans':
            return ('BitTransposeGather', {})
        elif part == 'intern':
            return ('InternRank', {})
        elif part == 'dupxor':
            return ('MultiplicityParityXor', {})
        elif part == 'nbrdeg':
            return ('ProximityDegree', {})
        elif part == 'cfreq':
            return ('CumFreqRank', {})
        elif part == 'dsub':
            return ('DistinctSubseqCount', {})
        elif part == 'codonskip':
            return ('CodonSkip', {})
        elif part == 'fraglen':
            return ('FragLen', {})
        elif part == 'divzeta':
            return ('DivisorZeta', {})
        elif part == 'mobinv':
            return ('MobiusInvert', {})
        elif part == 'factnorm':
            return ('FactoradicNormalize', {})
        elif part == 'nonadj4':
            return ('NonAdjacentMax4', {})
        elif part == 'hkconf':
            return ('BoundedConfidenceMean', {})
        elif part == 'baksnep':
            return ('ExtremalReplace', {})
        elif part == 'hsifte':
            return ('Hsifte', {})
        elif part == 'lprobe':
            return ('LinearProbe', {})
        elif part == 'dutch':
            return ('FlagPartition', {})
        elif part == 'slew2':
            return ('Slew2', {})
        elif part == 'rms3':
            return ('Rms3', {})
        elif part == 'spn2':
            return ('Spn2', {})
        elif part == 'xpar':
            return ('Xpar', {})
        else:
            raise ValueError(f"Unknown operation code: {part}")

    # Build tree from outside-in (reverse parts to go innermost-first)
    parts_reversed = parts[::-1]

    expr = None
    for part in parts_reversed:
        op_name, params = parse_op(part)

        if op_name == 'Id':
            expr = Id(input_length)
        elif op_name == 'Const':
            expr = Const(params['digit'], params['length'], input_length)
        elif op_name == 'Reverse':
            if expr is None:
                raise ValueError("Reverse requires a child expression")
            expr = Reverse(expr)
        elif op_name == 'Rotate':
            if expr is None:
                raise ValueError("Rotate requires a child expression")
            expr = Rotate(params['k'], expr)
        elif op_name == 'Take':
            if expr is None:
                raise ValueError("Take requires a child expression")
            expr = Take(params['k'], expr)
        elif op_name == 'Drop':
            if expr is None:
                raise ValueError("Drop requires a child expression")
            expr = Drop(params['k'], expr)
        elif op_name == 'MapAdd':
            if expr is None:
                raise ValueError("MapAdd requires a child expression")
            expr = MapAdd(params['c'], expr)
        elif op_name == 'MapMul':
            if expr is None:
                raise ValueError("MapMul requires a child expression")
            expr = MapMul(params['c'], expr)
        elif op_name == 'DoubleOdd':
            if expr is None:
                raise ValueError("DoubleOdd requires a child expression")
            expr = DoubleOdd(expr)
        elif op_name == 'PadTo':
            if expr is None:
                raise ValueError("PadTo requires a child expression")
            expr = PadTo(params['M'], params['fill'], expr)
        elif op_name == 'JumpIndex':
            if expr is None:
                raise ValueError("JumpIndex requires a child expression")
            expr = JumpIndex(params['T'], expr)
        elif op_name == 'PrefixMax':
            if expr is None:
                raise ValueError("PrefixMax requires a child expression")
            expr = PrefixMax(expr)
        elif op_name == 'PrefixSumMod':
            if expr is None:
                raise ValueError("PrefixSumMod requires a child expression")
            expr = PrefixSumMod(expr)
        elif op_name == 'PrefixSumOdd':
            if expr is None:
                raise ValueError("PrefixSumOdd requires a child expression")
            expr = PrefixSumOdd(expr)
        elif op_name == 'RunningProduct':
            if expr is None:
                raise ValueError("RunningProduct requires a child expression")
            expr = RunningProduct(expr)
        elif op_name == 'CountOdd':
            if expr is None:
                raise ValueError("CountOdd requires a child expression")
            expr = CountOdd(expr)
        elif op_name == 'RecordIndicator':
            if expr is None:
                raise ValueError("RecordIndicator requires a child expression")
            expr = RecordIndicator(expr)
        elif op_name == 'NoveltyIndicator':
            if expr is None:
                raise ValueError("NoveltyIndicator requires a child expression")
            expr = NoveltyIndicator(expr)
        elif op_name == 'FixedPointIndicator':
            if expr is None:
                raise ValueError("FixedPointIndicator requires a child expression")
            expr = FixedPointIndicator(expr)
        elif op_name == 'ResetRamp':
            if expr is None:
                raise ValueError("ResetRamp requires a child expression")
            expr = ResetRamp(params['d'], expr)
        elif op_name == 'TokenResetMax':
            if expr is None:
                raise ValueError("TokenResetMax requires a child expression")
            expr = TokenResetMax(params['d'], expr)
        elif op_name == 'SignedWalk':
            if expr is None:
                raise ValueError("SignedWalk requires a child expression")
            expr = SignedWalk(params['up'], params['down'], expr)
        elif op_name == 'SRLatch':
            if expr is None:
                raise ValueError("SRLatch requires a child expression")
            expr = SRLatch(params['set_tok'], params['reset_tok'], expr)
        elif op_name == 'SampleHold':
            if expr is None:
                raise ValueError("SampleHold requires a child expression")
            expr = SampleHold(params['mode'], params['thresh'], expr)
        elif op_name == 'Deplete':
            if expr is None:
                raise ValueError("Deplete requires a child expression")
            expr = Deplete(params['start'], params['floor'], expr)
        elif op_name == 'AnchorSubtract':
            if expr is None:
                raise ValueError("AnchorSubtract requires a child expression")
            expr = AnchorSubtract(params['a'], expr)
        elif op_name == 'AnchorClamp':
            if expr is None:
                raise ValueError("AnchorClamp requires a child expression")
            expr = AnchorClamp(params['mode'], params['a'], expr)
        elif op_name == 'DataRotate':
            if expr is None:
                raise ValueError("DataRotate requires a child expression")
            expr = DataRotate(params['a'], expr)
        elif op_name == 'DataReflect':
            if expr is None:
                raise ValueError("DataReflect requires a child expression")
            expr = DataReflect(params['a'], expr)
        elif op_name == 'StrideGather':
            if expr is None:
                raise ValueError("StrideGather requires a child expression")
            expr = StrideGather(params['a'], expr)
        elif op_name == 'PrefixReverseK':
            if expr is None:
                raise ValueError("PrefixReverseK requires a child expression")
            expr = PrefixReverseK(params['a'], expr)
        elif op_name == 'PrefixSortK':
            if expr is None:
                raise ValueError("PrefixSortK requires a child expression")
            expr = PrefixSortK(params['a'], expr)
        elif op_name == 'CompareSwap':
            if expr is None:
                raise ValueError("CompareSwap requires a child expression")
            expr = CompareSwap(params['i'], params['j'], expr)
        elif op_name == 'WindowMax':
            if expr is None:
                raise ValueError("WindowMax requires a child expression")
            expr = WindowMax(params['w'], expr)
        elif op_name == 'ParityRoute':
            if expr is None:
                raise ValueError("ParityRoute requires a child expression")
            expr = ParityRoute(expr)
        elif op_name == 'RippleCarry':
            if expr is None:
                raise ValueError("RippleCarry requires a child expression")
            expr = RippleCarry(params['k'], expr)
        elif op_name == 'ArgMinMax':
            if expr is None:
                raise ValueError("ArgMinMax requires a child expression")
            expr = ArgMinMax(expr)
        elif op_name == 'RunningCountDigit':
            if expr is None:
                raise ValueError("RunningCountDigit requires a child expression")
            expr = RunningCountDigit(params['d'], expr)
        elif op_name == 'CompareShift':
            if expr is None:
                raise ValueError("CompareShift requires a child expression")
            expr = CompareShift(params['k'], expr)
        elif op_name == 'DiffShift':
            if expr is None:
                raise ValueError("DiffShift requires a child expression")
            expr = DiffShift(params['k'], expr)
        elif op_name == 'HashScan':
            if expr is None:
                raise ValueError("HashScan requires a child expression")
            expr = HashScan(params['a'], params['b'], params['c'], params['seed'], expr)
        elif op_name == 'WindowHash':
            if expr is None:
                raise ValueError("WindowHash requires a child expression")
            expr = WindowHash(params['T'], params['w'], params['a'], params['b'], params['c'], expr)
        elif op_name == 'IndexMix':
            if expr is None:
                raise ValueError("IndexMix requires a child expression")
            expr = IndexMix(params['T'], params['a'], params['b'], expr)
        elif op_name == 'PermuteByRank':
            if expr is None:
                raise ValueError("PermuteByRank requires a child expression")
            expr = PermuteByRank(params['T'], expr)
        elif op_name == 'CycleWalk':
            if expr is None:
                raise ValueError("CycleWalk requires a child expression")
            expr = CycleWalk(params['T'], expr)
        elif op_name == 'TwoPointerMerge':
            if expr is None:
                raise ValueError("TwoPointerMerge requires a child expression")
            expr = TwoPointerMerge(params['T'], expr)
        elif op_name == 'ConditionalJump':
            if expr is None:
                raise ValueError("ConditionalJump requires a child expression")
            expr = ConditionalJump(params['T'], params['threshold'], expr)
        elif op_name == 'TreeWalkDFS':
            if expr is None:
                raise ValueError("TreeWalkDFS requires a child expression")
            expr = TreeWalkDFS(params['T'], expr)
        elif op_name == 'TreeWalkBFS':
            if expr is None:
                raise ValueError("TreeWalkBFS requires a child expression")
            expr = TreeWalkBFS(params['T'], expr)
        elif op_name == 'CellularAutomaton':
            if expr is None:
                raise ValueError("CellularAutomaton requires a child expression")
            expr = CellularAutomaton(params['steps'], params['a'], params['b'], params['c'], expr)
        elif op_name == 'LFSR':
            if expr is None:
                raise ValueError("LFSR requires a child expression")
            expr = LFSR(params['T'], params['width'], params['tap1'], params['tap2'], params['c1'], params['c2'], expr)
        elif op_name == 'ParityInterleave':
            if expr is None:
                raise ValueError("ParityInterleave requires a child expression")
            expr = ParityInterleave(params['a'], params['b'], params['c'], params['d'], expr)
        elif op_name == 'ReductionTree':
            if expr is None:
                raise ValueError("ReductionTree requires a child expression")
            expr = ReductionTree(params['T'], params['a'], params['b'], expr)
        elif op_name == 'ChainedIndex':
            if expr is None:
                raise ValueError("ChainedIndex requires a child expression")
            expr = ChainedIndex(params['T'], params['depth'], expr)
        elif op_name == 'SelfConvolve':
            if expr is None:
                raise ValueError("SelfConvolve requires a child expression")
            expr = SelfConvolve(params['T'], expr)
        elif op_name == 'AccumulateReset':
            if expr is None:
                raise ValueError("AccumulateReset requires a child expression")
            expr = AccumulateReset(params['threshold'], expr)
        elif op_name == 'ScatterWrite':
            if expr is None:
                raise ValueError("ScatterWrite requires a child expression")
            expr = ScatterWrite(params['T'], params['a'], expr)
        elif op_name == 'InverseLookup':
            if expr is None:
                raise ValueError("InverseLookup requires a child expression")
            expr = InverseLookup(params['T'], expr)
        elif op_name == 'RunLengthEncode':
            if expr is None:
                raise ValueError("RunLengthEncode requires a child expression")
            expr = RunLengthEncode(params['T'], expr)
        elif op_name == 'ChunkSort':
            if expr is None:
                raise ValueError("ChunkSort requires a child expression")
            expr = ChunkSort(params['i'], params['j'], params['asc'], expr)
        elif op_name == 'SegmentReverse':
            if expr is None:
                raise ValueError("SegmentReverse requires a child expression")
            expr = SegmentReverse(params['boundary_digit'], expr)
        elif op_name == 'XORScan':
            if expr is None:
                raise ValueError("XORScan requires a child expression")
            expr = XORScan(expr)
        elif op_name == 'ModRemainder':
            if expr is None:
                raise ValueError("ModRemainder requires a child expression")
            expr = ModRemainder(params['m'], expr)
        elif op_name == 'ZigzagCount':
            if expr is None:
                raise ValueError("ZigzagCount requires a child expression")
            expr = ZigzagCount(expr)
        elif op_name == 'RecordCount':
            if expr is None:
                raise ValueError("RecordCount requires a child expression")
            expr = RecordCount(expr)
        elif op_name == 'KMPFailure':
            if expr is None:
                raise ValueError("KMPFailure requires a child expression")
            expr = KMPFailure(expr)
        elif op_name == 'BWT':
            if expr is None:
                raise ValueError("BWT requires a child expression")
            expr = BWT(expr)
        elif op_name == 'LZ78Complexity':
            if expr is None:
                raise ValueError("LZ78Complexity requires a child expression")
            expr = LZ78Complexity(expr)
        elif op_name == 'VanEckExtension':
            if expr is None:
                raise ValueError("VanEckExtension requires a child expression")
            expr = VanEckExtension(params['T'], expr)
        elif op_name == 'SwitchedScan':
            if expr is None:
                raise ValueError("SwitchedScan requires a child expression")
            expr = SwitchedScan(params['a1'], params['b1'], params['a2'], params['b2'], params['thresh'], expr)

        elif op_name == 'Sarank':
            if expr is None:
                raise ValueError("Sarank requires a child expression")
            expr = Sarank(expr)
        elif op_name == 'TaskMux':
            if expr is not None:
                raise ValueError("TaskMux is a leaf token; nothing may appear right of it")
            expr = TaskMux([decode_algorithm_id(mid, input_length=10)
                            for mid in params['member_ids']])
        elif op_name == 'Pmaxv9':
            if expr is None:
                raise ValueError("Pmaxv9 requires a child expression")
            expr = Pmaxv9(expr)
        elif op_name == 'SmallDFATrace':
            if expr is None:
                raise ValueError("SmallDFATrace requires a child expression")
            expr = SmallDFATrace(expr)
        elif op_name == 'Indecho':
            if expr is None:
                raise ValueError("Indecho requires a child expression")
            expr = Indecho(expr)
        elif op_name == 'GroupCosetProductS3':
            if expr is None:
                raise ValueError("GroupCosetProductS3 requires a child expression")
            expr = GroupCosetProductS3(expr)
        elif op_name == 'Kanc2':
            if expr is None:
                raise ValueError("Kanc2 requires a child expression")
            expr = Kanc2(expr)
        elif op_name == 'Kanc4':
            if expr is None:
                raise ValueError("Kanc4 requires a child expression")
            expr = Kanc4(expr)
        elif op_name == 'Kanc8':
            if expr is None:
                raise ValueError("Kanc8 requires a child expression")
            expr = Kanc8(expr)
        elif op_name == 'Kanc10':
            if expr is None:
                raise ValueError("Kanc10 requires a child expression")
            expr = Kanc10(expr)
        elif op_name == 'Sanc2':
            if expr is None:
                raise ValueError("Sanc2 requires a child expression")
            expr = Sanc2(expr)
        elif op_name == 'Tmap5':
            if expr is None:
                raise ValueError("Tmap5 requires a child expression")
            expr = Tmap5(expr)
        elif op_name == 'Addfix':
            if expr is None:
                raise ValueError("Addfix requires a child expression")
            expr = Addfix(expr)
        elif op_name == 'Sjf1':
            if expr is None:
                raise ValueError("Sjf1 requires a child expression")
            expr = Sjf1(expr)
        elif op_name == 'Duelsum':
            if expr is None:
                raise ValueError("Duelsum requires a child expression")
            expr = Duelsum(expr)
        elif op_name == 'Netrm':
            if expr is None:
                raise ValueError("Netrm requires a child expression")
            expr = Netrm(expr)
        elif op_name == 'Convfix':
            if expr is None:
                raise ValueError("Convfix requires a child expression")
            expr = Convfix(expr)
        elif op_name == 'Usum':
            if expr is None:
                raise ValueError("Usum requires a child expression")
            expr = Usum(expr)
        elif op_name == 'Perase0':
            if expr is None:
                raise ValueError("Perase0 requires a child expression")
            expr = Perase0(expr)
        elif op_name == 'Lagf3':
            if expr is None:
                raise ValueError("Lagf3 requires a child expression")
            expr = Lagf3(expr)
        elif op_name == 'Nearidx':
            if expr is None:
                raise ValueError("Nearidx requires a child expression")
            expr = Nearidx(expr)
        elif op_name == 'Tresetsum3':
            if expr is None:
                raise ValueError("Tresetsum3 requires a child expression")
            expr = Tresetsum3(expr)
        elif op_name == 'Oepass0':
            if expr is None:
                raise ValueError("Oepass0 requires a child expression")
            expr = Oepass0(expr)
        elif op_name == 'Pge':
            if expr is None:
                raise ValueError("Pge requires a child expression")
            expr = Pge(expr)
        elif op_name == 'Cgpmax09':
            if expr is None:
                raise ValueError("Cgpmax09 requires a child expression")
            expr = Cgpmax09(expr)
        elif op_name == 'Lanesum':
            if expr is None:
                raise ValueError("Lanesum requires a child expression")
            expr = Lanesum(expr)
        elif op_name == 'Addpmax':
            if expr is None:
                raise ValueError("Addpmax requires a child expression")
            expr = Addpmax(expr)
        elif op_name == 'Maxrun':
            if expr is None:
                raise ValueError("Maxrun requires a child expression")
            expr = Maxrun(expr)
        elif op_name == 'Dualtie':
            if expr is None:
                raise ValueError("Dualtie requires a child expression")
            expr = Dualtie(expr)
        elif op_name == 'Cap59':
            if expr is None:
                raise ValueError("Cap59 requires a child expression")
            expr = Cap59(expr)
        elif op_name == 'Cgrev09':
            if expr is None:
                raise ValueError("Cgrev09 requires a child expression")
            expr = Cgrev09(expr)
        elif op_name == 'Dupmax':
            if expr is None:
                raise ValueError("Dupmax requires a child expression")
            expr = Dupmax(expr)
        elif op_name == 'Rmaj5':
            if expr is None:
                raise ValueError("Rmaj5 requires a child expression")
            expr = Rmaj5(expr)
        elif op_name == 'Majmask':
            if expr is None:
                raise ValueError("Majmask requires a child expression")
            expr = Majmask(expr)
        elif op_name == 'AnchorBroadcast':
            if expr is None:
                raise ValueError("AnchorBroadcast requires a child expression")
            expr = AnchorBroadcast(expr)
        elif op_name == 'Zpeak':
            if expr is None:
                raise ValueError("Zpeak requires a child expression")
            expr = Zpeak(expr)
        elif op_name == 'Mulanc0':
            if expr is None:
                raise ValueError("Mulanc0 requires a child expression")
            expr = Mulanc0(expr)
        elif op_name == 'Recdist':
            if expr is None:
                raise ValueError("Recdist requires a child expression")
            expr = Recdist(expr)
        elif op_name == 'Pmrep':
            if expr is None:
                raise ValueError("Pmrep requires a child expression")
            expr = Pmrep(expr)
        elif op_name == 'Mulpair':
            if expr is None:
                raise ValueError("Mulpair requires a child expression")
            expr = Mulpair(expr)
        elif op_name == 'PrevSegmentMax':
            if expr is None:
                raise ValueError("PrevSegmentMax requires a child expression")
            expr = PrevSegmentMax(expr)
        elif op_name == 'Occcnt':
            if expr is None:
                raise ValueError("Occcnt requires a child expression")
            expr = Occcnt(expr)
        elif op_name == 'SegmentDistinct':
            if expr is None:
                raise ValueError("SegmentDistinct requires a child expression")
            expr = SegmentDistinct(expr)
        elif op_name == 'Dclamp':
            if expr is None:
                raise ValueError("Dclamp requires a child expression")
            expr = Dclamp(expr)
        elif op_name == 'FirstUnique':
            if expr is None:
                raise ValueError("FirstUnique requires a child expression")
            expr = FirstUnique(expr)
        elif op_name == 'Runmed':
            if expr is None:
                raise ValueError("Runmed requires a child expression")
            expr = Runmed(expr)
        elif op_name == 'Cntge3':
            if expr is None:
                raise ValueError("Cntge3 requires a child expression")
            expr = Cntge3(expr)
        elif op_name == 'Twosum10':
            if expr is None:
                raise ValueError("Twosum10 requires a child expression")
            expr = Twosum10(expr)
        elif op_name == 'Holdodd':
            if expr is None:
                raise ValueError("Holdodd requires a child expression")
            expr = Holdodd(expr)
        elif op_name == 'Addlastodd':
            if expr is None:
                raise ValueError("Addlastodd requires a child expression")
            expr = Addlastodd(expr)
        elif op_name == 'Bmcand':
            if expr is None:
                raise ValueError("Bmcand requires a child expression")
            expr = Bmcand(expr)
        elif op_name == 'Hold2ge5':
            if expr is None:
                raise ValueError("Hold2ge5 requires a child expression")
            expr = Hold2ge5(expr)
        elif op_name == 'Sjf2':
            if expr is None:
                raise ValueError("Sjf2 requires a child expression")
            expr = Sjf2(expr)
        elif op_name == 'Gapge5':
            if expr is None:
                raise ValueError("Gapge5 requires a child expression")
            expr = Gapge5(expr)
        elif op_name == 'Riseerase':
            if expr is None:
                raise ValueError("Riseerase requires a child expression")
            expr = Riseerase(expr)
        elif op_name == 'Ebud3':
            if expr is None:
                raise ValueError("Ebud3 requires a child expression")
            expr = Ebud3(expr)
        elif op_name == 'Tieswap':
            if expr is None:
                raise ValueError("Tieswap requires a child expression")
            expr = Tieswap(expr)
        elif op_name == 'Risemax':
            if expr is None:
                raise ValueError("Risemax requires a child expression")
            expr = Risemax(expr)
        elif op_name == 'Delc2':
            if expr is None:
                raise ValueError("Delc2 requires a child expression")
            expr = Delc2(expr)
        elif op_name == 'Riskeep':
            if expr is None:
                raise ValueError("Riskeep requires a child expression")
            expr = Riskeep(expr)
        elif op_name == 'Mixtie':
            if expr is None:
                raise ValueError("Mixtie requires a child expression")
            expr = Mixtie(expr)
        elif op_name == 'Pmaxv89':
            if expr is None:
                raise ValueError("Pmaxv89 requires a child expression")
            expr = Pmaxv89(expr)
        elif op_name == 'Pmaxv789':
            if expr is None:
                raise ValueError("Pmaxv789 requires a child expression")
            expr = Pmaxv789(expr)
        elif op_name == 'Togsum0':
            if expr is None:
                raise ValueError("Togsum0 requires a child expression")
            expr = Togsum0(expr)
        elif op_name == 'Signsum0':
            if expr is None:
                raise ValueError("Signsum0 requires a child expression")
            expr = Signsum0(expr)
        elif op_name == 'Median3':
            if expr is None:
                raise ValueError("Median3 requires a child expression")
            expr = Median3(expr)
        elif op_name == 'Majority3':
            if expr is None:
                raise ValueError("Majority3 requires a child expression")
            expr = Majority3(expr)
        elif op_name == 'Spile4':
            if expr is None:
                raise ValueError("Spile4 requires a child expression")
            expr = Spile4(expr)
        elif op_name == 'Spile5':
            if expr is None:
                raise ValueError("Spile5 requires a child expression")
            expr = Spile5(expr)
        elif op_name == 'Spile6':
            if expr is None:
                raise ValueError("Spile6 requires a child expression")
            expr = Spile6(expr)
        elif op_name == 'Cyca':
            if expr is None:
                raise ValueError("Cyca requires a child expression")
            expr = Cyca(expr)
        elif op_name == 'Rule184Thr4':
            if expr is None:
                raise ValueError("Rule184Thr4 requires a child expression")
            expr = Rule184Thr4(expr)
        elif op_name == 'Rule184Thr5':
            if expr is None:
                raise ValueError("Rule184Thr5 requires a child expression")
            expr = Rule184Thr5(expr)
        elif op_name == 'Rule184Thr6':
            if expr is None:
                raise ValueError("Rule184Thr6 requires a child expression")
            expr = Rule184Thr6(expr)
        elif op_name == 'Ewma2':
            if expr is None:
                raise ValueError("Ewma2 requires a child expression")
            expr = Ewma2(expr)
        elif op_name == 'Zcross5':
            if expr is None:
                raise ValueError("Zcross5 requires a child expression")
            expr = Zcross5(expr)
        elif op_name == 'Heapmax':
            if expr is None:
                raise ValueError("Heapmax requires a child expression")
            expr = Heapmax(expr)
        elif op_name == 'Vig2':
            if expr is None:
                raise ValueError("Vig2 requires a child expression")
            expr = Vig2(expr)
        elif op_name == 'Vig3':
            if expr is None:
                raise ValueError("Vig3 requires a child expression")
            expr = Vig3(expr)
        elif op_name == 'Vig5':
            if expr is None:
                raise ValueError("Vig5 requires a child expression")
            expr = Vig5(expr)
        elif op_name == 'Hamk3':
            if expr is None:
                raise ValueError("Hamk3 requires a child expression")
            expr = Hamk3(expr)
        elif op_name == 'Altrun':
            if expr is None:
                raise ValueError("Altrun requires a child expression")
            expr = Altrun(expr)
        elif op_name == 'BlockSwapParity':
            if expr is None:
                raise ValueError("BlockSwapParity requires a child expression")
            expr = BlockSwapParity(expr)
        elif op_name == 'BrickSort2':
            if expr is None:
                raise ValueError("BrickSort2 requires a child expression")
            expr = BrickSort2(expr)
        elif op_name == 'FeistelAdd':
            if expr is None:
                raise ValueError("FeistelAdd requires a child expression")
            expr = FeistelAdd(expr)
        elif op_name == 'BlockMax2':
            if expr is None:
                raise ValueError("BlockMax2 requires a child expression")
            expr = BlockMax2(expr)
        elif op_name == 'ErrorDiffuse2':
            if expr is None:
                raise ValueError("ErrorDiffuse2 requires a child expression")
            expr = ErrorDiffuse2(expr)
        elif op_name == 'ErrorDiffuse3':
            if expr is None:
                raise ValueError("ErrorDiffuse3 requires a child expression")
            expr = ErrorDiffuse3(expr)
        elif op_name == 'ErrorDiffuse5':
            if expr is None:
                raise ValueError("ErrorDiffuse5 requires a child expression")
            expr = ErrorDiffuse5(expr)
        elif op_name == 'BoxSum2':
            if expr is None:
                raise ValueError("BoxSum2 requires a child expression")
            expr = BoxSum2(expr)
        elif op_name == 'BoxSum3':
            if expr is None:
                raise ValueError("BoxSum3 requires a child expression")
            expr = BoxSum3(expr)
        elif op_name == 'BoxSum4':
            if expr is None:
                raise ValueError("BoxSum4 requires a child expression")
            expr = BoxSum4(expr)
        elif op_name == 'BoxSum5':
            if expr is None:
                raise ValueError("BoxSum5 requires a child expression")
            expr = BoxSum5(expr)
        elif op_name == 'Schmitt':
            if expr is None:
                raise ValueError("Schmitt requires a child expression")
            expr = Schmitt(expr)
        elif op_name == 'RailFence2':
            if expr is None:
                raise ValueError("RailFence2 requires a child expression")
            expr = RailFence2(expr)
        elif op_name == 'RailFence3':
            if expr is None:
                raise ValueError("RailFence3 requires a child expression")
            expr = RailFence3(expr)
        elif op_name == 'RailFence4':
            if expr is None:
                raise ValueError("RailFence4 requires a child expression")
            expr = RailFence4(expr)
        elif op_name == 'QueueWalk3':
            if expr is None:
                raise ValueError("QueueWalk3 requires a child expression")
            expr = QueueWalk3(expr)
        elif op_name == 'QueueWalk4':
            if expr is None:
                raise ValueError("QueueWalk4 requires a child expression")
            expr = QueueWalk4(expr)
        elif op_name == 'QueueWalk5':
            if expr is None:
                raise ValueError("QueueWalk5 requires a child expression")
            expr = QueueWalk5(expr)
        elif op_name == 'QueueWalk6':
            if expr is None:
                raise ValueError("QueueWalk6 requires a child expression")
            expr = QueueWalk6(expr)
        elif op_name == 'MinBottleneck':
            if expr is None:
                raise ValueError("MinBottleneck requires a child expression")
            expr = MinBottleneck(expr)
        elif op_name == 'MaxCapacity':
            if expr is None:
                raise ValueError("MaxCapacity requires a child expression")
            expr = MaxCapacity(expr)
        elif op_name == 'HeapSubtreeMax':
            if expr is None:
                raise ValueError("HeapSubtreeMax requires a child expression")
            expr = HeapSubtreeMax(expr)
        elif op_name == 'IncRunMax':
            if expr is None:
                raise ValueError("IncRunMax requires a child expression")
            expr = IncRunMax(expr)
        elif op_name == 'HaarStep':
            if expr is None:
                raise ValueError("HaarStep requires a child expression")
            expr = HaarStep(expr)
        elif op_name == 'RunMAD':
            if expr is None:
                raise ValueError("RunMAD requires a child expression")
            expr = RunMAD(expr)
        elif op_name == 'ContFrac':
            if expr is None:
                raise ValueError("ContFrac requires a child expression")
            expr = ContFrac(expr)
        elif op_name == 'GroupProductS3':
            if expr is None:
                raise ValueError("GroupProductS3 requires a child expression")
            expr = GroupProductS3(expr)
        elif op_name == 'GroupProductD4':
            if expr is None:
                raise ValueError("GroupProductD4 requires a child expression")
            expr = GroupProductD4(expr)
        elif op_name == 'GroupProductQ8':
            if expr is None:
                raise ValueError("GroupProductQ8 requires a child expression")
            expr = GroupProductQ8(expr)
        elif op_name == 'GroupProductD5':
            if expr is None:
                raise ValueError("GroupProductD5 requires a child expression")
            expr = GroupProductD5(expr)
        elif op_name == 'D4Character':
            if expr is None:
                raise ValueError("D4Character requires a child expression")
            expr = D4Character(expr)
        elif op_name == 'GroupActionD5':
            if expr is None:
                raise ValueError("GroupActionD5 requires a child expression")
            expr = GroupActionD5(expr)
        elif op_name == 'RunningRangeAdd':
            if expr is None:
                raise ValueError("RunningRangeAdd requires a child expression")
            expr = RunningRangeAdd(expr)
        elif op_name == 'RunningMedianAdd':
            if expr is None:
                raise ValueError("RunningMedianAdd requires a child expression")
            expr = RunningMedianAdd(expr)
        elif op_name == 'ThirdOrderScan':
            if expr is None:
                raise ValueError("ThirdOrderScan requires a child expression")
            expr = ThirdOrderScan(expr)
        elif op_name == 'LinearRotor3':
            if expr is None:
                raise ValueError("LinearRotor3 requires a child expression")
            expr = LinearRotor3(expr)
        elif op_name == 'RunningModeCountAdd':
            if expr is None:
                raise ValueError("RunningModeCountAdd requires a child expression")
            expr = RunningModeCountAdd(expr)
        elif op_name == 'Oscpos':
            if expr is None:
                raise ValueError("Oscpos requires a child expression")
            expr = Oscpos(expr)
        elif op_name == 'Oscmom':
            if expr is None:
                raise ValueError("Oscmom requires a child expression")
            expr = Oscmom(expr)
        elif op_name == 'RefractoryVeto':
            if expr is None:
                raise ValueError("RefractoryVeto requires a child expression")
            expr = RefractoryVeto(expr)
        elif op_name == 'MexicanHatVeto':
            if expr is None:
                raise ValueError("MexicanHatVeto requires a child expression")
            expr = MexicanHatVeto(expr)
        elif op_name == 'MaxFrequency':
            if expr is None:
                raise ValueError("MaxFrequency requires a child expression")
            expr = MaxFrequency(expr)
        elif op_name == 'DucciStep':
            if expr is None:
                raise ValueError("DucciStep requires a child expression")
            expr = DucciStep(expr)
        elif op_name == 'PairGcd':
            if expr is None:
                raise ValueError("PairGcd requires a child expression")
            expr = PairGcd(expr)
        elif op_name == 'SecondOrderScan':
            if expr is None:
                raise ValueError("SecondOrderScan requires a child expression")
            expr = SecondOrderScan(expr)
        elif op_name == 'NivenByCount':
            if expr is None:
                raise ValueError("NivenByCount requires a child expression")
            expr = NivenByCount(expr)
        elif op_name == 'EuclidStep':
            if expr is None:
                raise ValueError("EuclidStep requires a child expression")
            expr = EuclidStep(expr)
        elif op_name == 'PairCarry':
            if expr is None:
                raise ValueError("PairCarry requires a child expression")
            expr = PairCarry(expr)
        elif op_name == 'InversionCount':
            if expr is None:
                raise ValueError("InversionCount requires a child expression")
            expr = InversionCount(expr)
        elif op_name == 'MajorIndex':
            if expr is None:
                raise ValueError("MajorIndex requires a child expression")
            expr = MajorIndex(expr)
        elif op_name == 'NimHistogram':
            if expr is None:
                raise ValueError("NimHistogram requires a child expression")
            expr = NimHistogram(expr)
        elif op_name == 'MeanRelax':
            if expr is None:
                raise ValueError("MeanRelax requires a child expression")
            expr = MeanRelax(expr)
        elif op_name == 'AnchorNudge':
            if expr is None:
                raise ValueError("AnchorNudge requires a child expression")
            expr = AnchorNudge(expr)
        elif op_name == 'PhaseNudge':
            if expr is None:
                raise ValueError("PhaseNudge requires a child expression")
            expr = PhaseNudge(expr)
        elif op_name == 'HeatExchange':
            if expr is None:
                raise ValueError("HeatExchange requires a child expression")
            expr = HeatExchange(expr)
        elif op_name == 'ElemSym2':
            if expr is None:
                raise ValueError("ElemSym2 requires a child expression")
            expr = ElemSym2(expr)
        elif op_name == 'RunningMex':
            if expr is None:
                raise ValueError("RunningMex requires a child expression")
            expr = RunningMex(expr)
        elif op_name == 'MirrorAdd':
            if expr is None:
                raise ValueError("MirrorAdd requires a child expression")
            expr = MirrorAdd(expr)
        elif op_name == 'MirrorMax':
            if expr is None:
                raise ValueError("MirrorMax requires a child expression")
            expr = MirrorMax(expr)
        elif op_name == 'MirrorMin':
            if expr is None:
                raise ValueError("MirrorMin requires a child expression")
            expr = MirrorMin(expr)
        elif op_name == 'OnlineRankAdd':
            if expr is None:
                raise ValueError("OnlineRankAdd requires a child expression")
            expr = OnlineRankAdd(expr)
        elif op_name == 'SignDiffuse':
            if expr is None:
                raise ValueError("SignDiffuse requires a child expression")
            expr = SignDiffuse(expr)
        elif op_name == 'DoubleGE':
            if expr is None:
                raise ValueError("DoubleGE requires a child expression")
            expr = DoubleGE(expr)
        elif op_name == 'MidrangeAdd':
            if expr is None:
                raise ValueError("MidrangeAdd requires a child expression")
            expr = MidrangeAdd(expr)
        elif op_name == 'SpacedSeedSum':
            if expr is None:
                raise ValueError("SpacedSeedSum requires a child expression")
            expr = SpacedSeedSum(expr)
        elif op_name == 'ArgPosAdd':
            if expr is None:
                raise ValueError("ArgPosAdd requires a child expression")
            expr = ArgPosAdd(expr)
        elif op_name == 'LaneRunningMax':
            if expr is None:
                raise ValueError("LaneRunningMax requires a child expression")
            expr = LaneRunningMax(expr)
        elif op_name == 'PancakeFlip':
            if expr is None:
                raise ValueError("PancakeFlip requires a child expression")
            expr = PancakeFlip(expr)
        elif op_name == 'BitPlaneMajority':
            if expr is None:
                raise ValueError("BitPlaneMajority requires a child expression")
            expr = BitPlaneMajority(expr)
        elif op_name == 'PredicatedSkip':
            if expr is None:
                raise ValueError("PredicatedSkip requires a child expression")
            expr = PredicatedSkip(expr)
        elif op_name == 'BitTransitionCount':
            if expr is None:
                raise ValueError("BitTransitionCount requires a child expression")
            expr = BitTransitionCount(expr)
        elif op_name == 'OpcodeMachine':
            if expr is None:
                raise ValueError("OpcodeMachine requires a child expression")
            expr = OpcodeMachine(expr)
        elif op_name == 'KnockoutDepth':
            if expr is None:
                raise ValueError("KnockoutDepth requires a child expression")
            expr = KnockoutDepth(expr)
        elif op_name == 'EnvelopeMax':
            if expr is None:
                raise ValueError("EnvelopeMax requires a child expression")
            expr = EnvelopeMax(expr)
        elif op_name == 'CountMinSketch':
            if expr is None:
                raise ValueError("CountMinSketch requires a child expression")
            expr = CountMinSketch(expr)
        elif op_name == 'BitTransposeGather':
            if expr is None:
                raise ValueError("BitTransposeGather requires a child expression")
            expr = BitTransposeGather(expr)
        elif op_name == 'InternRank':
            if expr is None:
                raise ValueError("InternRank requires a child expression")
            expr = InternRank(expr)
        elif op_name == 'MultiplicityParityXor':
            if expr is None:
                raise ValueError("MultiplicityParityXor requires a child expression")
            expr = MultiplicityParityXor(expr)
        elif op_name == 'ProximityDegree':
            if expr is None:
                raise ValueError("ProximityDegree requires a child expression")
            expr = ProximityDegree(expr)
        elif op_name == 'CumFreqRank':
            if expr is None:
                raise ValueError("CumFreqRank requires a child expression")
            expr = CumFreqRank(expr)
        elif op_name == 'DistinctSubseqCount':
            if expr is None:
                raise ValueError("DistinctSubseqCount requires a child expression")
            expr = DistinctSubseqCount(expr)
        elif op_name == 'CodonSkip':
            if expr is None:
                raise ValueError("CodonSkip requires a child expression")
            expr = CodonSkip(expr)
        elif op_name == 'FragLen':
            if expr is None:
                raise ValueError("FragLen requires a child expression")
            expr = FragLen(expr)
        elif op_name == 'DivisorZeta':
            if expr is None:
                raise ValueError("DivisorZeta requires a child expression")
            expr = DivisorZeta(expr)
        elif op_name == 'MobiusInvert':
            if expr is None:
                raise ValueError("MobiusInvert requires a child expression")
            expr = MobiusInvert(expr)
        elif op_name == 'FactoradicNormalize':
            if expr is None:
                raise ValueError("FactoradicNormalize requires a child expression")
            expr = FactoradicNormalize(expr)
        elif op_name == 'NonAdjacentMax4':
            if expr is None:
                raise ValueError("NonAdjacentMax4 requires a child expression")
            expr = NonAdjacentMax4(expr)
        elif op_name == 'BoundedConfidenceMean':
            if expr is None:
                raise ValueError("BoundedConfidenceMean requires a child expression")
            expr = BoundedConfidenceMean(expr)
        elif op_name == 'ExtremalReplace':
            if expr is None:
                raise ValueError("ExtremalReplace requires a child expression")
            expr = ExtremalReplace(expr)
        elif op_name == 'Hsifte':
            if expr is None:
                raise ValueError("Hsifte requires a child expression")
            expr = Hsifte(expr)
        elif op_name == 'LinearProbe':
            if expr is None:
                raise ValueError("LinearProbe requires a child expression")
            expr = LinearProbe(expr)
        elif op_name == 'FlagPartition':
            if expr is None:
                raise ValueError("FlagPartition requires a child expression")
            expr = FlagPartition(expr)
        elif op_name == 'Slew2':
            if expr is None:
                raise ValueError("Slew2 requires a child expression")
            expr = Slew2(expr)
        elif op_name == 'Rms3':
            if expr is None:
                raise ValueError("Rms3 requires a child expression")
            expr = Rms3(expr)
        elif op_name == 'Spn2':
            if expr is None:
                raise ValueError("Spn2 requires a child expression")
            expr = Spn2(expr)
        elif op_name == 'Xpar':
            if expr is None:
                raise ValueError("Xpar requires a child expression")
            expr = Xpar(expr)
    if expr is None:
        raise ValueError(f"Failed to decode algorithm ID: {algo_id}")

    return expr


def register_dsl_algorithm(name: Optional[str], expr: DSLExpr, use_generated_id: bool = False) -> str:
    """
    Register DSL expression as seq-to-seq algorithm in ALGORITHM_REGISTRY.

    NOTE: DSL algorithms are ALWAYS sequence-to-sequence (never reduced to scalar).

    Args:
        name: Algorithm name (or use generated ID if use_generated_id=True)
        expr: DSL expression
        use_generated_id: If True, use generate_algorithm_id() instead of name

    Returns:
        The algorithm name used for registration
    """
    # Generate ID if requested
    if use_generated_id:
        name = generate_algorithm_id(expr)

    if name is None:
        raise ValueError("Must provide name or set use_generated_id=True")

    # DSL algorithms always return sequences
    def algorithm_fn(perm: torch.Tensor) -> torch.Tensor:
        return expr.eval(perm)

    # Import utils module (use absolute import to avoid relative import issues)
    import utils

    # Register the algorithm
    utils.ALGORITHM_REGISTRY[name] = algorithm_fn

    # Store metadata in separate metadata dictionary
    utils.ALGORITHM_METADATA[name] = {
        'expr': expr,
        'output_length': expr.output_length(),
        'id': generate_algorithm_id(expr),
        'str': str(expr),
        'is_dsl': True,  # Mark as DSL algorithm
    }

    return name


# Operator families for DSL generation
BASIC_OPS = ['Reverse', 'Rotate', 'Take', 'Drop', 'MapAdd', 'MapMul', 'DoubleOdd', 'PadTo', 'ParityRoute']
COMPLEX_OPS = ['JumpIndex', 'PrefixMax', 'PrefixSumMod', 'PrefixSumOdd',
               'RunningProduct', 'CountOdd', 'RecordIndicator', 'NoveltyIndicator', 'FixedPointIndicator',
               'ResetRamp', 'TokenResetMax', 'SignedWalk', 'SRLatch', 'SampleHold', 'Deplete',
               'AnchorSubtract', 'AnchorClamp', 'DataRotate', 'DataReflect', 'StrideGather',
               'PrefixReverseK', 'PrefixSortK', 'CompareSwap',
               'WindowMax', 'RippleCarry', 'ArgMinMax',
               'RunningCountDigit', 'CompareShift', 'DiffShift', 'HashScan',
               'WindowHash', 'IndexMix', 'PermuteByRank', 'CycleWalk', 'TwoPointerMerge', 'ConditionalJump',
               'TreeWalkDFS', 'TreeWalkBFS',
               'CellularAutomaton', 'LFSR', 'ParityInterleave', 'ReductionTree', 'ChainedIndex',
               'SelfConvolve', 'AccumulateReset', 'ScatterWrite', 'InverseLookup',
               'RunLengthEncode', 'ChunkSort', 'SegmentReverse',
               'XORScan', 'ModRemainder', 'ZigzagCount', 'RecordCount', 'KMPFailure',
               'BWT', 'LZ78Complexity', 'VanEckExtension', 'SwitchedScan']
ALL_OPS = BASIC_OPS + COMPLEX_OPS


class DSLProgramGenerator:
    """Generate random DSL programs with depth constraints.

    When op_family='all', enforces that at least half the operators are complex,
    with odd counts favoring complex ops (e.g., depth 4 = 3 ops -> 2 complex, 1 basic).
    """

    def __init__(self, input_length: int, depth: int, seed: Optional[int] = None,
                 op_family: str = 'all'):
        """
        Args:
            input_length: Input sequence length
            depth: Exact depth of DSL tree (1 = single op, 2 = two nested ops, etc.)
            seed: Random seed for reproducibility
            op_family: Which operators to use - 'basic', 'complex', or 'all' (default)
                       When 'all', at least ceil((depth-1)/2) ops will be complex.
        """
        self.input_length = input_length
        self.depth = depth
        self.rng = random.Random(seed)
        self.op_family = op_family

        if op_family == 'basic':
            self.ops = BASIC_OPS
        elif op_family == 'complex':
            self.ops = COMPLEX_OPS
        elif op_family == 'all':
            self.ops = ALL_OPS
        else:
            raise ValueError(f"Unknown op_family: {op_family}. Use 'basic', 'complex', or 'all'")

    def generate(self) -> DSLExpr:
        """
        Generate single random DSL program with exact depth.

        Depth examples:
            depth=1: Id(), Const(5, 3), etc. (leaf nodes)
            depth=2: Reverse(Id()), Take(5, Id()), etc. (one op + leaf)
            depth=3: Reverse(Take(5, Id())), MapAdd(2, Drop(3, Id())), etc.

        When op_family='all', the operator distribution is:
            depth=2 (1 op):  1 complex, 0 basic
            depth=3 (2 ops): 1 complex, 1 basic
            depth=4 (3 ops): 2 complex, 1 basic
            depth=5 (4 ops): 2 complex, 2 basic
            etc.
        """
        if self.op_family == 'all' and self.depth > 1:
            # Enforce complex/basic ratio
            num_ops = self.depth - 1
            min_complex = (num_ops + 1) // 2  # ceil(num_ops / 2)
            max_basic = num_ops - min_complex
            return self._generate_with_quota(self.depth, min_complex, max_basic)
        else:
            return self._generate_with_depth(self.depth)

    def _generate_with_quota(self, depth: int, complex_remaining: int, basic_remaining: int) -> DSLExpr:
        """Generate with quota enforcement for complex/basic ops."""
        if depth == 1:
            return self._generate_leaf()

        # Decide which pool to draw from based on remaining quotas
        ops_remaining = depth - 1  # How many more ops we need to place

        # Must use complex if we've used up all basic slots
        # Must use basic if we've used up all complex slots (but complex_remaining should always be >= basic usage needed)
        if basic_remaining <= 0:
            # Must use complex
            op = self.rng.choice(COMPLEX_OPS)
            child = self._generate_with_quota(depth - 1, complex_remaining - 1, basic_remaining)
        elif complex_remaining <= 0:
            # Must use basic
            op = self.rng.choice(BASIC_OPS)
            child = self._generate_with_quota(depth - 1, complex_remaining, basic_remaining - 1)
        else:
            # Can use either, but check if we must use complex to meet quota
            # If remaining ops == complex_remaining, all remaining must be complex
            if ops_remaining == complex_remaining:
                op = self.rng.choice(COMPLEX_OPS)
                child = self._generate_with_quota(depth - 1, complex_remaining - 1, basic_remaining)
            else:
                # Free choice
                op = self.rng.choice(ALL_OPS)
                if op in COMPLEX_OPS:
                    child = self._generate_with_quota(depth - 1, complex_remaining - 1, basic_remaining)
                else:
                    child = self._generate_with_quota(depth - 1, complex_remaining, basic_remaining - 1)

        return self._build_op(op, child)

    def _generate_with_depth(self, depth: int) -> DSLExpr:
        """Recursive generation ensuring exact depth (no quota enforcement)."""
        if depth == 1:
            return self._generate_leaf()
        else:
            op = self.rng.choice(self.ops)
            child = self._generate_with_depth(depth - 1)
            return self._build_op(op, child)

    def _build_op(self, op: str, child: DSLExpr) -> DSLExpr:
        """Build an operator node with randomized parameters."""
        if op == 'Reverse':
            return Reverse(child)
        elif op == 'Rotate':
            k = self.rng.randint(-10, 10)
            return Rotate(k, child)
        elif op == 'Take':
            k = self.rng.randint(1, 10)
            return Take(k, child)
        elif op == 'Drop':
            k = self.rng.randint(1, 3)  # Limit to avoid dropping too much
            return Drop(k, child)
        elif op == 'MapAdd':
            c = self.rng.randint(1, 9)
            return MapAdd(c, child)
        elif op == 'MapMul':
            c = self.rng.randint(2, 9)
            return MapMul(c, child)
        elif op == 'DoubleOdd':
            return DoubleOdd(child)
        elif op == 'PadTo':
            # Limit padding to at most 3 elements beyond input length
            M = self.rng.randint(self.input_length, self.input_length + 3)
            fill = self.rng.randint(0, 9)
            return PadTo(M, fill, child)
        elif op == 'JumpIndex':
            T = self.rng.randint(5, 10)  # Minimum 5 for better diversity
            return JumpIndex(T, child)
        elif op == 'PrefixMax':
            return PrefixMax(child)
        elif op == 'PrefixSumMod':
            return PrefixSumMod(child)
        elif op == 'PrefixSumOdd':
            return PrefixSumOdd(child)
        elif op == 'RunningProduct':
            return RunningProduct(child)
        elif op == 'CountOdd':
            return CountOdd(child)
        elif op == 'RecordIndicator':
            return RecordIndicator(child)
        elif op == 'NoveltyIndicator':
            return NoveltyIndicator(child)
        elif op == 'FixedPointIndicator':
            return FixedPointIndicator(child)
        elif op == 'ResetRamp':
            return ResetRamp(self.rng.randint(0, 9), child)
        elif op == 'TokenResetMax':
            return TokenResetMax(self.rng.randint(0, 9), child)
        elif op == 'SignedWalk':
            up = self.rng.randint(0, 9)
            down = (up + 1 + self.rng.randint(0, 8)) % 10
            return SignedWalk(up, down, child)
        elif op == 'SRLatch':
            s = self.rng.randint(0, 9)
            r = (s + 1 + self.rng.randint(0, 8)) % 10
            return SRLatch(s, r, child)
        elif op == 'SampleHold':
            mode = 'ge' if self.rng.randint(0, 1) == 0 else 'le'
            return SampleHold(mode, self.rng.randint(1, 8), child)
        elif op == 'Deplete':
            return Deplete(self.rng.randint(1, 9), 0, child)
        elif op == 'AnchorSubtract':
            return AnchorSubtract(self.rng.randint(0, 9), child)
        elif op == 'AnchorClamp':
            mode = 'floor' if self.rng.randint(0, 1) == 0 else 'ceil'
            return AnchorClamp(mode, self.rng.randint(0, 9), child)
        elif op == 'DataRotate':
            return DataRotate(self.rng.randint(0, 9), child)
        elif op == 'DataReflect':
            return DataReflect(self.rng.randint(0, 9), child)
        elif op == 'StrideGather':
            return StrideGather(self.rng.randint(0, 9), child)
        elif op == 'PrefixReverseK':
            return PrefixReverseK(self.rng.randint(0, 9), child)
        elif op == 'PrefixSortK':
            return PrefixSortK(self.rng.randint(0, 9), child)
        elif op == 'CompareSwap':
            i = self.rng.randint(0, 9)
            j = (i + 1 + self.rng.randint(0, 8)) % 10
            return CompareSwap(i, j, child)
        elif op == 'WindowMax':
            return WindowMax(self.rng.randint(1, 3), child)
        elif op == 'ParityRoute':
            return ParityRoute(child)
        elif op == 'RippleCarry':
            return RippleCarry(self.rng.randint(1, 9), child)
        elif op == 'ArgMinMax':
            return ArgMinMax(child)
        elif op == 'RunningCountDigit':
            d = self.rng.randint(0, 9)
            return RunningCountDigit(d, child)
        elif op == 'CompareShift':
            k = self.rng.randint(1, 3)  # Limit to avoid degenerate all-zeros outputs
            return CompareShift(k, child)
        elif op == 'DiffShift':
            k = self.rng.randint(1, 3)  # Keep consistent with CompareShift
            return DiffShift(k, child)
        elif op == 'HashScan':
            a = self.rng.choice([1, 3, 7, 9])
            b = self.rng.randint(1, 9)
            c = self.rng.randint(0, 9)
            seed = self.rng.randint(0, 9)
            return HashScan(a, b, c, seed, child)
        elif op == 'WindowHash':
            T = self.rng.randint(6, 10)
            w = self.rng.randint(2, 4)  # Window size 2-4
            a = self.rng.choice([2, 3])  # Base for polynomial
            b = self.rng.randint(0, 3)  # Position multiplier
            c = self.rng.randint(0, 9)
            return WindowHash(T, w, a, b, c, child)
        elif op == 'IndexMix':
            T = self.rng.randint(6, 10)
            a = self.rng.choice([3, 5, 7])  # State multiplier
            b = self.rng.randint(1, 5)  # Value multiplier
            return IndexMix(T, a, b, child)
        elif op == 'PermuteByRank':
            T = self.rng.randint(6, 10)
            return PermuteByRank(T, child)
        elif op == 'CycleWalk':
            T = self.rng.randint(6, 10)
            return CycleWalk(T, child)
        elif op == 'TwoPointerMerge':
            T = self.rng.randint(6, 10)
            return TwoPointerMerge(T, child)
        elif op == 'ConditionalJump':
            T = self.rng.randint(6, 10)
            threshold = self.rng.randint(3, 6)  # Threshold 3-6 for balanced branching
            return ConditionalJump(T, threshold, child)
        elif op == 'TreeWalkDFS':
            T = self.rng.randint(6, 10)
            return TreeWalkDFS(T, child)
        elif op == 'TreeWalkBFS':
            T = self.rng.randint(6, 10)
            return TreeWalkBFS(T, child)
        elif op == 'CellularAutomaton':
            steps = self.rng.randint(1, 2)  # 1-2 steps to keep learnable
            a = self.rng.choice([1, 3, 7])
            b = self.rng.choice([1, 2, 3])
            c = self.rng.choice([1, 2, 7])
            return CellularAutomaton(steps, a, b, c, child)
        elif op == 'LFSR':
            T = self.rng.randint(6, 10)
            width = self.rng.randint(3, 5)
            tap1 = self.rng.randint(0, width - 1)
            tap2 = self.rng.randint(0, width - 1)
            while tap2 == tap1:
                tap2 = self.rng.randint(0, width - 1)
            c1 = self.rng.choice([1, 3, 7])
            c2 = self.rng.choice([1, 3, 7])
            return LFSR(T, width, tap1, tap2, c1, c2, child)
        elif op == 'ParityInterleave':
            a = self.rng.choice([1, 3, 7, 9])
            b = self.rng.randint(0, 9)
            c = self.rng.choice([1, 3, 7, 9])
            d = self.rng.randint(0, 9)
            return ParityInterleave(a, b, c, d, child)
        elif op == 'ReductionTree':
            T = self.rng.randint(6, 10)
            a = self.rng.choice([1, 3, 7])
            b = self.rng.choice([1, 2, 3])
            return ReductionTree(T, a, b, child)
        elif op == 'ChainedIndex':
            T = self.rng.randint(6, 10)
            depth = self.rng.randint(1, 3)
            return ChainedIndex(T, depth, child)
        elif op == 'SelfConvolve':
            T = self.rng.randint(6, 10)
            return SelfConvolve(T, child)
        elif op == 'AccumulateReset':
            threshold = self.rng.choice([11, 13, 15, 17, 19])
            return AccumulateReset(threshold, child)
        elif op == 'ScatterWrite':
            T = self.rng.randint(6, 10)
            a = self.rng.randint(1, 7)
            return ScatterWrite(T, a, child)
        elif op == 'InverseLookup':
            T = self.rng.randint(6, 10)
            return InverseLookup(T, child)
        elif op == 'RunLengthEncode':
            T = self.rng.randint(6, 10)
            return RunLengthEncode(T, child)
        elif op == 'ChunkSort':
            child_len = child.output_length()
            max_chunk = min(6, child_len)
            if max_chunk < 2:
                # Fallback: sort first 2 elements
                return ChunkSort(0, 2, self.rng.choice([True, False]), child)
            chunk_size = self.rng.randint(2, max_chunk)
            i = self.rng.randint(0, max(0, child_len - chunk_size))
            j = i + chunk_size
            asc = self.rng.choice([True, False])
            return ChunkSort(i, j, asc, child)
        elif op == 'SegmentReverse':
            boundary_digit = self.rng.randint(0, 9)
            return SegmentReverse(boundary_digit, child)
        elif op == 'XORScan':
            return XORScan(child)
        elif op == 'ModRemainder':
            m = self.rng.choice([7, 11, 13, 17])
            return ModRemainder(m, child)
        elif op == 'ZigzagCount':
            return ZigzagCount(child)
        elif op == 'RecordCount':
            return RecordCount(child)
        elif op == 'KMPFailure':
            return KMPFailure(child)
        elif op == 'BWT':
            return BWT(child)
        elif op == 'LZ78Complexity':
            return LZ78Complexity(child)
        elif op == 'VanEckExtension':
            T = self.rng.randint(6, 10)
            return VanEckExtension(T, child)
        elif op == 'SwitchedScan':
            a1 = self.rng.choice([1, 3, 7, 9])
            b1 = self.rng.randint(1, 9)
            a2 = self.rng.choice([1, 3, 7, 9])
            b2 = self.rng.randint(1, 9)
            thresh = self.rng.randint(3, 7)
            return SwitchedScan(a1, b1, a2, b2, thresh, child)
        else:
            raise ValueError(f"Unknown operator: {op}")

    def _generate_leaf(self) -> DSLExpr:
        """Generate leaf node (always Id)."""
        return Id(self.input_length)

    def generate_batch(self, n: int) -> list[DSLExpr]:
        """Generate n distinct programs (deduplicated by algorithm ID)."""
        seen_ids = set()
        programs = []

        attempts = 0
        max_attempts = n * 100  # Prevent infinite loop

        while len(programs) < n and attempts < max_attempts:
            prog = self.generate()
            prog_id = generate_algorithm_id(prog)

            if prog_id not in seen_ids:
                seen_ids.add(prog_id)
                programs.append(prog)

            attempts += 1

        return programs


# =========================================================================== #
# pipeline shim — batch task functions over the DSL                 #
# --------------------------------------------------------------------------- #
# This module IS the pipeline's `algorithms` engine (it replaces the old vetted-
# primitive registry + compose()). A model is named by its DSL ``algorithm_id``
# and its task function is recovered purely from the name via
# ``decode_algorithm_id`` — no registry, no stored composition list.
#
# The pipeline task_fn contract is  inputs[B, input_len] (long) -> outputs
# [B, output_len] (long).  ``DSLExpr.eval`` operates on a SINGLE 1-D sequence
# (it uses Python ``.item()`` loops), so we evaluate row-by-row and stack.
# =========================================================================== #

DEFAULT_INPUT_LENGTH = 10


def expr_for(algorithm_id: str, input_length: int = DEFAULT_INPUT_LENGTH) -> DSLExpr:
    """The DSLExpr AST for a model name (its algorithm_id)."""
    return decode_algorithm_id(algorithm_id, input_length=input_length)


def output_length_for(algorithm_id: str,
                      input_length: int = DEFAULT_INPUT_LENGTH) -> int:
    """Static output length for a model name (no evaluation)."""
    return decode_algorithm_id(algorithm_id, input_length=input_length).output_length()


def batch_eval(expr: DSLExpr, inputs: torch.Tensor) -> torch.Tensor:
    """Evaluate a DSLExpr over a ``[B, L_in]`` long tensor -> ``[B, L_out]``.

    ``DSLExpr.eval`` consumes a single 1-D sequence, so we evaluate row-by-row
    and stack. All rows share ``L_out = expr.output_length()`` so the stack is
    rectangular. Output is long, on the input's device. A 1-D input is evaluated
    directly (returns 1-D)."""
    if inputs.dim() == 1:
        return expr.eval(inputs.long())
    rows = [expr.eval(inputs[b].long()) for b in range(inputs.shape[0])]
    return torch.stack(rows, dim=0).to(inputs.device)


def task_fn_for(algorithm_id: str,
                input_length: int = DEFAULT_INPUT_LENGTH) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the pipeline task function for a model name:
    inputs[B, input_length] (long) -> outputs[B, output_length] (long), via the
    DSL. The returned callable carries ``.algorithm_id`` and ``.output_length``."""
    expr = decode_algorithm_id(algorithm_id, input_length=input_length)

    def _task_fn(inputs: torch.Tensor) -> torch.Tensor:
        return batch_eval(expr, inputs)

    _task_fn.algorithm_id = algorithm_id
    _task_fn.output_length = expr.output_length()
    return _task_fn
