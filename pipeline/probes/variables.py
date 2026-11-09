#!/usr/bin/env python3
"""ProbeVariable registry: internal DSL variables to probe on the residual stream.

Mirrors the DomainCircuit mixin pattern (models/circuits.py): a variable subclasses
ProbeVariable, sets ``variable_id``/``algorithm``/``n_classes``, and implements

    labels(inputs [B,10]) -> LongTensor [B, P_valid]   # pure-Python DSL replay, never the model
    valid_positions()     -> LongTensor [P_valid]      # INPUT-position indices where defined

The teacher-forcing confound (docs/reference/probe_variable_catalog.md) is why valid positions
are INPUT positions only: at output position 9+k the gold y_{k-1} is already in context.
A declared position must also be prefix-determined and non-constant, or the probe scores
free accuracy there; ``validate`` enforces both.

``check_row`` carries two obligations. INDEPENDENCE: at least one raise-condition constrains
``lab`` and ``self.expr.eval`` jointly — express the reference output as a function of the
label, invert a reference recurrence, or read the latent out of the reference by input
surgery; a clause that recomputes the label with the expression that generated it only
re-tests labels(), never the claim. COMPLETENESS: every single-label corruption must be
rejected; where the reference output provably cannot see one (a many-to-one outer op, a veto
set, a floor-halved midrange), ``_pins_label`` closes that gap definitionally.

``bridge_variable(composite_id, split_k)`` is the compositionality-gate tool: the
intermediate value of a composed algorithm's inner sub-expression, on input sites that are
prefix-determined, non-constant, and LOAD-BEARING (the composite output must depend on the
slot, else no graft can catch a wrong value). Its check grafts the CLAIMED labels back into
the composite, so a wrong intermediate — or the right intermediate at the wrong depth —
fails; n_classes is the measured cardinality.
"""
import math, os, sys
from collections import Counter

_PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PIPELINE_ROOT, "models"))

import torch

import algorithms

INPUT_LEN = 10


class ProbeError(Exception):
    pass


VARIABLES: list[type] = []


class ProbeVariable:
    """A DSL-computable internal variable of one model, labelable without the model."""

    variable_id: str = ""
    algorithm: str = ""      # the model (DSL algorithm_id) the variable lives in
    n_classes: int = 0

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.variable_id and not cls.algorithm:
            return   # abstract base
        if not cls.variable_id or not cls.algorithm or not cls.n_classes:
            raise ProbeError(f"{cls.__name__} must set variable_id, algorithm, n_classes")
        if any(v.variable_id == cls.variable_id and v.algorithm == cls.algorithm for v in VARIABLES):
            raise ProbeError(f"duplicate variable {cls.algorithm}/{cls.variable_id}")
        VARIABLES.append(cls)

    def __init__(self) -> None:
        self.expr = algorithms.decode_algorithm_id(
            algorithms.base_algorithm_id(self.algorithm), input_length=INPUT_LEN)

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        """[B, 10] digit inputs -> [B, P_valid] class labels (aligned with valid_positions)."""
        raise NotImplementedError

    def valid_positions(self) -> torch.Tensor:
        raise NotImplementedError

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        """Raise ProbeError unless ``lab`` is jointly consistent with self.expr.eval(x) — and
        unless every single-label corruption is rejected (``_pins_label`` closes what the
        reference output cannot see)."""
        raise NotImplementedError

    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        """DSL-replayed TRUE output had the variable at position p been new_class
        (None = counterfactual undefined for this variable; causal gate skips it)."""
        return None


def for_model(name: str) -> list[type]:
    base = algorithms.base_algorithm_id(name)
    return [v for v in VARIABLES if v.algorithm == base]


def get(name: str, variable_id: str) -> type:
    base = algorithms.base_algorithm_id(name)
    matches = [v for v in VARIABLES if v.algorithm == base and v.variable_id == variable_id]
    if not matches:
        raise KeyError(f"no variable {variable_id!r} for model {name!r}; "
                       f"know {[v.variable_id for v in for_model(name)]}")
    return matches[0]


def models_txt() -> list[str]:
    path = os.path.join(_PIPELINE_ROOT, "models", "models.txt")
    return [ln.split("#", 1)[0].strip() for ln in open(path) if ln.split("#", 1)[0].strip()]


def _sample(trials: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    return torch.randint(0, 10, (trials, INPUT_LEN), generator=g)


def validate(cls: type, trials: int = 2000, seed: int = 0) -> int:
    """Draw ``trials`` uniform inputs and raise unless the DECLARATION survives four tests
    that can fail: labels shaped [trials, P_valid] and inside [0, n_classes); no declared
    position where the label is constant (a free column that inflates probe accuracy); every
    declared position determined by its own prefix; and every row's ``check_row`` against
    models/algorithms.py eval(). Returns the number of rows checked."""
    inst = cls()
    xs = _sample(trials, seed)
    labs = inst.labels(xs)
    pos = inst.valid_positions()
    if labs.shape != (trials, pos.numel()):
        raise ProbeError(f"{cls.variable_id}: labels shape {tuple(labs.shape)} != ({trials}, {pos.numel()})")
    if labs.min() < 0 or labs.max() >= cls.n_classes:
        raise ProbeError(f"{cls.variable_id}: labels out of [0, {cls.n_classes})")
    const = [p for j, p in enumerate(pos.tolist()) if labs[:, j].unique().numel() == 1]
    if const:
        raise ProbeError(f"{cls.variable_id}: label is constant at declared positions {const} "
                         f"over {trials} rows — those columns are free accuracy, not evidence")
    kept = set(causal_positions(inst.labels, pos).tolist())
    if len(kept) != pos.numel():
        raise ProbeError(f"{cls.variable_id}: labels at {[p for p in pos.tolist() if p not in kept]} "
                         "move under suffix resampling — no causal residual site can hold them")
    for i in range(trials):
        inst.check_row(xs[i], labs[i])
    return trials


def causal_positions(label_fn, positions: torch.Tensor, trials: int = 64,
                     seed: int = 0) -> torch.Tensor:
    """Subset of ``positions`` whose labels never move under suffix resampling — the only
    sites a causal residual stream could actually hold. label_fn: [B, INPUT_LEN] -> [B, P]
    aligned with ``positions``."""
    g = torch.Generator().manual_seed(int(seed))
    xs = torch.randint(0, 10, (trials, INPUT_LEN), generator=g)
    y = label_fn(xs)
    keep = []
    for j, p in enumerate(positions.tolist()):
        x2 = xs.clone()
        x2[:, p + 1:] = torch.randint(0, 10, (trials, INPUT_LEN - p - 1), generator=g)
        if torch.equal(label_fn(x2)[:, j], y[:, j]):
            keep.append(p)
    return torch.tensor(keep, dtype=torch.long)


def causal_trim(var: ProbeVariable) -> tuple[torch.Tensor, list[int], list[int]]:
    """(kept positions, kept label columns, dropped positions) after the determinacy check.
    Raises when nothing survives: every declared position's labels depend on future tokens,
    so the variable is unanswerable as posed."""
    declared = var.valid_positions()
    kept = causal_positions(var.labels, declared)
    if kept.numel() == 0:
        raise ProbeError(f"{var.variable_id}: labels depend on future tokens at every "
                         "declared position — no causal residual stream can hold them; "
                         "the variable is unanswerable as posed")
    keep = set(kept.tolist())
    cols = [j for j, p in enumerate(declared.tolist()) if p in keep]
    return kept, cols, [p for p in declared.tolist() if p not in keep]


_CA111_UNSTEP = (7, 7, 6, 7, 7, 6, 7, 7, 6, 7)   # circulant inverse of the a=b=c=1 CA step mod 10


def _ca111_unstep(grid: torch.Tensor, times: int) -> torch.Tensor:
    """Run the cyclic sum-of-3 CA step BACKWARDS (its circulant has det -3, a unit mod 10, so the
    stage grid is recoverable from any later stage — this is what makes the CA labels checkable)."""
    for _ in range(times):
        grid = torch.tensor([sum(_CA111_UNSTEP[(j - i) % INPUT_LEN] * int(grid[j])
                                 for j in range(INPUT_LEN)) % 10 for i in range(INPUT_LEN)])
    return grid


def _pins_label(var: ProbeVariable, x: torch.Tensor, lab: torch.Tensor) -> None:
    """Completeness clause for label corruptions the reference output provably cannot see: the
    row's labels must BE the declared latent. Definitional, so it is a tripwire, not evidence —
    the joint lab/eval clause in the same check_row is what tests the claim."""
    if not torch.equal(lab, var.labels(x.unsqueeze(0))[0]):
        raise ProbeError(f"{var.variable_id}: labels are not the declared latent on {x.tolist()}")


# --- Concrete variables (docs/reference/probe_variable_catalog.md Group A) ---


def _areset_acc(row: list, threshold: int) -> list:
    """AccumulateReset register replay: acc AFTER each input position (0..threshold-1)."""
    out, acc = [], 0
    for v in row:
        acc += v
        if acc >= threshold:
            acc %= threshold
        out.append(acc)
    return out


class Areset13Register(ProbeVariable):
    """The full mod-13 register acc(i) — its mod-10 shadow is what the output exposes."""
    variable_id = "areset13_register"
    algorithm = "areset13_id"
    n_classes = 13

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tensor([_areset_acc(r, self.expr.threshold) for r in inputs.tolist()])

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # acc(9) is only ever emitted mod 10, so its +-10 alias has no successor to expose it
        T = self.expr.threshold
        if not torch.equal(lab % 10, self.expr.eval(x)):
            raise ProbeError(f"{self.variable_id}: labels%10 != eval on {x.tolist()}")
        for k, i in enumerate(self.valid_positions().tolist()):
            if i + 1 >= INPUT_LEN: continue
            z = x.clone(); z[i + 1] = 9   # surgery: a forced 9 next wraps mod 13 iff the claimed acc is the high alias
            if int(self.expr.eval(z)[i + 1]) != (int(lab[k]) + 9) % T % 10:
                raise ProbeError(f"{self.variable_id}: forced 9 at {i + 1} contradicts claimed acc on {x.tolist()}")
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor:
        T = self.expr.threshold
        y = self.labels(x.unsqueeze(0))[0].tolist()
        y[p] = new_class
        for i in range(p + 1, INPUT_LEN):   # re-run the recurrence from the forced acc
            acc = y[i - 1] + int(x[i])
            y[i] = acc % T if acc >= T else acc
        return torch.tensor(y) % 10


class Areset13Tens(ProbeVariable):
    """The never-emitted tens bit 1[acc >= 10] of the mod-13 register."""
    variable_id = "areset13_tens"
    algorithm = "areset13_id"
    n_classes = 2

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        accs = torch.tensor([_areset_acc(r, self.expr.threshold) for r in inputs.tolist()])
        return (accs >= 10).long()[:, self.valid_positions()]

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # acc(0) = x_0 <= 9, so tens(0) is always 0

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # Reconstruct acc = out + 10*tens and demand it satisfies the AccumulateReset recurrence.
        T = self.expr.threshold
        tens = torch.zeros(INPUT_LEN, dtype=torch.long)
        tens[self.valid_positions()] = lab
        acc = (self.expr.eval(x) + 10 * tens).tolist()
        prev = 0
        for i, v in enumerate(x.tolist()):
            want = prev + v
            if want >= T:
                want %= T
            if acc[i] != want:
                raise ProbeError(f"{self.variable_id}: reconstructed acc breaks recurrence on {x.tolist()}")
            prev = acc[i]


class Segrev3SegmentIndex(ProbeVariable):
    """seg_idx(i) = #{j<=i: x_j == boundary} — the reversal's routing key (never emitted)."""
    variable_id = "segrev3_segment_index"
    algorithm = "segrev3_id"
    n_classes = 11   # counts 0..10 (catalog says 10c; 10 boundaries is reachable)

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs == self.expr.boundary_digit).long().cumsum(dim=1)

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # eval emits bd exactly at boundaries, so its running count pins every segment index
        bd, xs, y = self.expr.boundary_digit, x.tolist(), self.expr.eval(x)
        out = [bd if v == bd else 0 for v in xs]
        for s in range(int(lab.max()) + 1):
            seg = [i for i in range(INPUT_LEN) if int(lab[i]) == s and xs[i] != bd]
            for i, j in zip(seg, reversed(seg)):
                out[i] = xs[j]
        if out != y.tolist():
            raise ProbeError(f"{self.variable_id}: seg_idx-rebuilt output != eval on {xs}")
        if not torch.equal(lab, (y == bd).long().cumsum(0)[self.valid_positions()]):
            raise ProbeError(f"{self.variable_id}: seg_idx != running count of eval's boundaries on {xs}")
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        # Force p into segment new_class: both the donor's and the recipient's mirror pairings move.
        bd, xs = self.expr.boundary_digit, x.tolist()
        if xs[p] == bd:
            return None                     # boundary outputs are pinned to bd
        lab = self.labels(x.unsqueeze(0))[0].tolist()
        lab[p] = new_class
        out = [bd if v == bd else 0 for v in xs]
        for s in set(lab):
            seg = [i for i in range(INPUT_LEN) if lab[i] == s and xs[i] != bd]
            for i, j in zip(seg, reversed(seg)):
                out[i] = xs[j]
        return torch.tensor(out)


class NovelSeen(ProbeVariable):
    """seen(i) = 1[x_i appeared in x[:i]] — position 0 excluded (constant there)."""
    variable_id = "novel_seen"
    algorithm = "novel_id"
    n_classes = 2

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for row in inputs.tolist():
            seen: set = set()
            flags = []
            for i, v in enumerate(row):
                flags.append(int(v in seen))
                seen.add(v)
            out.append(flags[1:])
        return torch.tensor(out)

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        if not torch.equal(lab, 1 - self.expr.eval(x)[1:]):
            raise ProbeError(f"{self.variable_id}: seen != 1 - novelty on {x.tolist()}")

    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor:
        y = self.expr.eval(x).clone()   # seen bit only feeds y_p; history is value-based
        y[p] = 1 - new_class
        return y


# --- More concrete variables (docs/reference/probe_variable_catalog.md Groups A/B/C) ---


def _fwd_delta(y: torch.Tensor) -> torch.Tensor:
    """y[i] - y[i-1] (mod 10), with y[-1] treated as 0 — a scan's per-step increment."""
    return (y - torch.cat([y.new_zeros(1), y[:-1]])) % 10


def _last_index(flags: list, sentinel: int) -> list:
    """Running index of the most recent True in ``flags`` (``sentinel`` before the first)."""
    out, last = [], sentinel
    for i, f in enumerate(flags):
        if f:
            last = i
        out.append(last)
    return out


def _segrev_src(vals: list, bd: int) -> list:
    """Source index each output position pulls from under SegmentReverse (boundaries fixed)."""
    L = len(vals)
    src, start = list(range(L)), 0
    for i in range(L):
        if vals[i] == bd:
            seg = list(range(start, i))
            for a, b in zip(seg, reversed(seg)):
                src[a] = b
            start = i + 1
    seg = list(range(start, L))
    for a, b in zip(seg, reversed(seg)):
        src[a] = b
    return src


class Areset13Crossed(ProbeVariable):
    """crossed(i) = 1[acc_pre + x_i >= 13] — the reset trigger the mod-10 output hides."""
    variable_id = "areset13_crossed"
    algorithm = "areset13_id"
    n_classes = 2

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        T, out = self.expr.threshold, []
        for r in inputs.tolist():
            acc, row = 0, []
            for v in r:
                row.append(int(acc + v >= T))
                acc += v
                if acc >= T:
                    acc %= T
            out.append(row)
        return torch.tensor(out)[:, self.valid_positions()]

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # x_0 alone never reaches the threshold

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # rebuild the mod-13 accumulator out of eval + the claimed reset bits
        T, y = self.expr.threshold, self.expr.eval(x).tolist()
        claim = {i: int(lab[k]) for k, i in enumerate(self.valid_positions().tolist())}
        acc = 0
        for i, v in enumerate(x.tolist()):
            cands = [acc + v - T * b for b in ([claim[i]] if i in claim else [0, 1])]
            ok = [c for c in cands if 0 <= c < T and c % 10 == y[i]]
            if len(ok) != 1:
                raise ProbeError(f'areset13_crossed: no in-range accumulator matches eval at {i} on {x.tolist()}')
            acc = ok[0]


class ReccountCurMax(ProbeVariable):
    """cur_max(i) = max(x[0..i]) — the value RecordCount tracks but never emits (it emits a count)."""
    variable_id = "reccount_cur_max"
    algorithm = "reccount_id"
    n_classes = 10

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.cummax(1).values

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        rec = (lab > torch.cat([lab.new_full((1,), -1), lab[:-1]])).long()
        if not torch.equal(rec.cumsum(0) % 10, y):
            raise ProbeError(f"{self.variable_id}: record count from cur_max != eval on {x.tolist()}")
        flag = _fwd_delta(y)   # eval's own record steps say when the max moved, and it moves to x_i
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(lab[k]) != (int(x[i]) if int(flag[i]) else int(lab[k - 1])):
                raise ProbeError(f"{self.variable_id}: cur_max != eval's record step at {i} on {x.tolist()}")
class ReccountIsRecord(ProbeVariable):
    """is_record(i) = 1[x_i > max(x[0..i-1])] — new left-to-right maximum (x_0 always a record)."""
    variable_id = "reccount_is_record"
    algorithm = "reccount_id"
    n_classes = 2

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        cm = inputs.cummax(1).values
        prev = torch.cat([inputs.new_full((inputs.shape[0], 1), -1), cm[:, :-1]], dim=1)
        return (inputs > prev).long()[:, self.valid_positions()]

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # is_record(0) is always 1

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        rec = torch.ones(INPUT_LEN, dtype=torch.long)   # x_0 is always a record
        rec[self.valid_positions()] = lab
        if not torch.equal(rec.cumsum(0) % 10, self.expr.eval(x)):
            raise ProbeError(f"{self.variable_id}: record count from is_record != eval on {x.tolist()}")


class RecordindCurMax(ProbeVariable):
    """cur_max(i) = max(x[0..i]) — RecordIndicator emits only the 0/1 flag, never this value."""
    variable_id = "recordind_cur_max"
    algorithm = "recordind_id"
    n_classes = 10

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.cummax(1).values

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        flag = self.expr.eval(x)   # eval IS the record flag: where it fires the max moved to x_i, elsewhere it held
        rec = (lab > torch.cat([lab.new_full((1,), -1), lab[:-1]])).long()
        if not torch.equal(rec, flag):
            raise ProbeError(f"{self.variable_id}: record flag from cur_max != eval on {x.tolist()}")
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(lab[k]) != (int(x[i]) if int(flag[i]) else int(lab[k - 1])):
                raise ProbeError(f"{self.variable_id}: cur_max != eval's record step at {i} on {x.tolist()}")
def _register_seen_digit(d: int) -> type:
    """A per-digit set-membership probe seen_d(i) = 1[d in x[0..i-1]] for novel_id."""

    class SeenDigit(ProbeVariable):
        variable_id = f"novel_seen{d}"
        algorithm = "novel_id"
        n_classes = 2
        _d = d

        def labels(self, inputs: torch.Tensor) -> torch.Tensor:
            out = []
            for r in inputs.tolist():
                seen, flags = set(), []
                for v in r:
                    flags.append(int(self._d in seen))
                    seen.add(v)
                out.append(flags[1:])
            return torch.tensor(out)

        def valid_positions(self) -> torch.Tensor:
            return torch.arange(1, INPUT_LEN)

        def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
            for k, i in enumerate(self.valid_positions().tolist()):   # d forced at i: eval's novelty there is 1 - seen_d(i)
                xd = x.clone()
                xd[i] = self._d
                if int(self.expr.eval(xd)[i]) != 1 - int(lab[k]):
                    raise ProbeError(f"{self.variable_id}: novelty under a forced {self._d} at {i} "
                                     f"!= 1 - seen on {x.tolist()}")

    SeenDigit.__name__ = f"NovelSeen{d}"
    return SeenDigit


_SEEN_DIGITS = [_register_seen_digit(d) for d in range(10)]


class Segrev3Is3(ProbeVariable):
    """is3(i) = 1[x_i == boundary] — the 3-detector; the output shows it only at boundaries."""
    variable_id = "segrev3_is3"
    algorithm = "segrev3_id"
    n_classes = 2

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs == self.expr.boundary_digit).long()

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # A position outputs the boundary digit iff it IS one (segment values are never bd).
        if not torch.equal(lab, (self.expr.eval(x) == self.expr.boundary_digit).long()):
            raise ProbeError(f"{self.variable_id}: is3 != (eval == boundary) on {x.tolist()}")


class Segrev3Src(ProbeVariable):
    """src(i) = mirror source index feeding output i — the address behind the emitted value."""
    variable_id = "segrev3_src"
    algorithm = "segrev3_id"
    n_classes = 10

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        bd = self.expr.boundary_digit
        src = torch.tensor([_segrev_src(r, bd) for r in inputs.tolist()])
        return src[:, self.valid_positions()]

    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([INPUT_LEN - 1], dtype=torch.long)   # the mirror needs its segment's right edge

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # input surgery: two distinct non-boundary probes at the claimed source pin the address
        bd, vals, y = self.expr.boundary_digit, x.tolist(), self.expr.eval(x)
        for c, i in enumerate(self.valid_positions().tolist()):
            s = int(lab[c])
            if vals[s] != int(y[i]) or (vals[s] == bd and s != i):
                raise ProbeError(f"{self.variable_id}: claimed src {s} contradicts eval at {i} on {vals}")
            for v in (() if vals[s] == bd else ((bd + 1) % 10, (bd + 2) % 10)):
                x2 = x.clone(); x2[s] = v
                if int(self.expr.eval(x2)[i]) != v:
                    raise ProbeError(f"{self.variable_id}: probe {v} at src {s} never reaches {i} on {vals}")
class _OddParity(ProbeVariable):
    """odd(i) = 1[x_i is odd] — the shared parity gate (calibrated on dblodd, reused on cnt/psum)."""
    n_classes = 2

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs % 2).long()

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)


class DbloddOdd(_OddParity):
    variable_id = "dblodd_odd"
    algorithm = "dblodd_id"

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # 0 is a fixed point of the doubling, so eval cannot see the gate there
        if not torch.equal(torch.where(lab.bool(), (x * 2) % 10, x), self.expr.eval(x)):
            raise ProbeError(f"{self.variable_id}: doubling gated by odd != eval on {x.tolist()}")
class CntoddOdd(_OddParity):
    variable_id = "cntodd_odd"
    algorithm = "cntodd_id"

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        if not torch.equal(_fwd_delta(self.expr.eval(x)), lab):
            raise ProbeError(f"{self.variable_id}: count increment != odd on {x.tolist()}")


class PsumoddOdd(_OddParity):
    variable_id = "psumodd_odd"
    algorithm = "psumodd_id"

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        if not torch.equal((_fwd_delta(self.expr.eval(x)) != 0).long(), lab):
            raise ProbeError(f"{self.variable_id}: sum moved iff odd broken on {x.tolist()}")


class PsumoddContrib(ProbeVariable):
    """contrib(i) = x_i if x_i odd else 0 — the payload the parity gate lets into the sum."""
    variable_id = "psumodd_contrib"
    algorithm = "psumodd_id"
    n_classes = 10

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * (inputs % 2)

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        if not torch.equal(_fwd_delta(self.expr.eval(x)), lab):
            raise ProbeError(f"{self.variable_id}: sum increment != contrib on {x.tolist()}")


class Rcnt3Is3(ProbeVariable):
    """is3(i) = 1[x_i == d] — the counted-digit detector (rcnt3 emits only its running count)."""
    variable_id = "rcnt3_is3"
    algorithm = "rcnt3_id"
    n_classes = 2

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs == self.expr.d).long()

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        if not torch.equal(_fwd_delta(self.expr.eval(x)), lab):
            raise ProbeError(f"{self.variable_id}: count increment != is3 on {x.tolist()}")


class Walk12N1(ProbeVariable):
    """n1(i) = (#up-tokens in x[0..i]) mod 10 — one of the two counters behind the emitted (n1-n2)."""
    variable_id = "walk12_n1"
    algorithm = "walk12_id"
    n_classes = 10

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs == self.expr.up).long().cumsum(1) % 10

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        n2 = (x == self.expr.down).long().cumsum(0) % 10
        if not torch.equal((lab - n2) % 10, self.expr.eval(x)):
            raise ProbeError(f"{self.variable_id}: (n1 - n2) != eval on {x.tolist()}")


class Walk12N2(ProbeVariable):
    """n2(i) = (#down-tokens in x[0..i]) mod 10 — the down counter of the signed walk."""
    variable_id = "walk12_n2"
    algorithm = "walk12_id"
    n_classes = 10

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs == self.expr.down).long().cumsum(1) % 10

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        n1 = (x == self.expr.up).long().cumsum(0) % 10
        if not torch.equal((n1 - lab) % 10, self.expr.eval(x)):
            raise ProbeError(f"{self.variable_id}: (n1 - n2) != eval on {x.tolist()}")


class Reset3Last3(ProbeVariable):
    """last3(i) = index of the most recent d in x[0..i] (0 before the first) — the reset pointer."""
    variable_id = "reset3_last3"
    algorithm = "reset3_id"
    n_classes = 10

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        d = self.expr.d
        last = torch.tensor([_last_index([v == d for v in r], 0) for r in inputs.tolist()])
        return last[:, self.valid_positions()]

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # last3(0) = 0 whether or not x_0 == d

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # ResetRamp emits (i - last) mod 10; i - last in [0,9] here, so it reconstructs exactly.
        last = torch.zeros(INPUT_LEN, dtype=torch.long)
        last[self.valid_positions()] = lab
        if not torch.equal((torch.arange(INPUT_LEN) - last) % 10, self.expr.eval(x)):
            raise ProbeError(f"{self.variable_id}: (i - last3) != eval on {x.tolist()}")


class Tresetmax3Last3(ProbeVariable):
    """last3(i) = index of the most recent reset token d (class 10 = none yet) — the segment start."""
    variable_id = "tresetmax3_last3"
    algorithm = "tresetmax3_id"
    n_classes = 11   # positions 0..9 plus a "no reset seen" sentinel (TokenResetMax has no distance output)

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        d = self.expr.d
        return torch.tensor([_last_index([v == d for v in r], 10) for r in inputs.tolist()])

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # x_0 = 0 and a reset at 0 leave eval identical at every position
        vals, y = x.tolist(), self.expr.eval(x)
        for k, i in enumerate(self.valid_positions().tolist()):
            last, prev = int(lab[k]), int(lab[k - 1]) if k else 10
            if max([0] + vals[0 if last == 10 else last + 1:i + 1]) != int(y[i]) or last not in (i, prev):
                raise ProbeError(f"{self.variable_id}: segment max from last3 != eval on {vals}")
            if not k:
                continue
            x2 = x.clone(); x2[i - 1] = 9
            if int(self.expr.eval(x2)[i]) != (0 if last == i else 9):
                raise ProbeError(f"{self.variable_id}: forced-9 reset probe != last3 step at {i} on {vals}")
class ProdHaszero(ProbeVariable):
    """haszero(i) = 1[0 in x[0..i]] — the absorbing-state flag (a lone 0 pins the product to 0)."""
    variable_id = "prod_haszero"
    algorithm = "prod_id"
    n_classes = 2

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs == 0).long().cummax(1).values

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # the mod-10 product absorbs at 0: a 0 factor and a 2*5 collapse read alike
        y = self.expr.eval(x)
        kill = [int(y[j]) == 0 and math.gcd(int(y[j - 1]) if j else 1, 10) == 1 for j in range(INPUT_LEN)]
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(lab[k]) == 1 and int(y[i]) != 0:
                raise ProbeError(f"{self.variable_id}: haszero but nonzero product at {i} on {x.tolist()}")
            if int(lab[k]) == 0 and any(kill[:i + 1]):
                raise ProbeError(f"{self.variable_id}: a unit product collapsed to 0 by {i}, which only a "
                                 f"0 factor can do, on {x.tolist()}")
class ProdMod2(ProbeVariable):
    """prod2(i) = running product mod 2 — a CRT component (1 iff every factor so far is odd)."""
    variable_id = "prod_prod2"
    algorithm = "prod_id"
    n_classes = 2

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (1 - (inputs % 2 == 0).long().cummax(1).values)[:, self.valid_positions()]

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(6)   # beyond position 5 an all-odd prefix is rarer than 1 row in 60

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)[self.valid_positions()]
        if not torch.equal(lab, y % 2):
            raise ProbeError(f"{self.variable_id}: prod mod 2 != eval mod 2 on {x.tolist()}")


class ProdMod5(ProbeVariable):
    """prod5(i) = running product mod 5 — the other CRT component of the mod-10 product."""
    variable_id = "prod_prod5"
    algorithm = "prod_id"
    n_classes = 5

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            p, row = 1, []
            for v in r:
                p = (p * v) % 5
                row.append(p)
            out.append(row)
        return torch.tensor(out)

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        if not torch.equal(lab, self.expr.eval(x) % 5):
            raise ProbeError(f"{self.variable_id}: prod mod 5 != eval mod 5 on {x.tolist()}")


class Dep9f0Floored(ProbeVariable):
    """floored(i) = 1[accumulator saturated at the floor] — the depletion's absorbing flag."""
    variable_id = "dep9f0_floored"
    algorithm = "dep9f0_id"
    n_classes = 2

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        start, floor, out = self.expr.start, self.expr.floor, []
        for r in inputs.tolist():
            acc, row = start, []
            for v in r:
                acc = max(floor, acc - v)
                row.append(int(acc == floor))
            out.append(row)
        return torch.tensor(out)[:, self.valid_positions()]

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(5)   # beyond position 4 the accumulator is floored in >99% of rows

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # start<=9 keeps acc in [floor,9], so acc==floor iff the emitted digit equals floor.
        y = self.expr.eval(x)[self.valid_positions()]
        if not torch.equal(lab, (y == self.expr.floor).long()):
            raise ProbeError(f"{self.variable_id}: floored != (eval == floor) on {x.tolist()}")


class Anc0Anchor(ProbeVariable):
    """anchor(i) = x_a carried to position i — the broadcast reference AnchorSubtract mixes in."""
    variable_id = "anc0_anchor"
    algorithm = "anc0_id"
    n_classes = 10

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, self.expr.a:self.expr.a + 1].repeat(1, INPUT_LEN)

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        if not torch.equal((x - lab) % 10, self.expr.eval(x)):
            raise ProbeError(f"{self.variable_id}: (x - anchor) mod 10 != eval on {x.tolist()}")


class Fclamp0Ge(ProbeVariable):
    """ge(i) = 1[x_i >= x_a] — the comparator bit feeding the floor clamp against the anchor."""
    variable_id = "fclamp0_ge"
    algorithm = "fclamp0_id"
    n_classes = 2

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs >= inputs[:, self.expr.a:self.expr.a + 1]).long()[:, self.valid_positions()]

    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([i for i in range(INPUT_LEN) if i != self.expr.a])   # ge(a) = 1 trivially

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        a, y = self.expr.a % INPUT_LEN, self.expr.eval(x)
        ge = torch.ones(INPUT_LEN, dtype=torch.long)
        ge[self.valid_positions()] = lab
        if not torch.equal(torch.where(ge.bool(), x, torch.full_like(x, int(y[a]))), y):
            raise ProbeError(f"{self.variable_id}: max(x, anchor) from ge != eval on {x.tolist()}")
        x2 = x.clone(); x2[a] = 0   # a zero anchor makes the clamp transparent, so eval shows both sides
        if not torch.equal(lab, (self.expr.eval(x2)[self.valid_positions()] >= int(y[a])).long()):
            raise ProbeError(f"{self.variable_id}: ge != (unclamped eval >= anchor) on {x.tolist()}")
class _GatherSrc(ProbeVariable):
    """src(i): the source index a data-parameterized gather reads for output i (out[i] = x[src(i)])."""
    n_classes = 10

    def _src(self, vals: list) -> list:
        raise NotImplementedError

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self._src(r) for r in inputs.tolist()])[:, self.valid_positions()]

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # input surgery: hold the anchor, tag every slot uniquely, and the gather reads src straight out
        a, vals = self.expr.a % INPUT_LEN, x.tolist()
        x2 = torch.arange(INPUT_LEN)
        x2[vals[a]], x2[a] = a, vals[a]
        y, y2 = self.expr.eval(x), self.expr.eval(x2)
        for k, i in enumerate(self.valid_positions().tolist()):
            s = int(lab[k])
            if vals[s] != int(y[i]) or int(x2[s]) != int(y2[i]):
                raise ProbeError(f"{self.variable_id}: claimed src {s} fails the unique-tag readout at {i} on {vals}")
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor:
        y = self.expr.eval(x).clone()                 # each output gathers independently
        y[p] = int(x[new_class])
        return y


class Drot0Src(_GatherSrc):
    variable_id = "drot0_src"
    algorithm = "drot0_id"

    def _src(self, vals: list) -> list:
        L = len(vals)
        r = vals[self.expr.a % L] % L
        return [(i + r) % L for i in range(L)]


class Refl0Src(_GatherSrc):
    variable_id = "refl0_src"
    algorithm = "refl0_id"

    def _src(self, vals: list) -> list:
        L = len(vals)
        p = vals[self.expr.a % L]
        return [(2 * p - i) % L for i in range(L)]


class Stride0Src(_GatherSrc):
    variable_id = "stride0_src"
    algorithm = "stride0_id"

    def _src(self, vals: list) -> list:
        L = len(vals)
        k = vals[self.expr.a % L]
        return [(i * k) % L for i in range(L)]

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # src(0) = (0*k) % L = 0 for every stride


class Pref0Inprefix(ProbeVariable):
    """inprefix(i) = 1[i < k], k = x_a — the region-membership gate of the data-length prefix reverse."""
    variable_id = "pref0_inprefix"
    algorithm = "pref0_id"
    n_classes = 2

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(INPUT_LEN)
        k = inputs[:, self.expr.a:self.expr.a + 1].clamp(0, INPUT_LEN)
        return (idx.unsqueeze(0) < k).long()[:, self.valid_positions()]

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN - 1)   # k = x_a <= 9, so position 9 is never in-prefix

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # eval reads k below, but k=0 and k=1 are one map: that pair stays definitional
        vals, a, k = x.tolist(), self.expr.a % INPUT_LEN, int(lab.sum())
        tag = torch.arange(INPUT_LEN)
        tag[vals[a]], tag[a] = a, vals[a]
        moved = (self.expr.eval(tag) != tag).nonzero()
        kref = int(moved.max()) + 1 if moved.numel() else min(k, 1)
        if k != kref or not torch.equal(lab, (self.valid_positions() < k).long()):
            raise ProbeError(f"{self.variable_id}: inprefix flags != eval's reversed prefix [0,{kref}) on {vals}")
        if not torch.equal(torch.tensor(vals[:k][::-1] + vals[k:]), self.expr.eval(x)):
            raise ProbeError(f"{self.variable_id}: prefix reverse from inprefix != eval on {vals}")
class Cmpswap0p9Swap(ProbeVariable):
    """swap = 1[x_i > x_j] — the comparator bit (defined only once both positions are in context)."""
    variable_id = "cmpswap0p9_swap"
    algorithm = "cmpswap0p9_id"
    n_classes = 2

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs[:, self.expr.i] > inputs[:, self.expr.j]).long().unsqueeze(1)

    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([max(self.expr.i, self.expr.j)])

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        i, j, vals, y = self.expr.i, self.expr.j, x.tolist(), self.expr.eval(x)
        swap = bool(lab[0])
        if int(y[i]) != (vals[j] if swap else vals[i]) or int(y[j]) != (vals[i] if swap else vals[j]):
            raise ProbeError(f"{self.variable_id}: swap does not reconstruct eval on {vals}")
        if swap and int(y[i]) == int(y[j]):   # the strict comparator never fires on a tie eval would show
            raise ProbeError(f"{self.variable_id}: claimed swap on a tie in eval on {vals}")
class Holdge5Qual(ProbeVariable):
    """qual(i) = 1[x_i passes the threshold gate] — the sample-enable of the hold register."""
    variable_id = "holdge5_qual"
    algorithm = "holdge5_id"
    n_classes = 2

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        t = self.expr.thresh
        return (inputs >= t).long() if self.expr.mode == "ge" else (inputs <= t).long()

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # x_0 = 0 latches the 0 the register already starts at, leaving no trace
        vals, y, reg = x.tolist(), self.expr.eval(x), 0
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(lab[k]):
                reg = vals[i]
            if reg != int(y[i]):
                raise ProbeError(f"{self.variable_id}: hold from qual != eval on {vals}")
            if not k:
                continue
            x2 = x.clone(); x2[i - 1] = 9 if vals[i] != 9 else 5
            if int(self.expr.eval(x2)[i]) != (vals[i] if int(lab[k]) else int(x2[i - 1])):
                raise ProbeError(f"{self.variable_id}: forced-qualifier probe != claimed gate at {i} on {vals}")
class Holdge5Lastqual(ProbeVariable):
    """lastqual(i) = index of the most recent qualifier (class 10 = none) — the hold-register pointer."""
    variable_id = "holdge5_lastqual"
    algorithm = "holdge5_id"
    n_classes = 11

    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        t, ge = self.expr.thresh, self.expr.mode == "ge"
        out = []
        for r in inputs.tolist():
            flags = [(v >= t) if ge else (v <= t) for v in r]
            out.append(_last_index(flags, 10))
        return torch.tensor(out)

    def valid_positions(self) -> torch.Tensor:
        return torch.arange(INPUT_LEN)

    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # x_0 = 0 latches the 0 the register already starts at, leaving no trace
        vals, y = x.tolist(), self.expr.eval(x)
        for k, i in enumerate(self.valid_positions().tolist()):
            last, prev = int(lab[k]), int(lab[k - 1]) if k else 10
            if (0 if last == 10 else vals[last]) != int(y[i]) or last not in (i, prev):
                raise ProbeError(f"{self.variable_id}: register from lastqual != eval on {vals}")
            if not k:
                continue
            x2 = x.clone(); x2[i - 1] = 9 if vals[i] != 9 else 5
            if int(self.expr.eval(x2)[i]) != (vals[i] if last == i else int(x2[i - 1])):
                raise ProbeError(f"{self.variable_id}: forced-qualifier probe != lastqual step at {i} on {vals}")
def _inner_expr(composite_id: str, split_k: int) -> tuple:
    """(inner_expr, inner_id) after stripping the outermost split_k ops of composite_id."""
    parts = algorithms.base_algorithm_id(composite_id).split("_")
    if not 1 <= split_k < len(parts):
        raise ProbeError(f"split_k={split_k} out of range for {composite_id!r} ({len(parts)} ops)")
    inner_id = "_".join(parts[split_k:])
    return algorithms.decode_algorithm_id(inner_id, input_length=INPUT_LEN), inner_id


def _causal_input_positions(inner) -> torch.Tensor:
    """Bridge valid positions: inner-eval determinacy over min(out_len, INPUT_LEN) sites."""
    P = min(inner.output_length(), INPUT_LEN)
    return causal_positions(lambda xs: torch.stack([inner.eval(r)[:P] for r in xs]),
                            torch.arange(P))


class _Fixed(algorithms.DSLExpr):
    """Stub child returning a fixed tensor (for replaying the OUTER ops on labels)."""
    def __init__(self, values: torch.Tensor, input_length: int):
        super().__init__(input_length)
        self.values = values
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return self.values.clone()
    def output_length(self) -> int:
        return int(self.values.numel())


def _graft(base: str, split_k: int, values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Composite output with the depth-``split_k`` intermediate REPLACED by ``values``."""
    outer = algorithms.decode_algorithm_id(base, input_length=INPUT_LEN)
    node = outer
    for _ in range(split_k - 1):
        node = node.child
    node.child = _Fixed(values, INPUT_LEN)
    return outer.eval(x)


def _graft_matters(base: str, split_k: int, values: torch.Tensor, p: int,
                  x: torch.Tensor) -> bool:
    """Does slot p of the intermediate reach the composite output at all on this row?"""
    y = _graft(base, split_k, values, x)
    return any(not torch.equal(_graft(base, split_k, _with(values, p, v), x), y)
               for v in range(10))


def _with(values: torch.Tensor, p: int, v: int) -> torch.Tensor:
    out = values.clone()
    out[p] = v
    return out


def _bridge_positions(base: str, split_k: int, inner, trials: int = 64) -> torch.Tensor:
    """Causally determinate, non-constant, and LOAD-BEARING intermediate slots: the graft
    test can only catch a wrong value where that slot reaches the composite output."""
    xs = _sample(trials)
    vals = torch.stack([inner.eval(r) for r in xs])
    return torch.tensor([p for p in _causal_input_positions(inner).tolist()
                         if vals[:, p].unique().numel() > 1
                         and all(_graft_matters(base, split_k, vals[i], p, xs[i])
                                 for i in range(trials))], dtype=torch.long)


def _split_is_identified(base: str, split_k: int, inner, positions: torch.Tensor,
                         trials: int = 32) -> bool:
    """Does the depth-split_k intermediate differ from every other depth's on these slots?
    When two depths agree, no graft test can tell them apart."""
    depth = len(base.split("_")) - 1
    xs = _sample(trials)
    for other in range(1, depth + 1):
        if other == split_k:
            continue
        alt, _ = _inner_expr(base, other)
        if alt.output_length() <= int(positions.max()):
            continue
        if all(torch.equal(alt.eval(r)[positions], inner.eval(r)[positions]) for r in xs):
            return False
    return True


def bridge_variable(composite_id: str, split_k: int) -> type:
    """A ProbeVariable class for the intermediate value of a composed algorithm's inner
    sub-expression (outer ops [0:split_k) stripped). Registered on creation; idempotent."""
    inner, inner_id = _inner_expr(composite_id, split_k)
    base = algorithms.base_algorithm_id(composite_id)
    vid = f"bridge{split_k}_{inner_id}"
    for v in VARIABLES:
        if v.algorithm == base and v.variable_id == vid:
            return v
    if isinstance(inner, algorithms.Id):
        raise ProbeError(f"{vid}: split_k={split_k} strips every op — the 'intermediate' is "
                         "the input itself, which no probe can distinguish from the embedding")
    positions = _bridge_positions(base, split_k, inner)
    if positions.numel() == 0:
        raise ProbeError(f"{vid}: no causally determinate, non-constant intermediate slot "
                         "reaches the composite output — this split carries nothing")
    if not _split_is_identified(base, split_k, inner, positions):
        raise ProbeError(f"{vid}: another split depth yields the same intermediate — the "
                         "depth-{split_k} claim is not identifiable")
    labels_seen = torch.stack([inner.eval(r) for r in _sample(2048)])[:, positions]

    class BridgeVariable(ProbeVariable):
        variable_id = vid
        algorithm = base
        n_classes = int(labels_seen.max()) + 1     # measured cardinality, not a padded 10
        _inner, _positions, _split_k = inner, positions, split_k

        def labels(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.stack([self._inner.eval(r)[self._positions] for r in inputs])

        def valid_positions(self) -> torch.Tensor:
            return self._positions.clone()

        def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
            _pins_label(self, x, lab)   # the outer ops are many-to-one: some wrong values graft clean
            values = self._inner.eval(x)   # graft the CLAIMED intermediate, not a recomputed one
            values[self._positions] = lab
            y = self.expr.eval(x)
            if not torch.equal(_graft(self.algorithm, self._split_k, values, x), y):
                raise ProbeError(f"{self.variable_id}: grafting the labels at depth "
                                 f"{self._split_k} != full eval on {x.tolist()}")
            p = int(self._positions[int(x.sum()) % self._positions.numel()])
            if not _graft_matters(self.algorithm, self._split_k, values, p, x):
                raise ProbeError(f"{self.variable_id}: composite output is blind to the "
                                 f"intermediate at {p} on {x.tolist()}")

    BridgeVariable.__name__ = f"Bridge_{base}_k{split_k}"
    return BridgeVariable



class Pmaxv9Shadow(ProbeVariable):
    """pmaxv9_shadow on pmaxv9_id (integrated)."""
    variable_id = 'pmaxv9_shadow'
    algorithm = 'pmaxv9_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cummax(inputs, dim=1).values.long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        if not torch.equal(lab, torch.cummax(x, dim=0).values):
            raise ProbeError(f'pmaxv9_shadow: labels != running max on {x.tolist()}')
        y = self.expr.eval(x)
        if not torch.equal(y[lab < 9], lab[lab < 9]):
            raise ProbeError(f'pmaxv9_shadow: eval != shadow where no 9 seen on {x.tolist()}')

bridge_variable('dblodd_walk10_recordind_id', 1)

bridge_variable('dblodd_walk10_recordind_id', 2)


class IndechoGap(ProbeVariable):
    """indecho_gap on indecho_id (integrated)."""
    variable_id = 'indecho_gap'
    algorithm = 'indecho_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            last, lab = {}, []
            for i, v in enumerate(r):
                lab.append(i - last[v] if v in last else 0)
                last[v] = i
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # gap(0) = 0, there is no earlier occurrence
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # gap 0 and gap 1 both make eval print x_i, and nothing else moves
        y, free = self.expr.eval(x), [v for v in range(10) if v not in x.tolist()]
        for k, i in enumerate(self.valid_positions().tolist()):
            g = int(lab[k])
            s = i - g + 1 if g else i
            x2 = x.clone()
            if g > 1 and free:
                x2[s] = free[0]
            if g > i or int(y[i]) != int(x[s]) or int(self.expr.eval(x2)[i]) != int(x2[s]):
                raise ProbeError(f'indecho_gap: claimed echo slot != eval at {i} on {x.tolist()}')
class CosetprodElement(ProbeVariable):
    """cosetprod_element on cosetprod_id (integrated)."""
    variable_id = 'cosetprod_element'
    algorithm = 'cosetprod_id'
    n_classes = 6
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        E = [(0, 1, 2), (0, 2, 1), (1, 2, 0), (1, 0, 2), (2, 0, 1), (2, 1, 0)]
        idx = {e: i for i, e in enumerate(E)}
        out = []
        for r in inputs.tolist():
            cur = (0, 1, 2); lab = []
            for d in r:
                e = E[d % 6]; cur = tuple(e[cur[k]] for k in range(3)); lab.append(idx[cur])
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        E = [(0, 1, 2), (0, 2, 1), (1, 2, 0), (1, 0, 2), (2, 0, 1), (2, 1, 0)]
        C = [0, 1, 1, 0, 2, 2]   # eval prints only the coset; a forced generator + the transition pin the element
        mul = lambda g, e: E.index(tuple(E[g][E[e][j]] for j in range(3)))
        y, pos = self.expr.eval(x), self.valid_positions().tolist()
        for k, i in enumerate(pos):
            e = int(lab[k])
            if int(y[i]) != C[e] or (k and pos[k - 1] == i - 1 and mul(int(x[i]) % 6, int(lab[k - 1])) != e):
                raise ProbeError(f'cosetprod_element: claimed element contradicts eval at {i} on {x.tolist()}')
            if i + 1 < INPUT_LEN:
                x2 = x.clone(); x2[i + 1] = 1
                if int(self.expr.eval(x2)[i + 1]) != C[mul(1, e)]:
                    raise ProbeError(f'cosetprod_element: forced-generator image != eval at {i + 1} on {x.tolist()}')
class Sanc2Regime(ProbeVariable):
    """sanc2_regime on sanc2_id (integrated)."""
    variable_id = 'sanc2_regime'
    algorithm = 'sanc2_id'
    n_classes = 2
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (torch.cumsum(inputs % 2, dim=1) % 2).long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        M = [3, 5, 7, 9, 1, 3, 7, 5, 9, 1]; B = [1, 2, 5, 0, 3, 8, 6, 4, 9, 7]
        y = torch.tensor([(int(x[i]) + M[int(lab[i])] * int(x[0]) + B[int(lab[i])]) % 10 for i in range(10)])
        if not torch.equal(y, self.expr.eval(x)):
            raise ProbeError(f'sanc2_regime: regime-reconstructed output != eval on {x.tolist()}')

bridge_variable('dblodd_walk10_cmpsh1_id', 1)

bridge_variable('dblodd_walk10_cmpsh1_id', 2)


class AddfixF(ProbeVariable):
    """addfix_F on addfix_id (integrated)."""
    variable_id = 'addfix_F'
    algorithm = 'addfix_id'
    n_classes = 11
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs == torch.arange(10)).long().cumsum(dim=1)[:, 1:]
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y, f = self.expr.eval(x), 0
        claim = {i: int(lab[k]) for k, i in enumerate(self.valid_positions().tolist())}
        for i in range(INPUT_LEN):
            d = (int(y[i]) - int(x[i]) - f) % 10   # eval's increment is the 0/1 fixed-point flag
            if d > 1:
                raise ProbeError(f'addfix_F: eval increment at {i} is not a 0/1 flag on {x.tolist()}')
            f += d
            if i in claim and (claim[i] != f or (int(x[i]) + claim[i]) % 10 != int(y[i])):
                raise ProbeError(f'addfix_F: claimed F != eval-reconstructed count at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = self.expr.eval(x).clone()
        y[p] = (int(x[p]) + int(new_class)) % 10
        return y


class Ca2State1(ProbeVariable):
    """ca2_state1 on ca2a1b1c1_id (integrated)."""
    variable_id = 'ca2_state1'
    algorithm = 'ca2a1b1c1_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        y = inputs
        for _ in range(1):
            y = (torch.roll(y, 1, dims=1) + y + torch.roll(y, -1, dims=1)) % 10
        return y.long()[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([INPUT_LEN - 1], dtype=torch.long)   # the cyclic rule reads x[i+1]
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        s = _ca111_unstep(self.expr.eval(x), 1)   # the stage grid, read back out of eval
        for c, i in enumerate(self.valid_positions().tolist()):
            if int(lab[c]) != int(s[i]):
                raise ProbeError(f'ca2_state1: stage value != inverse-stepped eval at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = x
        for _ in range(1):
            y = (torch.roll(y, 1) + y + torch.roll(y, -1)) % 10
        y = y.clone(); y[p] = int(new_class)
        for _ in range(1):
            y = (torch.roll(y, 1) + y + torch.roll(y, -1)) % 10
        return y


class Ca3State1(ProbeVariable):
    """ca3_state1 on ca3a1b1c1_id (integrated)."""
    variable_id = 'ca3_state1'
    algorithm = 'ca3a1b1c1_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        y = inputs
        for _ in range(1):
            y = (torch.roll(y, 1, dims=1) + y + torch.roll(y, -1, dims=1)) % 10
        return y.long()[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([INPUT_LEN - 1], dtype=torch.long)   # the cyclic rule reads x[i+1]
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        s = _ca111_unstep(self.expr.eval(x), 2)   # the stage grid, read back out of eval
        for c, i in enumerate(self.valid_positions().tolist()):
            if int(lab[c]) != int(s[i]):
                raise ProbeError(f'ca3_state1: stage value != inverse-stepped eval at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = x
        for _ in range(1):
            y = (torch.roll(y, 1) + y + torch.roll(y, -1)) % 10
        y = y.clone(); y[p] = int(new_class)
        for _ in range(2):
            y = (torch.roll(y, 1) + y + torch.roll(y, -1)) % 10
        return y


class Ca3State2(ProbeVariable):
    """ca3_state2 on ca3a1b1c1_id (integrated)."""
    variable_id = 'ca3_state2'
    algorithm = 'ca3a1b1c1_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        y = inputs
        for _ in range(2):
            y = (torch.roll(y, 1, dims=1) + y + torch.roll(y, -1, dims=1)) % 10
        return y.long()[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([INPUT_LEN - 1], dtype=torch.long)   # the cyclic rule reads x[i+1]
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        s = _ca111_unstep(self.expr.eval(x), 1)   # the stage grid, read back out of eval
        for c, i in enumerate(self.valid_positions().tolist()):
            if int(lab[c]) != int(s[i]):
                raise ProbeError(f'ca3_state2: stage value != inverse-stepped eval at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = x
        for _ in range(2):
            y = (torch.roll(y, 1) + y + torch.roll(y, -1)) % 10
        y = y.clone(); y[p] = int(new_class)
        for _ in range(1):
            y = (torch.roll(y, 1) + y + torch.roll(y, -1)) % 10
        return y


class Ca5State1(ProbeVariable):
    """ca5_state1 on ca5a1b1c1_id (integrated)."""
    variable_id = 'ca5_state1'
    algorithm = 'ca5a1b1c1_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        y = inputs
        for _ in range(1):
            y = (torch.roll(y, 1, dims=1) + y + torch.roll(y, -1, dims=1)) % 10
        return y.long()[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([INPUT_LEN - 1], dtype=torch.long)   # the cyclic rule reads x[i+1]
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        s = _ca111_unstep(self.expr.eval(x), 4)   # the stage grid, read back out of eval
        for c, i in enumerate(self.valid_positions().tolist()):
            if int(lab[c]) != int(s[i]):
                raise ProbeError(f'ca5_state1: stage value != inverse-stepped eval at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = x
        for _ in range(1):
            y = (torch.roll(y, 1) + y + torch.roll(y, -1)) % 10
        y = y.clone(); y[p] = int(new_class)
        for _ in range(4):
            y = (torch.roll(y, 1) + y + torch.roll(y, -1)) % 10
        return y


class Ca5State2(ProbeVariable):
    """ca5_state2 on ca5a1b1c1_id (integrated)."""
    variable_id = 'ca5_state2'
    algorithm = 'ca5a1b1c1_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        y = inputs
        for _ in range(2):
            y = (torch.roll(y, 1, dims=1) + y + torch.roll(y, -1, dims=1)) % 10
        return y.long()[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([INPUT_LEN - 1], dtype=torch.long)   # the cyclic rule reads x[i+1]
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        s = _ca111_unstep(self.expr.eval(x), 3)   # the stage grid, read back out of eval
        for c, i in enumerate(self.valid_positions().tolist()):
            if int(lab[c]) != int(s[i]):
                raise ProbeError(f'ca5_state2: stage value != inverse-stepped eval at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = x
        for _ in range(2):
            y = (torch.roll(y, 1) + y + torch.roll(y, -1)) % 10
        y = y.clone(); y[p] = int(new_class)
        for _ in range(3):
            y = (torch.roll(y, 1) + y + torch.roll(y, -1)) % 10
        return y


class Ca5State3(ProbeVariable):
    """ca5_state3 on ca5a1b1c1_id (integrated)."""
    variable_id = 'ca5_state3'
    algorithm = 'ca5a1b1c1_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        y = inputs
        for _ in range(3):
            y = (torch.roll(y, 1, dims=1) + y + torch.roll(y, -1, dims=1)) % 10
        return y.long()[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([INPUT_LEN - 1], dtype=torch.long)   # the cyclic rule reads x[i+1]
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        s = _ca111_unstep(self.expr.eval(x), 2)   # the stage grid, read back out of eval
        for c, i in enumerate(self.valid_positions().tolist()):
            if int(lab[c]) != int(s[i]):
                raise ProbeError(f'ca5_state3: stage value != inverse-stepped eval at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = x
        for _ in range(3):
            y = (torch.roll(y, 1) + y + torch.roll(y, -1)) % 10
        y = y.clone(); y[p] = int(new_class)
        for _ in range(2):
            y = (torch.roll(y, 1) + y + torch.roll(y, -1)) % 10
        return y


class Ca5State4(ProbeVariable):
    """ca5_state4 on ca5a1b1c1_id (integrated)."""
    variable_id = 'ca5_state4'
    algorithm = 'ca5a1b1c1_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        y = inputs
        for _ in range(4):
            y = (torch.roll(y, 1, dims=1) + y + torch.roll(y, -1, dims=1)) % 10
        return y.long()[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([INPUT_LEN - 1], dtype=torch.long)   # the cyclic rule reads x[i+1]
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        s = _ca111_unstep(self.expr.eval(x), 1)   # the stage grid, read back out of eval
        for c, i in enumerate(self.valid_positions().tolist()):
            if int(lab[c]) != int(s[i]):
                raise ProbeError(f'ca5_state4: stage value != inverse-stepped eval at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = x
        for _ in range(4):
            y = (torch.roll(y, 1) + y + torch.roll(y, -1)) % 10
        y = y.clone(); y[p] = int(new_class)
        for _ in range(1):
            y = (torch.roll(y, 1) + y + torch.roll(y, -1)) % 10
        return y


class Chain2Ptr1(ProbeVariable):
    """chain2_ptr1 on chain10d2_id (integrated)."""
    variable_id = 'chain2_ptr1'
    algorithm = 'chain10d2_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(10).unsqueeze(0).expand(inputs.shape[0], 10)
        for _ in range(1):
            idx = torch.gather(inputs, 1, idx)
        return idx.long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        if not torch.equal(x[x[lab]], self.expr.eval(x)):
            raise ProbeError(f'chain2_ptr1: continued hops != eval on {x.tolist()}')
        for k, i in enumerate(self.valid_positions().tolist()):
            x2 = torch.arange(INPUT_LEN); x2[i] = int(x[i])   # identity slots make eval print the hop-1 address
            if int(self.expr.eval(x2)[i]) != int(lab[k]):
                raise ProbeError(f'chain2_ptr1: identity-graft readout != claimed pointer at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        idx = int(new_class)
        for _ in range(1):
            idx = int(x[idx])
        y = self.expr.eval(x).clone()
        y[p] = int(x[idx])
        return y


class Chain3Ptr1(ProbeVariable):
    """chain3_ptr1 on chain10d3_id (integrated)."""
    variable_id = 'chain3_ptr1'
    algorithm = 'chain10d3_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(10).unsqueeze(0).expand(inputs.shape[0], 10)
        for _ in range(1):
            idx = torch.gather(inputs, 1, idx)
        return idx.long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        if not torch.equal(x[x[x[lab]]], self.expr.eval(x)):
            raise ProbeError(f'chain3_ptr1: continued hops != eval on {x.tolist()}')
        for k, i in enumerate(self.valid_positions().tolist()):
            x2 = torch.arange(INPUT_LEN); x2[i] = int(x[i])   # identity slots make eval print the hop-1 address
            if int(self.expr.eval(x2)[i]) != int(lab[k]):
                raise ProbeError(f'chain3_ptr1: identity-graft readout != claimed pointer at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        idx = int(new_class)
        for _ in range(2):
            idx = int(x[idx])
        y = self.expr.eval(x).clone()
        y[p] = int(x[idx])
        return y


class Chain3Ptr2(ProbeVariable):
    """chain3_ptr2 on chain10d3_id (integrated)."""
    variable_id = 'chain3_ptr2'
    algorithm = 'chain10d3_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(INPUT_LEN).unsqueeze(0).expand(inputs.shape[0], INPUT_LEN)
        for _ in range(2):
            idx = torch.gather(inputs, 1, idx)
        return idx.long()[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([INPUT_LEN - 1], dtype=torch.long)   # a pointer can aim forward
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        for c, i in enumerate(self.valid_positions().tolist()):
            if int(x[int(x[int(lab[c])])]) != int(y[i]):
                raise ProbeError(f'chain3_ptr2: continued hops != eval at {i} on {x.tolist()}')
            a = int(x[i])
            x2 = torch.arange(INPUT_LEN); x2[i], x2[a] = a, int(x[a])   # identity slots leave only the hop-2 address
            if int(self.expr.eval(x2)[i]) != int(lab[c]):
                raise ProbeError(f'chain3_ptr2: identity-graft readout != claimed pointer at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        idx = int(new_class)
        for _ in range(1):
            idx = int(x[idx])
        y = self.expr.eval(x).clone()
        y[p] = int(x[idx])
        return y


class Sjf1Buf(ProbeVariable):
    """sjf1_buf on sjf1_id (integrated)."""
    variable_id = 'sjf1_buf'
    algorithm = 'sjf1_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cummax(inputs, dim=1).values[:, :-1].long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(y[i]) != min(int(lab[k]), int(x[i])):
                raise ProbeError(f'sjf1_buf: min(buf, x) != eval at {i} on {x.tolist()}')
            x2 = x.clone(); x2[i] = 9   # a forced 9 arrival makes eval print the buffer itself
            if int(self.expr.eval(x2)[i]) != int(lab[k]):
                raise ProbeError(f'sjf1_buf: forced-9 arrival != claimed buffer at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = self.expr.eval(x).clone()
        M = int(new_class)
        for i in range(p, 10):
            y[i] = min(M, int(x[i]))
            M = max(M, int(x[i]))
        return y


class Psum3Inner1(ProbeVariable):
    """psum3_inner1 on psum_psum_psum_id (integrated)."""
    variable_id = 'psum3_inner1'
    algorithm = 'psum_psum_psum_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        y = inputs
        for _ in range(1):
            y = torch.cumsum(y, dim=1) % 10
        return y.long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = lab.clone()
        for _ in range(2):
            y = torch.cumsum(y, dim=0) % 10
        if not torch.equal(y, self.expr.eval(x)):
            raise ProbeError(f'psum3_inner1: outer prefix-sums of inner != eval on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        a = x.clone()
        for _ in range(0):
            a = torch.cumsum(a, dim=0) % 10
        s = torch.cumsum(a, dim=0) % 10
        s[p] = int(new_class)
        for i in range(p + 1, 10):
            s[i] = (s[i - 1] + a[i]) % 10
        y = s
        for _ in range(2):
            y = torch.cumsum(y, dim=0) % 10
        return y


class Psum3Inner2(ProbeVariable):
    """psum3_inner2 on psum_psum_psum_id (integrated)."""
    variable_id = 'psum3_inner2'
    algorithm = 'psum_psum_psum_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        y = inputs
        for _ in range(2):
            y = torch.cumsum(y, dim=1) % 10
        return y.long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = lab.clone()
        for _ in range(1):
            y = torch.cumsum(y, dim=0) % 10
        if not torch.equal(y, self.expr.eval(x)):
            raise ProbeError(f'psum3_inner2: outer prefix-sums of inner != eval on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        a = x.clone()
        for _ in range(1):
            a = torch.cumsum(a, dim=0) % 10
        s = torch.cumsum(a, dim=0) % 10
        s[p] = int(new_class)
        for i in range(p + 1, 10):
            s[i] = (s[i - 1] + a[i]) % 10
        y = s
        for _ in range(1):
            y = torch.cumsum(y, dim=0) % 10
        return y


class DuelEsum(ProbeVariable):
    """duel_esum on duelsum_id (integrated)."""
    variable_id = 'duel_esum'
    algorithm = 'duelsum_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (torch.cumsum(torch.where(inputs % 2 == 0, inputs, torch.zeros_like(inputs)), dim=1) % 10).long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        e = o = 0
        for i in range(10):
            v = int(x[i])
            if v % 2 == 0: e += v
            else: o += v
            if int(lab[i]) != e % 10:
                raise ProbeError(f'duel_esum: label != even-sum residue at {i} on {x.tolist()}')
            if e >= o and int(y[i]) != int(lab[i]):
                raise ProbeError(f'duel_esum: leader print != esum at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        e = o = 0
        for i in range(int(p) + 1):
            v = int(x[i])
            if v % 2 == 0: e += v
            else: o += v
        if e <= o:
            return None
        y = self.expr.eval(x).clone()
        y[p] = int(new_class)
        return y


class DuelOsum(ProbeVariable):
    """duel_osum on duelsum_id (integrated)."""
    variable_id = 'duel_osum'
    algorithm = 'duelsum_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (torch.cumsum(torch.where(inputs % 2 == 1, inputs, torch.zeros_like(inputs)), dim=1) % 10).long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        e = o = 0
        for i in range(10):
            v = int(x[i])
            if v % 2 == 0: e += v
            else: o += v
            if int(lab[i]) != o % 10:
                raise ProbeError(f'duel_osum: label != odd-sum residue at {i} on {x.tolist()}')
            if o > e and int(y[i]) != int(lab[i]):
                raise ProbeError(f'duel_osum: leader print != osum at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        e = o = 0
        for i in range(int(p) + 1):
            v = int(x[i])
            if v % 2 == 0: e += v
            else: o += v
        if o <= e:
            return None
        y = self.expr.eval(x).clone()
        y[p] = int(new_class)
        return y


class DuelLead(ProbeVariable):
    """duel_lead on duelsum_id (integrated)."""
    variable_id = 'duel_lead'
    algorithm = 'duelsum_id'
    n_classes = 2
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        e = torch.cumsum(torch.where(inputs % 2 == 0, inputs, torch.zeros_like(inputs)), dim=1)
        o = torch.cumsum(torch.where(inputs % 2 == 1, inputs, torch.zeros_like(inputs)), dim=1)
        return (e >= o).long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        e = o = 0
        for i in range(10):
            v = int(x[i])
            if v % 2 == 0: e += v
            else: o += v
            if int(lab[i]) != int(e >= o):
                raise ProbeError(f'duel_lead: label != lead bit at {i} on {x.tolist()}')
            if int(y[i]) != (e % 10 if int(lab[i]) else o % 10):
                raise ProbeError(f'duel_lead: printed residue != lead register at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        e = o = 0
        for i in range(int(p) + 1):
            v = int(x[i])
            if v % 2 == 0: e += v
            else: o += v
        if e == o:
            return None
        y = self.expr.eval(x).clone()
        y[p] = (e if int(new_class) == 1 else o) % 10
        return y


class NetrmLoser(ProbeVariable):
    """netrm_loser on netrm_id (integrated)."""
    variable_id = 'netrm_loser'
    algorithm = 'netrm_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        M = torch.cummax(inputs, dim=1).values
        m = torch.cummin(inputs, dim=1).values
        return torch.where(2 * inputs >= M + m, m, M).long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        M = torch.cummax(x, 0).values
        m = torch.cummin(x, 0).values
        for i in range(10):
            if not ((int(y[i]) == int(M[i]) and int(lab[i]) == int(m[i])) or
                    (int(y[i]) == int(m[i]) and int(lab[i]) == int(M[i]))):
                raise ProbeError(f'netrm_loser: winner/loser pair != (M, m) at {i} on {x.tolist()}')


class ConvfixSynd(ProbeVariable):
    """convfix_synd on convfix_id (integrated)."""
    variable_id = 'convfix_synd'
    algorithm = 'convfix_id'
    n_classes = 2
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        s = inputs & 1
        z = ((inputs >> 1) & 1) ^ s
        z[:, 1:] = z[:, 1:] ^ s[:, :-1]
        return z[:, 1:].long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        xs = x.tolist()
        s = [d & 1 for d in xs]
        for k, i in enumerate(self.valid_positions().tolist()):   # force a neighbour syndrome to 1: eval then leaks z_i
            xr, j = x.clone(), min(i + 1, INPUT_LEN - 1)
            if j > i:
                xr[j] = 1 ^ s[i]
            else:
                xr[i - 2] = ((xs[i - 1] >> 1) & 1) ^ s[i - 1] ^ 1
            if int(lab[k]) != int(self.expr.eval(xr)[j]) ^ s[j - 1]:
                raise ProbeError(f'{self.variable_id}: syndrome leaked by a forced neighbour != label '
                                 f'at {i} on {xs}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        xs = x.tolist()
        s = [d & 1 for d in xs]
        z = [((xs[i] >> 1) & 1) ^ s[i] ^ (s[i - 1] if i else 0) for i in range(10)]
        z[p] = int(new_class)
        return torch.tensor([0] + [s[i - 1] ^ (z[i - 1] & z[i]) for i in range(1, 10)])


class Lagf3K(ProbeVariable):
    """lagf3_k on lagf3_id (integrated)."""
    variable_id = 'lagf3_k'
    algorithm = 'lagf3_id'
    n_classes = 11
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cumsum((inputs == 3).long(), dim=1)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # k reaches eval only as (i - k) mod 10, so k and k + 10 are aliases
        vals, y = x.tolist(), self.expr.eval(x)
        non3 = [v for v in range(10) if v != 3]
        x2 = torch.tensor([3 if v == 3 else non3[j % 9] for j, v in enumerate(vals)])
        y2 = self.expr.eval(x2)
        for k, i in enumerate(self.valid_positions().tolist()):
            s = (i - int(lab[k])) % 10
            if int(y[i]) != vals[s] or int(y2[i]) != int(x2[s]):
                raise ProbeError(f'lagf3_k: fetch from count-lag != eval at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = self.expr.eval(x).clone()
        y[p] = int(x[(int(p) - int(new_class)) % 10])
        return y


class NearidxLoseridx(ProbeVariable):
    """nearidx_loseridx on nearidx_id (integrated)."""
    variable_id = 'nearidx_loseridx'
    algorithm = 'nearidx_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            M, m, iM, im, lab = r[0], r[0], 0, 0, []
            for i, v in enumerate(r):
                if v > M:
                    M, iM = v, i
                if v < m:
                    m, im = v, i
                lab.append(im if 2 * v >= M + m else iM)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(2, INPUT_LEN)   # both extremes are still x_0 at positions 0 and 1
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y, r = self.expr.eval(x), x.tolist()
        loser = torch.zeros(INPUT_LEN, dtype=torch.long)
        loser[self.valid_positions()] = lab
        M, m, iM, im = r[0], r[0], 0, 0
        for i, v in enumerate(r):
            if v > M:
                M, iM = v, i
            if v < m:
                m, im = v, i
            if {int(y[i]), int(loser[i])} != {iM, im}:
                raise ProbeError(f'nearidx_loseridx: winner/loser indices != extremes at {i} on {x.tolist()}')


class Ts3Last3(ProbeVariable):
    """ts3_last3 on tresetsum3_id (integrated)."""
    variable_id = 'ts3_last3'
    algorithm = 'tresetsum3_id'
    n_classes = 11
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        B = inputs.shape[0]
        out = torch.full((B, 10), 10, dtype=torch.long)
        for b in range(B):
            last = 10
            for i in range(10):
                if int(inputs[b, i]) == 3:
                    last = i
                out[b, i] = last
        return out
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # x_0 = 0 and a reset at 0 leave eval identical at every position
        y = self.expr.eval(x)
        for k, i in enumerate(self.valid_positions().tolist()):
            last, prev = int(lab[k]), int(lab[k - 1]) if k else 10
            if int(y[i]) != int(x[0 if last == 10 else last + 1:i + 1].sum()) % 10 or last not in (i, prev):
                raise ProbeError(f'ts3_last3: segment sum from pointer != eval at {i} on {x.tolist()}')
            if not k:
                continue
            x2 = x.clone(); x2[i - 1] = 3
            if int(self.expr.eval(x2)[i]) != (0 if last == i else int(x[i])):
                raise ProbeError(f'ts3_last3: forced-reset probe != claimed pointer step at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        if int(new_class) != 10 and int(new_class) > int(p):
            return None
        lo = 0 if int(new_class) == 10 else int(new_class) + 1
        y = self.expr.eval(x).clone()
        y[p] = int(x[lo:int(p) + 1].sum()) % 10
        return y


class PgeDepth(ProbeVariable):
    """pge_depth on pge_id (integrated)."""
    variable_id = 'pge_depth'
    algorithm = 'pge_id'
    n_classes = 11
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            st, lab = [], []
            for v in r:
                while st and st[-1] <= v:
                    st.pop()
                st.append(v)
                lab.append(len(st))
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # the stack always holds exactly x_0 at position 0
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y, v, pos = self.expr.eval(x).tolist(), x.tolist(), self.valid_positions().tolist()
        col = {p: k for k, p in enumerate(pos)}
        for k, i in enumerate(pos):
            if not y[i]:
                if int(lab[k]) != 1:
                    raise ProbeError(f'pge_depth: eval 0 (empty stack) must be depth 1 at {i} on {v}')
                continue
            j = next((t for t in range(i - 1, -1, -1) if v[t] == y[i]), -1)   # pge site read off eval
            d = 1 if j == 0 else int(lab[col[j]])   # position 0 always holds the one-element stack
            if j < 0 or int(lab[k]) != d + 1:
                raise ProbeError(f'pge_depth: depth != depth(previous-greater)+1 at {i} on {v}')
class AddpmaxM(ProbeVariable):
    """addpmax_m on addpmax_id (integrated)."""
    variable_id = 'addpmax_m'
    algorithm = 'addpmax_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cummax(inputs, dim=1).values[:, :-1].long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        for i in range(1, 10):
            if int(y[i]) != (int(x[i]) + int(lab[i - 1])) % 10:
                raise ProbeError(f'addpmax_m: (x + m) % 10 != eval at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = self.expr.eval(x).clone()
        y[p] = (int(x[p]) + int(new_class)) % 10
        return y


class MaxrunCur(ProbeVariable):
    """maxrun_cur on maxrun_id (integrated)."""
    variable_id = 'maxrun_cur'
    algorithm = 'maxrun_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for row in inputs.tolist():
            r, prev, lab = 0, None, []
            for v in row:
                r = r + 1 if v == prev else 1
                lab.append(r)
                prev = v
            out.append(lab[2:9])
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y, pos = self.expr.eval(x), self.valid_positions().tolist()   # surgery: a flat prefix makes eval report the run at i
        for k, i in enumerate(pos):
            c = int(lab[k])
            if int(y[i]) != max(int(y[i - 1]), min(c, 9)) or not 1 <= c <= i + 1:
                raise ProbeError(f'maxrun_cur: eval != running best of claimed run at {i} on {x.tolist()}')
            if k and pos[k - 1] == i - 1 and c not in (1, int(lab[k - 1]) + 1):
                raise ProbeError(f'maxrun_cur: run neither reset nor incremented at {i} on {x.tolist()}')
            for n in (c, c + 1):
                if i - n + 1 < 0: continue
                z = x.clone(); z[:i - n + 1] = (int(x[i - n + 1]) + 1 + torch.arange(i - n + 1) % 2) % 10
                if (int(self.expr.eval(z)[i]) == n) != (n == c):
                    raise ProbeError(f'maxrun_cur: equal run at {i} is not exactly {c} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        r, best, y, prev = 0, 0, [], None
        for i, v in enumerate(x.tolist()):
            r = r + 1 if v == prev else 1
            if i == int(p):
                r = int(new_class)
            best = max(best, r)
            y.append(min(best, 9))
            prev = v
        return torch.tensor(y)


class Cgrev09Branchbit(ProbeVariable):
    """cgrev09_branchbit on cgrev09_id (integrated)."""
    variable_id = 'cgrev09_branchbit'
    algorithm = 'cgrev09_id'
    n_classes = 2
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        bit = (inputs[:, 0] > inputs[:, INPUT_LEN - 1]).long().unsqueeze(1).expand(-1, INPUT_LEN)
        return bit[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([INPUT_LEN - 1], dtype=torch.long)   # the branch compares x[0] with x[9]
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        for c, i in enumerate(self.valid_positions().tolist()):
            want = torch.flip(x, [0]) if int(lab[c]) else x
            if not torch.equal(y, want):
                raise ProbeError(f'cgrev09_branchbit: bit at {i} inconsistent with eval on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        return torch.flip(x, [0]) if int(new_class) == 1 else x.clone()


bridge_variable('dblodd_psum_novel_id', 1)

bridge_variable('dblodd_psum_novel_id', 2)


class Rmaj5Margin(ProbeVariable):
    """rmaj5_margin on rmaj5_id (integrated)."""
    variable_id = 'rmaj5_margin'
    algorithm = 'rmaj5_id'
    n_classes = 11
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        c = torch.cumsum((inputs >= 5).long(), dim=1)
        m = 2 * c - (torch.arange(10) + 1)
        return (m.clamp(-5, 5) + 5).long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        c = 0
        for i in range(10):
            c += int(int(x[i]) >= 5)
            m = 2 * c - (i + 1)
            if int(lab[i]) != max(-5, min(5, m)) + 5:
                raise ProbeError(f'rmaj5_margin: label != clipped margin at {i} on {x.tolist()}')
            if int(y[i]) != int(int(lab[i]) > 5):
                raise ProbeError(f'rmaj5_margin: sign of margin != eval at {i} on {x.tolist()}')

bridge_variable('reset1_cmpsh1_id', 1)

bridge_variable('reset1_novel_id', 1)

bridge_variable('psum_novel_id', 1)


class RecdistLastrec(ProbeVariable):
    """recdist_lastrec on recdist_id (integrated)."""
    variable_id = 'recdist_lastrec'
    algorithm = 'recdist_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            cur, last, lab = -1, 0, []
            for i, v in enumerate(r):
                if v > cur:
                    cur, last = v, i
                lab.append(last)
            out.append(lab[1:])
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        for i in range(1, 10):
            if int(y[i]) != i - int(lab[i - 1]):
                raise ProbeError(f'recdist_lastrec: i - last != eval at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        if int(new_class) > int(p):
            return None
        y = self.expr.eval(x).clone()
        y[p] = int(p) - int(new_class)
        return y

bridge_variable('rcnt1_cmpsh1_id', 1)

bridge_variable('holdge5_diffsh1_id', 1)


class PrevsegmaxCurmax(ProbeVariable):
    """prevsegmax_curmax on prevsegmax3_id (integrated)."""
    variable_id = 'prevsegmax_curmax'
    algorithm = 'prevsegmax3_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            cur, lab = 0, []
            for v in r:
                if v == 3:
                    cur = 0
                    lab.append(0)
                else:
                    cur = max(cur, v)
                    lab.append(cur)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        for k, i in enumerate(self.valid_positions().tolist()):   # a reset forced at i+1 makes eval emit cur(i)
            j = min(i + 1, INPUT_LEN - 1)
            xr = x.clone()
            xr[j] = 3
            cur = int(self.expr.eval(xr)[j])
            if j == i:
                cur = 0 if int(x[i]) == 3 else max(cur, int(x[i]))
            if int(lab[k]) != cur:
                raise ProbeError(f'{self.variable_id}: cur-max read from a forced reset != label '
                                 f'at {i} on {x.tolist()}')
class Segdist3Seen5(ProbeVariable):
    """segdist3_seen5 on segdist3_id (integrated)."""
    variable_id = 'segdist3_seen5'
    algorithm = 'segdist3_id'
    n_classes = 2
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            s, lab = 0, []
            for v in r:
                if v == 3: s = 0
                elif v == 5: s = 1
                lab.append(s)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # surgery: writing 5 at j leaves the distinct count flat iff j's segment already holds a 5
        y, v = self.expr.eval(x).tolist(), x.tolist()
        for k, i in enumerate(self.valid_positions().tolist()):
            j = i + 1 if i + 1 < INPUT_LEN else i
            z = x.clone(); z[j] = 5
            seen = int(self.expr.eval(z)[j]) == y[j - 1]
            want = int(seen) if j > i else int(v[i] != 3 and (v[i] == 5 or seen))
            if int(lab[k]) != want:
                raise ProbeError(f'segdist3_seen5: probed 5-in-segment at {i} != claimed bit on {x.tolist()}')


class FuPointer(ProbeVariable):
    """fu_pointer on firstuniq_id (integrated)."""
    variable_id = 'fu_pointer'
    algorithm = 'firstuniq_id'
    n_classes = 11
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            lab = []
            for i in range(INPUT_LEN):
                pre = r[:i + 1]
                cnt = {v: pre.count(v) for v in pre}
                home = 10
                for j, v in enumerate(pre):
                    if cnt[v] == 1:
                        home = j; break
                lab.append(home)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # x_0 is unique in its own prefix, so pointer(0) = 0
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y, r = self.expr.eval(x), x.tolist()
        x2 = x.clone(); x2[x == 0], x2[x == 1] = 1, 0   # swapping two digit values moves the fetched digit, not the pointer
        y2 = self.expr.eval(x2)
        for k, i in enumerate(self.valid_positions().tolist()):
            home = int(lab[k])
            if home > i and home != 10:
                raise ProbeError(f'fu_pointer: pointer {home} is past {i} on {x.tolist()}')
            if int(y[i]) != (0 if home == 10 else r[home]) or int(y2[i]) != (0 if home == 10 else int(x2[home])):
                raise ProbeError(f'fu_pointer: digit at pointer != eval at {i} on {x.tolist()}')
class AloH(ProbeVariable):
    """alo_h on addlastodd_id (integrated)."""
    variable_id = 'alo_h'
    algorithm = 'addlastodd_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            h, lab = 0, []
            for v in r:
                if v % 2 == 1: h = v
                lab.append(h)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        for i, v in enumerate(x.tolist()):
            if int(y[i]) != (v + int(lab[i])) % 10:
                raise ProbeError(f'alo_h: (x + held_odd) % 10 != eval at {i} on {x.tolist()}')


class BmCounter(ProbeVariable):
    """bm_counter on bmcand_id (integrated)."""
    variable_id = 'bm_counter'
    algorithm = 'bmcand_id'
    n_classes = 11
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            cand, score, lab = None, 0, []
            for v in r:
                if score == 0: cand, score = v, 1
                elif v == cand: score += 1
                else: score -= 1
                lab.append(min(score, 10))
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # the score is 1 after the first token
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # the candidate trajectory comes from eval; the claimed score must drive its every move
        y, v, dec = self.expr.eval(x).tolist(), x.tolist(), self.valid_positions().tolist()
        claim = {i: int(lab[k]) for k, i in enumerate(dec)}
        for i in dec:
            if i - 1 not in claim:
                continue
            prev, cand = claim[i - 1], y[i - 1]
            want = (v[i], 1) if prev == 0 else (cand, min(prev + (1 if v[i] == cand else -1), 10))
            if (y[i], claim[i]) != want:
                raise ProbeError(f'bm_counter: score/candidate step at {i} breaks Boyer-Moore on {x.tolist()}')


class Hold2Cur(ProbeVariable):
    """hold2_cur on hold2ge5_id (integrated)."""
    variable_id = 'hold2_cur'
    algorithm = 'hold2ge5_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            q, lab = [], []
            for v in r:
                if v >= 5: q.append(v)
                lab.append(q[-1] if q else 0)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # a qualifying x_9 is never re-emitted: eval only ever prints the previous one
        for k, i in enumerate(self.valid_positions().tolist()):
            j = i + 1 if i + 1 < INPUT_LEN else i
            if j == i and int(x[i]) >= 5:
                continue
            x2 = x.clone(); x2[j] = 9
            if int(self.expr.eval(x2)[j]) != int(lab[k]):
                raise ProbeError(f'hold2_cur: forced qualifier does not demote the claimed register at {i} on {x.tolist()}')
class Sjf2M2(ProbeVariable):
    """sjf2_m2 on sjf2_id (integrated)."""
    variable_id = 'sjf2_m2'
    algorithm = 'sjf2_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            room, lab = [], []
            for v in r:
                room.append(v)
                if len(room) >= 3:
                    room.sort(); room.pop(0)
                lab.append(min(room) if len(room) == 2 else 0)
            out.append(lab[1:])
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # the larger occupant only surfaces in a later eviction, so position 9 is unobservable
        y, pos = self.expr.eval(x), self.valid_positions().tolist()
        for k, i in enumerate(pos):
            if i + 1 < INPUT_LEN:   # a forced 9 arriving next must evict exactly the claimed room minimum
                x2 = x.clone(); x2[i + 1] = 9
                if int(self.expr.eval(x2)[i + 1]) != int(lab[k]):
                    raise ProbeError(f'sjf2_m2: forced-9 eviction != claimed room minimum at {i} on {x.tolist()}')
            elif k and pos[k - 1] == i - 1 and int(y[i]) == int(x[i]) and int(lab[k]) != int(lab[k - 1]):
                raise ProbeError(f'sjf2_m2: room minimum moved though eval evicted the arrival at {i} on {x.tolist()}')
class Gapge5Prevpos(ProbeVariable):
    """gapge5_prevpos on gapge5_id (integrated)."""
    variable_id = 'gapge5_prevpos'
    algorithm = 'gapge5_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            last, lab = 0, []
            for i, v in enumerate(r):
                if v >= 5: last = i
                lab.append(last)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # the pointer is 0 at position 0 either way
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y, pos = self.expr.eval(x).tolist(), self.valid_positions().tolist()
        for k, i in enumerate(pos):
            p = int(lab[k])
            q = int(lab[k - 1]) if k and pos[k - 1] == i - 1 else (0 if i == 1 else p)
            x2 = torch.zeros(INPUT_LEN, dtype=torch.long); x2[i - 1], x2[i] = 9, int(x[i])
            if p != (i if int(self.expr.eval(x2)[i]) == 1 else q):   # zeroed prefix, 9 at i-1: gap 1 iff x_i qualifies
                raise ProbeError(f'gapge5_prevpos: forced-gap probe != claimed pointer step at {i} on {x.tolist()}')
            if not k or pos[k - 1] != i - 1:
                continue
            if (p == q and y[i] != y[i - 1]) or (p != q and y[i] != (p - q if q or int(x[0]) >= 5 else 0)):
                raise ProbeError(f'gapge5_prevpos: gap != claimed pointer jump at {i} on {x.tolist()}')
class Pmaxv89Shadow(ProbeVariable):
    """pmaxv89_shadow on pmaxv89_id (integrated)."""
    variable_id = 'pmaxv89_shadow'
    algorithm = 'pmaxv89_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cummax(inputs, dim=1).values.long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # the reference register skips veto digits, so it never forms the raw max at all
        y = self.expr.eval(x)   # below the veto floor the raw max IS the vetoed max eval reports
        prev = 0
        for k, i in enumerate(self.valid_positions().tolist()):
            m = int(lab[k])
            if m < max(prev, int(x[i]), int(y[i])) or (m < 8 and m != int(y[i])):
                raise ProbeError(f'pmaxv89_shadow: raw max inconsistent with vetoed max at {i} on {x.tolist()}')
            prev = m
class Pmaxv789Shadow(ProbeVariable):
    """pmaxv789_shadow on pmaxv789_id (integrated)."""
    variable_id = 'pmaxv789_shadow'
    algorithm = 'pmaxv789_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cummax(inputs, dim=1).values.long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # the reference register skips veto digits, so it never forms the raw max at all
        y = self.expr.eval(x)   # below the veto floor the raw max IS the vetoed max eval reports
        prev = 0
        for k, i in enumerate(self.valid_positions().tolist()):
            m = int(lab[k])
            if m < max(prev, int(x[i]), int(y[i])) or (m < 7 and m != int(y[i])):
                raise ProbeError(f'pmaxv789_shadow: raw max inconsistent with vetoed max at {i} on {x.tolist()}')
            prev = m
class TgSel(ProbeVariable):
    """tg_sel on togsum0_id (integrated)."""
    variable_id = 'tg_sel'
    algorithm = 'togsum0_id'
    n_classes = 2
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (torch.cumsum((inputs == 0).long(), dim=1) % 2).long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # surgery: writing 1 vs 2 at i moves the printed lane by +1 (odd lane) or -2 (even lane)
        for k, i in enumerate(self.valid_positions().tolist()):
            z1, z2 = x.clone(), x.clone(); z1[i], z2[i] = 1, 2
            shift = (int(self.expr.eval(z1)[i]) - int(self.expr.eval(z2)[i])) % 10
            sel = int(lab[k]) ^ (int(x[i]) == 0)
            if shift != (1 if sel else 8):
                raise ProbeError(f'tg_sel: probed lane at {i} contradicts claimed selector on {x.tolist()}')


class SsLatch(ProbeVariable):
    """ss_latch on signsum0_id (integrated)."""
    variable_id = 'ss_latch'
    algorithm = 'signsum0_id'
    n_classes = 2
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (torch.cumsum((inputs == 0).long(), dim=1) % 2 == 0).long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # surgery: a 1 written at i moves the running sum by +1 or -1, revealing the sign latch
        y = self.expr.eval(x).tolist()
        for k, i in enumerate(self.valid_positions().tolist()):
            z = x.clone(); z[i] = 1
            step = (int(self.expr.eval(z)[i]) - (y[i - 1] if i else 0)) % 10
            pos = int(lab[k]) ^ (int(x[i]) == 0)
            if step != (1 if pos else 9):
                raise ProbeError(f'ss_latch: probed step at {i} contradicts claimed latch on {x.tolist()}')


class LeakyReg(ProbeVariable):
    """leaky_reg on ewma2_id (integrated)."""
    variable_id = 'leaky_reg'
    algorithm = 'ewma2_id'
    n_classes = 5
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            prev, lab = 0, []
            for v in r:
                cur = (v + prev // 2) % 10
                lab.append(cur // 2); prev = cur
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        for i in range(10):
            if int(lab[i]) != int(y[i]) // 2:
                raise ProbeError(f'leaky_reg: label != floor(eval/2) at {i} on {x.tolist()}')
            prev = 0 if i == 0 else int(y[i - 1])
            if int(y[i]) != (int(x[i]) + prev // 2) % 10:
                raise ProbeError(f'leaky_reg: eval recurrence broken at {i} on {x.tolist()}')


class ZcSide(ProbeVariable):
    """zc_side on zcross5_id (integrated)."""
    variable_id = 'zc_side'
    algorithm = 'zcross5_id'
    n_classes = 2
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs >= 5).long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        if not torch.equal(lab, (x >= 5).long()):
            raise ProbeError(f'zc_side: label != which-side bit on {x.tolist()}')
        y = self.expr.eval(x); c = 0
        for i in range(10):
            if i > 0 and int(lab[i]) != int(lab[i - 1]): c += 1
            if int(y[i]) != c % 10:
                raise ProbeError(f'zc_side: eval != crossing count at {i} on {x.tolist()}')


class AltrunCur(ProbeVariable):
    """altrun_cur on altrun_id (integrated)."""
    variable_id = 'altrun_cur'
    algorithm = 'altrun_id'
    n_classes = 11
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            cur, lastdir, lab = 1, 0, []
            for i in range(INPUT_LEN):
                if i == 0:
                    cur, lastdir = 1, 0
                else:
                    d = r[i] - r[i - 1]; s = (d > 0) - (d < 0)
                    if s == 0: cur, lastdir = 1, 0
                    elif lastdir == 0 or s == lastdir: cur, lastdir = 2, s
                    else: cur, lastdir = cur + 1, s
                lab.append(cur)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # cur(0) = 1, a run of one
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y, best = self.expr.eval(x), 1   # flatten-prefix surgery: eval then reports the claimed run at i itself
        for k, i in enumerate(self.valid_positions().tolist()):
            c = int(lab[k])
            best = max(best, c)
            if int(y[i]) != best or not 1 <= c <= i + 1:
                raise ProbeError(f'altrun_cur: claimed run breaks the eval prefix-max at {i} on {x.tolist()}')
            for n in (c, c + 1):
                if i - n + 1 < 0: continue
                z = x.clone(); z[:i - n + 1] = x[i - n + 1]
                if (int(self.expr.eval(z)[i]) == n) != (n == c):
                    raise ProbeError(f'altrun_cur: alternating run at {i} is not exactly {c} on {x.tolist()}')
class SchmittState(ProbeVariable):
    """schmitt_state on schmitt_id (integrated)."""
    variable_id = 'schmitt_state'
    algorithm = 'schmitt_id'
    n_classes = 2
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            s, lab = 1, []
            for v in r:
                if v <= 3: s = 0
                elif v >= 6: s = 1
                lab.append(s)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        for i in range(10):
            if int(y[i]) != 9 * int(lab[i]):
                raise ProbeError(f'schmitt_state: eval != 9*latch at {i} on {x.tolist()}')


class Qlen3Reg(ProbeVariable):
    """qlen3_reg on qlen3_id (integrated)."""
    variable_id = 'qlen3_reg'
    algorithm = 'qlen3_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            q, lab = 0, []
            for v in r:
                q = max(0, min(9, q + (1 if v >= 3 else -1)))
                lab.append(q)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        if not torch.equal(lab, self.expr.eval(x)):
            raise ProbeError(f'qlen3_reg: register != eval (queue length IS the output) on {x.tolist()}')


class Qlen4Reg(ProbeVariable):
    """qlen4_reg on qlen4_id (integrated)."""
    variable_id = 'qlen4_reg'
    algorithm = 'qlen4_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            q, lab = 0, []
            for v in r:
                q = max(0, min(9, q + (1 if v >= 4 else -1)))
                lab.append(q)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        if not torch.equal(lab, self.expr.eval(x)):
            raise ProbeError(f'qlen4_reg: register != eval (queue length IS the output) on {x.tolist()}')


class Qlen5Reg(ProbeVariable):
    """qlen5_reg on qlen5_id (integrated)."""
    variable_id = 'qlen5_reg'
    algorithm = 'qlen5_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            q, lab = 0, []
            for v in r:
                q = max(0, min(9, q + (1 if v >= 5 else -1)))
                lab.append(q)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        if not torch.equal(lab, self.expr.eval(x)):
            raise ProbeError(f'qlen5_reg: register != eval (queue length IS the output) on {x.tolist()}')


class Qlen6Reg(ProbeVariable):
    """qlen6_reg on qlen6_id (integrated)."""
    variable_id = 'qlen6_reg'
    algorithm = 'qlen6_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            q, lab = 0, []
            for v in r:
                q = max(0, min(9, q + (1 if v >= 6 else -1)))
                lab.append(q)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        if not torch.equal(lab, self.expr.eval(x)):
            raise ProbeError(f'qlen6_reg: register != eval (queue length IS the output) on {x.tolist()}')


class BpathSkipbit(ProbeVariable):
    """bpath_skipbit on bpath_id (integrated)."""
    variable_id = 'bpath_skipbit'
    algorithm = 'bpath_id'
    n_classes = 2
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            b, lab = [], []
            for i in range(10):
                if i == 0: b.append(r[0]); lab.append(0)
                elif i == 1: b.append(max(r[1], b[0])); lab.append(0)
                else:
                    lab.append(0 if b[i - 1] <= b[i - 2] else 1)
                    b.append(max(r[i], min(b[i - 1], b[i - 2])))
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(2, INPUT_LEN)   # no branch exists before two predecessors do
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x).tolist()
        for k, i in enumerate(self.valid_positions().tolist()):
            b = int(lab[k])
            if i < 2:
                if b:
                    raise ProbeError(f'bpath_skipbit: no branch exists at {i} on {x.tolist()}')
            # the claimed branch must both replay the bottleneck recurrence on eval and name eval's min predecessor
            elif y[i] != max(int(x[i]), y[i - 1 - b]) or b != (0 if y[i - 1] <= y[i - 2] else 1):
                raise ProbeError(f'bpath_skipbit: claimed branch breaks the eval bottleneck recurrence at {i} on {x.tolist()}')


class IncrunCur(ProbeVariable):
    """incrun_cur on incrun_id (integrated)."""
    variable_id = 'incrun_cur'
    algorithm = 'incrun_id'
    n_classes = 11
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            cur, lab = 1, []
            for i in range(INPUT_LEN):
                if i > 0 and r[i] > r[i - 1]: cur += 1
                elif i > 0: cur = 1
                lab.append(cur)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # cur(0) = 1, a run of one
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y, best = self.expr.eval(x), 1   # flatten-prefix surgery: a 9 filler resets the run, so eval reports it at i
        for k, i in enumerate(self.valid_positions().tolist()):
            c = int(lab[k])
            best = max(best, c)
            if int(y[i]) != best or not 1 <= c <= i + 1:
                raise ProbeError(f'incrun_cur: claimed run breaks the eval prefix-max at {i} on {x.tolist()}')
            for n in (c, c + 1):
                if i - n + 1 < 0: continue
                z = x.clone(); z[:i - n + 1] = 9
                if (int(self.expr.eval(z)[i]) == n) != (n == c):
                    raise ProbeError(f'incrun_cur: increasing run at {i} is not exactly {c} on {x.tolist()}')
class MadMean(ProbeVariable):
    """mad_mean on runmad_id (integrated)."""
    variable_id = 'mad_mean'
    algorithm = 'runmad_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            s, lab = 0, []
            for i, v in enumerate(r):
                s += v; lab.append(math.floor(s / (i + 1) + 0.5))
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x); s = 0
        for i in range(10):
            s += int(x[i])
            if int(lab[i]) != math.floor(s / (i + 1) + 0.5):
                raise ProbeError(f'mad_mean: label != rounded running mean at {i} on {x.tolist()}')
            if int(y[i]) != abs(int(x[i]) - int(lab[i])):
                raise ProbeError(f'mad_mean: eval != |x - mean| at {i} on {x.tolist()}')


class CfPrev(ProbeVariable):
    """cf_prev on contfrac_id (integrated)."""
    variable_id = 'cf_prev'
    algorithm = 'contfrac_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            qm2, qm1, lab = 0, 1, []
            for a in r:
                lab.append(qm1)
                q = (a * qm1 + qm2) % 10; qm2, qm1 = qm1, q
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # the seed convergent 1 is constant at position 0
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # independence: reference convergents q (initials 0, 1) via q_i = a_i*q_{i-1} + q_{i-2}
        y = self.expr.eval(x); q = [0, 1] + y.tolist()
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(lab[k]) != q[i + 1]:
                raise ProbeError(f'cf_prev: lab != preceding reference convergent at {i} on {x.tolist()}')
            if int(y[i]) != (int(x[i]) * int(lab[k]) + q[i]) % 10:
                raise ProbeError(f'cf_prev: eval != a*lab + q_i-2 at {i} on {x.tolist()}')


class D4charElement(ProbeVariable):
    """d4char_element on d4char_id (integrated)."""
    variable_id = 'd4char_element'
    algorithm = 'd4char_id'
    n_classes = 8
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        E = [(0, 1, 2, 3), (1, 2, 3, 0), (0, 3, 2, 1), (3, 0, 1, 2), (2, 1, 0, 3), (3, 2, 1, 0), (2, 3, 0, 1), (1, 0, 3, 2)]
        idx = {e: i for i, e in enumerate(E)}
        out = []
        for r in inputs.tolist():
            cur = (0, 1, 2, 3); lab = []
            for d in r:
                e = E[d % 8]; cur = tuple(e[cur[k]] for k in range(4)); lab.append(idx[cur])
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        E = [(0, 1, 2, 3), (1, 2, 3, 0), (0, 3, 2, 1), (3, 0, 1, 2), (2, 1, 0, 3), (3, 2, 1, 0), (2, 3, 0, 1), (1, 0, 3, 2)]
        CHI = [2, 0, 0, 0, 0, 0, 8, 0]
        y = self.expr.eval(x); cur = (0, 1, 2, 3)
        for i, d in enumerate(x.tolist()):
            e = E[d % 8]; cur = tuple(e[cur[k]] for k in range(4))
            if E.index(cur) != int(lab[i]):
                raise ProbeError(f'd4char_element: element index != label at {i} on {x.tolist()}')
            if CHI[int(lab[i])] != int(y[i]):
                raise ProbeError(f'd4char_element: character(element) != eval at {i} on {x.tolist()}')


class RangeaddRange(ProbeVariable):
    """rangeadd_range on rangeadd_id (integrated)."""
    variable_id = 'rangeadd_range'
    algorithm = 'rangeadd_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        hi = torch.cummax(inputs, dim=1).values; lo = torch.cummin(inputs, dim=1).values
        return (hi - lo).long()[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # spread(0) = 0, one element
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y, pos = self.expr.eval(x), self.valid_positions()
        hi = torch.cummax(x, 0).values; lo = torch.cummin(x, 0).values
        if not torch.equal(lab, (hi - lo)[pos]):
            raise ProbeError(f'rangeadd_range: label != spread on {x.tolist()}')
        for k, i in enumerate(pos.tolist()):
            if int(y[i]) != (int(x[i]) + int(lab[k])) % 10:
                raise ProbeError(f'rangeadd_range: (x + spread) % 10 != eval at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = self.expr.eval(x).clone(); y[p] = (int(x[p]) + int(new_class)) % 10
        return y


class MedaddMedian(ProbeVariable):
    """medadd_median on medadd_id (integrated)."""
    variable_id = 'medadd_median'
    algorithm = 'medadd_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            lab = [sorted(r[:i + 1])[i // 2] for i in range(10)]
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x); r = x.tolist()
        for i in range(10):
            if int(lab[i]) != sorted(r[:i + 1])[i // 2]:
                raise ProbeError(f'medadd_median: label != running median at {i} on {x.tolist()}')
            if int(y[i]) != (r[i] + int(lab[i])) % 10:
                raise ProbeError(f'medadd_median: (x + median) % 10 != eval at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = self.expr.eval(x).clone(); y[p] = (int(x[p]) + int(new_class)) % 10
        return y


class TribscanLag1(ProbeVariable):
    """tribscan_lag1 on tribscan_id (integrated)."""
    variable_id = 'tribscan_lag1'
    algorithm = 'tribscan_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            a = b = c = 0; lab = []
            for v in r:
                lab.append(a); q = (v + a + b + c) % 10; a, b, c = q, a, b
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # the zero initial condition is constant at position 0
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # independence: reference totals q (zero initials) via q_i = x_i + q_{i-1} + q_{i-2} + q_{i-3}
        y = self.expr.eval(x); q = [0, 0, 0] + y.tolist()
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(lab[k]) != q[i + 2]:
                raise ProbeError(f'tribscan_lag1: lab != preceding reference total at {i} on {x.tolist()}')
            if int(y[i]) != (int(x[i]) + int(lab[k]) + q[i + 1] + q[i]) % 10:
                raise ProbeError(f'tribscan_lag1: eval != x + lab + two earlier totals at {i} on {x.tolist()}')


class Rotor3W(ProbeVariable):
    """rotor3_w on rotor3_id (integrated)."""
    variable_id = 'rotor3_w'
    algorithm = 'rotor3_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            u = v = w = 0; lab = []
            for d in r:
                u = (u + w + d) % 10; v = (v + u) % 10; w = (w + v) % 10; lab.append(w)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # independence: w is the double prefix sum of the reference u output
        y = self.expr.eval(x)
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(lab[k]) != sum((i - j + 1) * int(y[j]) for j in range(i + 1)) % 10:
                raise ProbeError(f'rotor3_w: lab != double prefix sum of eval at {i} on {x.tolist()}')


class ModecntaddModecount(ProbeVariable):
    """modecntadd_modecount on modecntadd_id (integrated)."""
    variable_id = 'modecntadd_modecount'
    algorithm = 'modecntadd_id'
    n_classes = 11
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            seen = Counter(); lab = []
            for v in r:
                seen[v] += 1; lab.append(max(seen.values()))
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # the mode count is 1 after one token
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y, seen, counts = self.expr.eval(x), Counter(), []
        for v in x.tolist():
            seen[v] += 1; counts.append(max(seen.values()))
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(lab[k]) != counts[i]:
                raise ProbeError(f'modecntadd_modecount: label != mode count at {i} on {x.tolist()}')
            if int(y[i]) != (int(x[i]) + int(lab[k])) % 10:
                raise ProbeError(f'modecntadd_modecount: (x + modecount) % 10 != eval at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = self.expr.eval(x).clone(); y[p] = (int(x[p]) + int(new_class)) % 10
        return y


class OscposMomentum(ProbeVariable):
    """oscpos_momentum on oscpos_id (integrated)."""
    variable_id = 'oscpos_momentum'
    algorithm = 'oscpos_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            th = p = 0; lab = []
            for v in r:
                th = (th + p) % 10; p = (p - th + v) % 10; lab.append(p)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # independence: momentum is the prefix sum of (x - reference position), and th_{i+1} = th_i + p_i
        y = self.expr.eval(x)
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(lab[k]) != int((x[:i + 1] - y[:i + 1]).sum()) % 10:
                raise ProbeError(f'oscpos_momentum: lab != prefix sum of (x - eval) at {i} on {x.tolist()}')
            if i + 1 < y.numel() and int(y[i + 1]) != (int(y[i]) + int(lab[k])) % 10:
                raise ProbeError(f'oscpos_momentum: eval step != eval + lab at {i} on {x.tolist()}')


class OscmomPosition(ProbeVariable):
    """oscmom_position on oscmom_id (integrated)."""
    variable_id = 'oscmom_position'
    algorithm = 'oscmom_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            th = p = 0; lab = []
            for v in r:
                th = (th + p) % 10; p = (p - th + v) % 10; lab.append(th)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # the position starts at 0 before any momentum accrues
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # independence: position is the strict prefix sum of the reference momentum output
        y = self.expr.eval(x)
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(lab[k]) != int(y[:i].sum()) % 10:
                raise ProbeError(f'oscmom_position: lab != prefix sum of earlier eval at {i} on {x.tolist()}')
            if int(y[i]) != ((int(y[i - 1]) if i else 0) - int(lab[k]) + int(x[i])) % 10:
                raise ProbeError(f'oscmom_position: eval != prev eval - lab + x at {i} on {x.tolist()}')


class RefracInhib(ProbeVariable):
    """refrac_inhib on refrac_id (integrated)."""
    variable_id = 'refrac_inhib'
    algorithm = 'refrac_id'
    n_classes = 6
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            last = None; lab = []
            for i, v in enumerate(r):
                th = max(0, 5 - (i - last)) if last is not None else 0
                lab.append(th)
                if v >= 7: last = i
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # no spike can precede position 0, so the threshold is 0 there
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # thresholds 0 and 1 differ only at v=0, where silenced and printed are both 0
        for k, i in enumerate(self.valid_positions().tolist()):
            c = max(int(lab[k]), 1)   # eval prints d at i iff d >= threshold
            for d, want in (((c, c), (c - 1, 0)) if c > 1 else ((c, c),)):
                z = x.clone(); z[i] = d
                if int(self.expr.eval(z)[i]) != want:
                    raise ProbeError(f'refrac_inhib: probe {d} at {i} contradicts claimed threshold on {x.tolist()}')
class FibscanLag1(ProbeVariable):
    """fibscan_lag1 on fibscan_id (integrated)."""
    variable_id = 'fibscan_lag1'
    algorithm = 'fibscan_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            a = b = 0; lab = []
            for v in r:
                lab.append(a); q = (v + a + b) % 10; a, b = q, a
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # the zero initial condition is constant at position 0
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # independence: reference totals q (zero initials) via q_i = x_i + q_{i-1} + q_{i-2}
        y = self.expr.eval(x); q = [0, 0] + y.tolist()
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(lab[k]) != q[i + 1]:
                raise ProbeError(f'fibscan_lag1: lab != preceding reference total at {i} on {x.tolist()}')
            if int(y[i]) != (int(x[i]) + int(lab[k]) + q[i]) % 10:
                raise ProbeError(f'fibscan_lag1: eval != x + lab + earlier total at {i} on {x.tolist()}')


class InvcountSign(ProbeVariable):
    """invcount_sign on invcount_id (integrated)."""
    variable_id = 'invcount_sign'
    algorithm = 'invcount_id'
    n_classes = 2
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            c = 0; lab = []
            for i in range(INPUT_LEN):
                c += sum(1 for j in range(i) if r[j] > r[i]); lab.append(c % 2)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # no pair exists at position 0, so the parity is 0
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # independence: eval is the inversion count mod 10, so its parity must be the label
        y = self.expr.eval(x)
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(lab[k]) != int(y[i]) % 2:
                raise ProbeError(f'invcount_sign: lab != parity of eval at {i} on {x.tolist()}')


class GraviclumpMean(ProbeVariable):
    """graviclump_mean on graviclump_id (integrated)."""
    variable_id = 'graviclump_mean'
    algorithm = 'graviclump_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        m = torch.round(inputs.float().mean(dim=1)).long()
        return m.unsqueeze(1).expand(-1, INPUT_LEN)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([INPUT_LEN - 1], dtype=torch.long)   # the mean needs every digit
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # eval sees the mean only via sign(mean - x_i): any mean in the same digit gap reads alike
        y = self.expr.eval(x)
        for m in lab.tolist():
            for i in range(INPUT_LEN):
                if int(y[i]) - int(x[i]) != (m > int(x[i])) - (m < int(x[i])):
                    raise ProbeError(f'graviclump_mean: eval step != sign(lab - x) at {i} on {x.tolist()}')
class E2symRunsum(ProbeVariable):
    """e2sym_runsum on e2sym_id (integrated)."""
    variable_id = 'e2sym_runsum'
    algorithm = 'e2sym_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (torch.cumsum(inputs, dim=1) % 10).long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # s1(9) feeds only the never-emitted e2(10), so no successor digit can expose it
        y = self.expr.eval(x)
        for k, i in enumerate(self.valid_positions().tolist()):
            s, p2 = int(lab[k]), int((x[:i + 1] ** 2).sum())
            if 2 * int(y[i]) % 10 != (s ** 2 - p2) % 10:
                raise ProbeError(f'e2sym_runsum: 2*eval != lab^2 - sum of squares at {i} on {x.tolist()}')
            if int(y[i]) != ((int(y[i - 1]) if i else 0) + int(x[i]) * (s - int(x[i]))) % 10:
                raise ProbeError(f'e2sym_runsum: eval increment != x*(lab - x) at {i} on {x.tolist()}')
            if i + 1 >= INPUT_LEN: continue
            z = x.clone(); z[i + 1] = 1   # surgery: a forced 1 next emits the running sum as the whole e2 increment
            inc, probed = (int(y[i + 1]) - int(y[i])) % 10, int(self.expr.eval(z)[i + 1])
            if inc != int(x[i + 1]) * s % 10 or probed != (int(y[i]) + s) % 10:
                raise ProbeError(f'e2sym_runsum: next / forced-1 eval increment != x*lab at {i} on {x.tolist()}')
class RunmexFrontier(ProbeVariable):
    """runmex_frontier on runmex_id (integrated)."""
    variable_id = 'runmex_frontier'
    algorithm = 'runmex_id'
    n_classes = 11
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            seen = set(); lab = []
            for v in r:
                seen.add(v); m = 0
                while m in seen: m += 1
                lab.append(m)
            out.append(lab)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y, prev = self.expr.eval(x).tolist(), 0
        for k, i in enumerate(self.valid_positions().tolist()):
            m = int(lab[k])
            if y[i] != min(m, 9) or m < prev:
                raise ProbeError(f'runmex_frontier: mex {m} at {i} contradicts clamped eval on {x.tolist()}')
            prev = m


class PalmaxSide(ProbeVariable):
    """palmax_side on palmax_id (integrated)."""
    variable_id = 'palmax_side'
    algorithm = 'palmax_id'
    n_classes = 2
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        side = (inputs[:, :5] >= inputs[:, 5:].flip(1)).long()   # 1[first-half member of the pair wins]
        return torch.cat([side, side.flip(1)], dim=1)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(5, INPUT_LEN)   # pair (i, 9-i) is only settled at the later member
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        for c, i in enumerate(self.valid_positions().tolist()):
            m = INPUT_LEN - 1 - i
            won = int(x[m]) if int(lab[c]) else int(x[i])
            if won != int(y[i]) or won != int(y[m]):
                raise ProbeError(f'palmax_side: routing bit at {i} does not reconstruct eval on {x.tolist()}')
            x2 = x.clone(); x2[i] = 0   # zeroing the late member reads the early member out of the reference
            if not int(lab[c]) and int(y[i]) == int(self.expr.eval(x2)[i]):
                raise ProbeError(f'palmax_side: late-member win at {i} contradicts a tie in eval on {x.tolist()}')
class RankaddRank(ProbeVariable):
    """rankadd_rank on rankadd_id (integrated)."""
    variable_id = 'rankadd_rank'
    algorithm = 'rankadd_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            out.append([sum(1 for j in range(i) if r[j] < r[i]) % 10 for i in range(1, 10)])
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)   # joint route: the reference output is (x + claimed rank) % 10
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(y[i]) != (int(x[i]) + int(lab[k])) % 10:
                raise ProbeError(f'rankadd_rank: (x + lab) % 10 != eval at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = self.expr.eval(x).clone()
        y[p] = (int(x[p]) + int(new_class)) % 10
        return y


class SigndiffCurv(ProbeVariable):
    """signdiff_curv on signdiff_id (integrated)."""
    variable_id = 'signdiff_curv'
    algorithm = 'signdiff_id'
    n_classes = 3
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        left = torch.cat([inputs[:, :1], inputs[:, :-1]], dim=1)
        right = torch.cat([inputs[:, 1:], inputs[:, -1:]], dim=1)
        lap = left + right - 2 * inputs
        return ((lap > 0).long() - (lap < 0).long() + 1).long()[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([INPUT_LEN - 1], dtype=torch.long)   # the laplacian reads x[i+1]
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # the 0/9 clamp merges two of the three signs at the extremes
        y = self.expr.eval(x)
        for c, i in enumerate(self.valid_positions().tolist()):
            if int(y[i]) != min(9, max(0, int(x[i]) + int(lab[c]) - 1)):
                raise ProbeError(f'signdiff_curv: sign-register reconstruction != eval at {i} on {x.tolist()}')


class MidaddMid(ProbeVariable):
    """midadd_mid on midadd_id (integrated)."""
    variable_id = 'midadd_mid'
    algorithm = 'midadd_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        hi = torch.cummax(inputs, dim=1).values; lo = torch.cummin(inputs, dim=1).values
        return (((hi + lo) // 2) % 10)[:, 1:].long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)   # joint route: the reference output is (x + claimed midrange) % 10
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(y[i]) != (int(x[i]) + int(lab[k])) % 10:
                raise ProbeError(f'midadd_mid: (x + lab) % 10 != eval at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = self.expr.eval(x).clone(); y[p] = (int(x[p]) + int(new_class)) % 10
        return y


class MidaddMax(ProbeVariable):
    """midadd_max on midadd_id (integrated)."""
    variable_id = 'midadd_max'
    algorithm = 'midadd_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (torch.cummax(inputs, dim=1).values % 10)[:, 1:].long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # a running min of 0 stays 0, and floor((mx+0)/2) cannot split even mx from mx+1
        y, lo, prev = self.expr.eval(x), torch.cummin(x, 0).values, 0
        for k, i in enumerate(self.valid_positions().tolist()):
            m, t = int(lab[k]), int(lo[i]) - 1
            if m < max(prev, int(x[i])) or int(y[i]) != (int(x[i]) + (m + int(lo[i])) // 2) % 10:
                raise ProbeError(f'midadd_max: claimed max breaks the running max or eval midrange at {i} on {x.tolist()}')
            if t >= 0 and i + 1 < INPUT_LEN:
                z = x.clone(); z[i + 1] = t
                if int(self.expr.eval(z)[i + 1]) != (t + (m + t) // 2) % 10:
                    raise ProbeError(f'midadd_max: parity probe at {i + 1} contradicts claimed max on {x.tolist()}')
            prev = m
class MidaddMin(ProbeVariable):
    """midadd_min on midadd_id (integrated)."""
    variable_id = 'midadd_min'
    algorithm = 'midadd_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return (torch.cummin(inputs, dim=1).values % 10)[:, 1:].long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # a running max of 9 stays 9, and floor((9+mn)/2) cannot split mn from mn+1
        y, hi, prev = self.expr.eval(x), torch.cummax(x, 0).values, 9
        for k, i in enumerate(self.valid_positions().tolist()):
            m, t = int(lab[k]), int(hi[i]) + 1
            if m > min(prev, int(x[i])) or int(y[i]) != (int(x[i]) + (int(hi[i]) + m) // 2) % 10:
                raise ProbeError(f'midadd_min: claimed min breaks the running min or eval midrange at {i} on {x.tolist()}')
            if t <= 9 and i + 1 < INPUT_LEN:
                z = x.clone(); z[i + 1] = t
                if int(self.expr.eval(z)[i + 1]) != (t + (t + m) // 2) % 10:
                    raise ProbeError(f'midadd_min: parity probe at {i + 1} contradicts claimed min on {x.tolist()}')
            prev = m
class ArgposaddArgpos(ProbeVariable):
    """argposadd_argpos on argposadd_id (integrated)."""
    variable_id = 'argposadd_argpos'
    algorithm = 'argposadd_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            m = -1; ap = 0; row = []
            for i, d in enumerate(r):
                if d > m: m = d; ap = i
                if i >= 1: row.append(ap)
            out.append(row)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)   # joint route: the reference output is (x + claimed argmax position) % 10
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(y[i]) != (int(x[i]) + int(lab[k])) % 10:
                raise ProbeError(f'argposadd_argpos: (x + lab) % 10 != eval at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = self.expr.eval(x).clone()
        y[p] = (int(x[p]) + int(new_class)) % 10
        return y


class LanemaxRegs(ProbeVariable):
    """lanemax_regs on lanemax_id (integrated)."""
    variable_id = 'lanemax_regs'
    algorithm = 'lanemax_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            m = [-1, -1]; row = []
            for i, d in enumerate(r):
                if i >= 1:
                    row.append(m[1 - i % 2])
                m[i % 2] = max(m[i % 2], d)
            out.append(row)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x); vals = x.tolist()
        for k, i in enumerate(range(1, INPUT_LEN)):
            opp = max(vals[j] for j in range(i) if j % 2 != i % 2)
            if int(lab[k]) != opp or opp != int(y[i - 1]):
                raise ProbeError(f'lanemax_regs: opp-lane max != y[i-1] on {vals}')


class PancakeRunargmax(ProbeVariable):
    """pancake_runargmax on pancake_id (integrated)."""
    variable_id = 'pancake_runargmax'
    algorithm = 'pancake_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        B, L = inputs.shape
        out = torch.zeros(B, L, dtype=torch.long)
        idx = torch.arange(L)
        for t in range(L):
            pref = inputs[:, :t + 1]
            m = pref.max(dim=1, keepdim=True).values
            out[:, t] = torch.where(pref == m, idx[:t + 1], torch.full((t + 1,), t + 1)).min(dim=1).values
        return out[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # argmax(0) = 0, one element
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        xs, pos = x.tolist(), self.valid_positions().tolist()
        run, best, bi = [], None, 0
        for i, v in enumerate(xs):
            if best is None or v > best:
                best, bi = v, i
            run.append(bi)
        if [run[i] for i in pos] != lab.tolist():
            raise ProbeError(f'pancake_runargmax: running argmax mismatch on {xs}')
        k = int(lab[pos.index(INPUT_LEN - 1)])   # the flip index is the argmax over the whole input
        rebuilt = xs[:k + 1][::-1] + xs[k + 1:]
        if not torch.equal(torch.tensor(rebuilt, dtype=torch.long), self.expr.eval(x)):
            raise ProbeError(f'pancake_runargmax: flip-index rebuild != eval on {xs}')


class BitmajPlanevote(ProbeVariable):
    """bitmaj_planevote on bitmaj_id (integrated)."""
    variable_id = 'bitmaj_planevote'
    algorithm = 'bitmaj_id'
    n_classes = 2
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        b1 = ((inputs >> 1) & 1).long()
        cnt = torch.cumsum(b1, dim=1)
        n = torch.arange(1, inputs.shape[1] + 1)
        return (cnt * 2 > n).long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x).tolist()
        for k, i in enumerate(self.valid_positions().tolist()):
            hi = sum((int(d) >> 3) & 1 for d in x[:i + 1])
            # invert eval's mod 10: bit 3's majority says whether the word wrapped, then bit 1 is the claimed vote
            word = y[i] + 10 if hi * 2 > i + 1 and y[i] < 8 else y[i]
            if (word >> 1) & 1 != int(lab[k]):
                raise ProbeError(f'bitmaj_planevote: bit 1 of the reference word != claimed vote at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y = self.expr.eval(x).clone(); cnt = [0, 0, 0, 0]
        for i in range(p + 1):
            for b in range(4): cnt[b] += (int(x[i]) >> b) & 1
        maj = [1 if cnt[b] * 2 > p + 1 else 0 for b in range(4)]
        maj[1] = int(new_class)
        y[p] = sum(maj[b] << b for b in range(4)) % 10
        return y


class Skipn5Guard(ProbeVariable):
    """skipn5_guard on skipn5_id (integrated)."""
    variable_id = 'skipn5_guard'
    algorithm = 'skipn5_id'
    n_classes = 2
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            skip = False; row = []
            for d in r:
                row.append(int(skip))
                skip = (not skip) and (d >= 5)
            out.append(row[1:])
        return torch.tensor(out)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # surgery: a 1 written at i is printed iff the guard is unarmed there (the arming prefix is untouched)
        for k, i in enumerate(self.valid_positions().tolist()):
            z = x.clone(); z[i] = 1
            if int(self.expr.eval(z)[i]) != (0 if int(lab[k]) else 1):
                raise ProbeError(f'skipn5_guard: probe at {i} contradicts claimed armed bit on {x.tolist()}')


class BittvHamming(ProbeVariable):
    """bittv_hamming on bittv_id (integrated)."""
    variable_id = 'bittv_hamming'
    algorithm = 'bittv_id'
    n_classes = 5
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            out.append([bin(r[i] ^ r[i - 1]).count('1') for i in range(1, 10)])
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x).tolist()   # the running sum's increment IS the claimed edge distance
        for k, i in enumerate(self.valid_positions().tolist()):
            if i and (y[i] - y[i - 1]) % 10 != int(lab[k]):
                raise ProbeError(f'bittv_hamming: eval increment != claimed Hamming distance at {i} on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        if p == 0: return None
        r = x.tolist()
        d = [0] + [bin(r[i] ^ r[i - 1]).count('1') for i in range(1, 10)]
        d[p] = int(new_class)
        y = []; s = 0
        for i in range(10):
            s = (s + d[i]) % 10; y.append(s)
        return torch.tensor(y)


class TourneyR1win(ProbeVariable):
    """tourney_r1win on tourney_id (integrated)."""
    variable_id = 'tourney_r1win'
    algorithm = 'tourney_id'
    n_classes = 2
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        win = torch.zeros_like(inputs)
        ew = inputs[:, 0::2] >= inputs[:, 1::2]
        win[:, 0::2] = ew.long()
        win[:, 1::2] = (~ew).long()
        return win.long()[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN, 2)   # pair (2k, 2k+1) is only settled at the odd member
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        for c, i in enumerate(self.valid_positions().tolist()):
            if int(lab[c]) != int(int(y[i]) >= 1):
                raise ProbeError(f'{self.variable_id}: r1win != (depth>=1) at {i} on {x.tolist()}')
            if int(lab[c]) != int(int(x[i]) > int(x[i - 1])):
                raise ProbeError(f'{self.variable_id}: r1win != odd-member win bit at {i} on {x.tolist()}')


class EnvmaxAge(ProbeVariable):
    """envmax_age on envmax_id (integrated)."""
    variable_id = 'envmax_age'
    algorithm = 'envmax_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            m, src, ages = 0, 0, []
            for i, d in enumerate(r):
                if d >= m - 1:
                    src = i
                m = max(d, m - 1, 0)
                ages.append(i - src)
            out.append(ages[1:])
        return torch.tensor(out)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # the reference state is the envelope value alone; d == m-1 ties share one decay line
        y, v = self.expr.eval(x).tolist(), x.tolist()
        for k, i in enumerate(self.valid_positions().tolist()):
            a = int(lab[k]); src = i - a   # the claimed source must set the envelope and decay to it 1/step
            if src < 0 or y[src] != v[src] or y[i] != v[src] - a:
                raise ProbeError(f'envmax_age: age {a} at {i} does not decay eval from its source on {v}')
class CminCounters(ProbeVariable):
    """cmin_counters on cmin_id (integrated)."""
    variable_id = 'cmin_counters'
    algorithm = 'cmin_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            c = [0, 0, 0, 0]; row = []
            for d in r:
                c[d % 4] += 1; c[(d // 4) % 4] += 1
                row.append(c[0] % 10)
            out.append(row)
        return torch.tensor(out)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        # surgery: a 0 written at j hashes to (0, 0), so eval prints counter0 + 2 there
        for k, i in enumerate(self.valid_positions().tolist()):
            j = i + 1 if i + 1 < INPUT_LEN else i
            z = x.clone(); z[j] = 0
            c0 = (int(self.expr.eval(z)[j]) - 2) % 10
            if j == i:
                d = int(x[i]); c0 = (c0 + (d % 4 == 0) + (d // 4 % 4 == 0)) % 10
            if int(lab[k]) != c0:
                raise ProbeError(f'cmin_counters: probed counter0 at {i} != claimed value on {x.tolist()}')


class InternDict(ProbeVariable):
    """intern_dict on intern_id (integrated)."""
    variable_id = 'intern_dict'
    algorithm = 'intern_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            seen, row = set(), []
            for v in r:
                seen.add(v); row.append(len(seen) % 10)
            out.append(row)
        return torch.tensor(out)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.arange(1, INPUT_LEN)   # the dict holds exactly one value after the first token
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x).tolist()   # each distinct input owns a rank, so eval's alphabet size IS the dict size
        for k, i in enumerate(self.valid_positions().tolist()):
            if int(lab[k]) != len(set(y[:i + 1])) % 10:
                raise ProbeError(f'intern_dict: dict size at {i} != distinct ranks emitted by eval on {x.tolist()}')


class DupxorParityvec(ProbeVariable):
    """dupxor_parityvec on dupxor_id (integrated)."""
    variable_id = 'dupxor_parityvec'
    algorithm = 'dupxor_id'
    n_classes = 2
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            c = Counter(r)
            out.append([1 if c[v] % 2 == 1 else 0 for v in r])
        return torch.tensor(out, dtype=torch.long)[:, self.valid_positions()]
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([INPUT_LEN - 1], dtype=torch.long)   # multiplicity needs the whole row
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y, v = self.expr.eval(x).tolist(), x.tolist()
        for k, i in enumerate(self.valid_positions().tolist()):
            if y[i] != (v[i] ^ 1 if int(lab[k]) else v[i]):   # the claimed parity IS eval's LSB flip
                raise ProbeError(f'dupxor_parityvec: claimed parity != LSB flip in eval at {i} on {v}')


class CfreqPrefixCount(ProbeVariable):
    """cfreq_prefix_count on cfreq_id (integrated)."""
    variable_id = 'cfreq_prefix_count'
    algorithm = 'cfreq_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            c, row = Counter(), []
            for v in r:
                row.append(c[v]); c[v] += 1
            out.append(row[1:])
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # cfreq reads only the COUNT MULTISET, so a non-final prefix count leaves no trace in the output
        full = torch.cat([torch.zeros(1, dtype=torch.long), lab]).tolist()
        counts = Counter()
        for i, v in enumerate(x.tolist()):
            counts[v] = max(counts[v], full[i] + 1)
        rank = [sum(counts[u] for u in counts if counts[u] > counts[int(v)]) for v in x]
        if not torch.equal(torch.tensor(rank), self.expr.eval(x)):
            raise ProbeError(f'cfreq_prefix_count: the ranks rebuilt from the labels != eval on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        row = x.tolist(); v = row[p]
        occ = [j for j in range(p) if row[j] == v]
        if new_class > len(occ):
            free = [j for j in range(p) if row[j] != v]
            if len(free) < new_class - len(occ):
                return None                                  # not enough earlier slots to host the extra copies
            for j in free[:new_class - len(occ)]:
                row[j] = v
        elif new_class < len(occ):
            spare = [d for d in range(10) if d not in row]
            if not spare:
                return None                                  # no unused digit to evict the surplus copies to
            for j in occ[:len(occ) - new_class]:
                row[j] = spare[0]
        return self.expr.eval(torch.tensor(row))


class DsubCorr(ProbeVariable):
    """dsub_corr on dsub_id (integrated)."""
    variable_id = 'dsub_corr'
    algorithm = 'dsub_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            d, last, row = [1], {}, []
            for i, v in enumerate(r):
                corr = d[last[v]] if v in last else 0
                row.append(corr % 10)
                last[v] = i
                d.append(2 * d[-1] - corr)
            out.append(row[1:])
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        d, out = 1, []
        for c in torch.cat([torch.zeros(1, dtype=torch.long), lab]).tolist():
            d = (2 * d - c) % 10
            out.append(d)
        if not torch.equal(torch.tensor(out), self.expr.eval(x)):
            raise ProbeError(f'dsub_corr: the mod-10 recurrence replayed from the labels != eval on {x.tolist()}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        d, last, out = [1], {}, []
        for i, v in enumerate(x.tolist()):
            corr = new_class if i == p else (d[last[v]] if v in last else 0)
            last[v] = i
            d.append((2 * d[-1] - corr) % 10)
            out.append(d[-1])
        return torch.tensor(out)


class CodonskipFrame(ProbeVariable):
    """codonskip_frame on codonskip_id (integrated)."""
    variable_id = 'codonskip_frame'
    algorithm = 'codonskip_id'
    n_classes = 3
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        return ((inputs != 0).cumsum(1) % 3).long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y, gap = self.expr.eval(x), x == 0
        if int(y[gap].abs().sum()):
            raise ProbeError(f'codonskip_frame: eval prints a nonzero codon at a gap on {x.tolist()}')
        if not torch.equal(y[~gap], (lab[~gap] - 1) % 3 + 1):
            raise ProbeError(f'codonskip_frame: eval != (frame - 1) % 3 + 1 at the bases on {x.tolist()}')
        _pins_label(self, x, lab)   # eval prints 0 at a gap, so the register is unobservable there
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        y, c = self.expr.eval(x).clone(), int(new_class)
        for i in range(int(p), INPUT_LEN):
            if int(x[i]) == 0:
                continue                        # a gap neither advances the register nor prints it
            if i > int(p):
                c = (c + 1) % 3
            y[i] = (c - 1) % 3 + 1
        return y


class FraglenCurstart(ProbeVariable):
    """fraglen_curstart on fraglen_id (integrated)."""
    variable_id = 'fraglen_curstart'
    algorithm = 'fraglen_id'
    n_classes = 10
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        cut = inputs >= 5
        cut[:, 0] = True                        # position 0 always starts a piece
        idx = torch.arange(inputs.shape[1]).expand_as(inputs)
        return torch.cummax(idx * cut, dim=1).values[:, 1:].long()
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        y = self.expr.eval(x)
        for k, i in enumerate(range(1, INPUT_LEN)):
            s = int(lab[k])
            if s and int(x[s]) < 5:
                raise ProbeError(f'fraglen_curstart: claimed start {s} is not a cut site on {x.tolist()}')
            e = next((j for j in range(i + 1, INPUT_LEN) if int(x[j]) >= 5), INPUT_LEN)
            if int(y[i]) != min(e - s, 9):
                raise ProbeError(f'fraglen_curstart: eval != next-cut minus claimed start at {i} on {x.tolist()}')
        _pins_label(self, x, lab)   # a piece longer than 9 saturates the cap, hiding the start under it
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        p, s = int(p), int(new_class)
        if s > p:
            return None                         # a piece cannot start after the position it contains
        y = self.expr.eval(x).clone()
        for i in range(p, INPUT_LEN):
            if i > p and int(x[i]) >= 5:
                s = i                           # a real cut re-anchors the piece start
            e = next((j for j in range(i + 1, INPUT_LEN) if int(x[j]) >= 5), INPUT_LEN)
            y[i] = min(e - s, 9)
        return y


class FactnormCarry(ProbeVariable):
    """factnorm_carry on factnorm_id (integrated)."""
    variable_id = 'factnorm_carry'
    algorithm = 'factnorm_id'
    n_classes = 5
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            c, row = 0, []
            for i in range(1, 9):
                c = (r[i - 1] + c) // min(i + 1, 10)
                row.append(c)
            out.append(row)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        v, y = x.tolist(), self.expr.eval(x)
        R = [min(i + 2, 10) for i in range(len(v))]
        c = {p: int(lab[k]) for k, p in enumerate(self.valid_positions().tolist())}
        if c[1] != v[0] // R[0]:
            raise ProbeError(f'factnorm_carry: c1 != x0//2 on {v}')
        for i in range(1, 9):
            if int(y[i]) != (v[i] + c[i]) % R[i]:
                raise ProbeError(f'factnorm_carry: (x{i}+c{i}) mod r{i} != eval[{i}] on {v}')
            if i < 8 and c[i + 1] != (v[i] + c[i]) // R[i]:
                raise ProbeError(f'factnorm_carry: c{i+1} breaks the carry recurrence on {v}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        v, y = x.tolist(), self.expr.eval(x).tolist()
        c = int(new_class)
        for i in range(int(p), len(v)):
            t = v[i] + c
            r = min(i + 2, 10)
            y[i] = t % r; c = t // r
        return torch.tensor(y, dtype=torch.long)


class Nonadj4Pick(ProbeVariable):
    """nonadj4_pick on nonadj4_id (integrated)."""
    variable_id = 'nonadj4_pick'
    algorithm = 'nonadj4_id'
    n_classes = 3
    def labels(self, inputs: torch.Tensor) -> torch.Tensor:
        out = []
        for r in inputs.tolist():
            row = []
            for i in range(3, 10):
                sums = [r[i - 3] + r[i - 1], r[i - 3] + r[i], r[i - 2] + r[i]]
                row.append(sums.index(max(sums)))
            out.append(row)
        return torch.tensor(out, dtype=torch.long)
    def valid_positions(self) -> torch.Tensor:
        return torch.tensor([3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    def check_row(self, x: torch.Tensor, lab: torch.Tensor) -> None:
        _pins_label(self, x, lab)   # three pairs routinely share a residue, so the printed digit cannot see every mislabel
        v, y = x.tolist(), self.expr.eval(x)
        for k, i in enumerate(self.valid_positions().tolist()):
            a, b = [(i - 3, i - 1), (i - 3, i), (i - 2, i)][int(lab[k])]
            if int(y[i]) != (v[a] + v[b]) % 10:
                raise ProbeError(f'nonadj4_pick: picked pair sum != eval[{i}] on {v}')
            if v[a] + v[b] != max(v[i - 3] + v[i - 1], v[i - 3] + v[i], v[i - 2] + v[i]):
                raise ProbeError(f'nonadj4_pick: picked pair is not the window optimum at {i} on {v}')
    def counterfactual(self, x: torch.Tensor, p: int, new_class: int) -> torch.Tensor | None:
        v = x.tolist()
        a, b = [(p - 3, p - 1), (p - 3, p), (p - 2, p)][int(new_class)]
        y = self.expr.eval(x)
        y[p] = (v[a] + v[b]) % 10
        return y

bridge_variable('dblodd_dsub_id', 1)

bridge_variable('slew2_psum_id', 1)

bridge_variable('rms3_psum_id', 1)

if __name__ == "__main__":
    known = models_txt()
    print(f"models.txt: {len(known)} models")
    bridge_variable("bwt_dblodd_id", 1)   # the factory unit test target
    failed = []
    for cls in VARIABLES:
        if cls.algorithm not in known:
            print(f"  SKIP {cls.algorithm}/{cls.variable_id}: model not in models.txt")
            continue
        try:
            n = validate(cls)
        except ProbeError as e:
            failed.append(f"{cls.algorithm}/{cls.variable_id}: {e}")
            print(f"  FAIL {failed[-1]}")
            continue
        print(f"  {cls.algorithm}/{cls.variable_id}: {n}/{n} label rows validated "
              f"({cls.n_classes} classes, positions {cls().valid_positions().tolist()})")
    if failed:
        raise SystemExit(f"{len(failed)} of {len(VARIABLES)} variables FAILED validation")
    print("variables self-check PASSED.")
