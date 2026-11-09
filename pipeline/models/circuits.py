#!/usr/bin/env python3
"""Circuits: the partial algorithm a trained model implements under a condition.

A circuit subclasses its parent DSL operator together with the DomainCircuit mixin:

    class ReverseOnlyCircuit(SegmentReverse, DomainCircuit):
        circuit_id = "reverse_only"                    # catalog name
        algorithm  = "segrev3_id"                      # the model it's found under
        _cond = conditions.exclude_digit(3)            # predicate + matching sampler
        def eval(self, x): ...                         # the collapsed sub-algorithm

condition(x) marks the inputs where eval() agrees with the true operator; both come
from ``_cond`` (a conditions.Condition) unless overridden. eval() defaults to the
parent operator (a domain restriction with no mechanism collapse); override it when
the circuit computes something simpler on its domain. Defining the class registers
it in CIRCUITS (via __init_subclass__); make_circuits_txt.py catalogs + validates.

Usually one circuit per algorithm, but an operator may have several (just define more).
Circuits are specs, not trained: eval() is ground truth to compare a model against.
"""
import torch

import conditions

from algorithms import DSLExpr, PrefixMax, RunningCountDigit, decode_algorithm_id, base_algorithm_id, BWT, Id, Sarank, AccumulateReset, AnchorClamp, AnchorSubtract, ArgMinMax, CompareSwap, ConditionalJump, CountOdd, DataReflect, DataRotate, Deplete, DoubleOdd, FixedPointIndicator, MapMul, NoveltyIndicator, ParityRoute, PrefixReverseK, PrefixSortK, PrefixSumMod, PrefixSumOdd, RecordCount, RecordIndicator, ResetRamp, Reverse, RippleCarry, RunningProduct, SRLatch, SampleHold, SegmentReverse, SignedWalk, StrideGather, TaskMux, TokenResetMax, WindowMax, Pmaxv9, SmallDFATrace, Indecho, GroupCosetProductS3, Kanc2, Kanc4, Kanc8, Kanc10, Sanc2, Tmap5, Addfix, CellularAutomaton, ChainedIndex, Sjf1, Duelsum, Netrm, Convfix, Usum, Perase0, Lagf3, Nearidx, Tresetsum3, Oepass0, Pge, Cgpmax09, Lanesum, Addpmax, Maxrun, Dualtie, Cap59, Cgrev09, Dupmax, CompareShift, Rmaj5, Majmask, DiffShift, AnchorBroadcast, Zpeak, Mulanc0, Recdist, Pmrep, Mulpair, PrevSegmentMax, Rotate, Occcnt, SegmentDistinct, Dclamp, FirstUnique, Runmed, Cntge3, Twosum10, Holdodd, Addlastodd, Bmcand, Hold2ge5, Sjf2, Gapge5, Riseerase, Ebud3, Tieswap, Risemax, Delc2, Riskeep, Mixtie, Pmaxv89, Pmaxv789, Togsum0, Signsum0, Median3, Majority3, Spile4, Spile5, Spile6, Cyca, Rule184Thr4, Rule184Thr5, Rule184Thr6, Ewma2, Zcross5, Heapmax, Vig2, Vig3, Vig5, Hamk3, Altrun, BlockSwapParity, BrickSort2, FeistelAdd, BlockMax2, ErrorDiffuse2, ErrorDiffuse3, ErrorDiffuse5, BoxSum2, BoxSum3, BoxSum4, BoxSum5, Schmitt, RailFence2, RailFence3, RailFence4, QueueWalk3, QueueWalk4, QueueWalk5, QueueWalk6, MinBottleneck, MaxCapacity, HeapSubtreeMax, IncRunMax, HaarStep, RunMAD, ContFrac, GroupProductS3, GroupProductD4, GroupProductQ8, GroupProductD5, D4Character, GroupActionD5, RunningRangeAdd, RunningMedianAdd, ThirdOrderScan, LinearRotor3, RunningModeCountAdd, Oscpos, Oscmom, RefractoryVeto, MexicanHatVeto, MaxFrequency, DucciStep, PairGcd, SecondOrderScan, NivenByCount, EuclidStep, PairCarry, InversionCount, MajorIndex, NimHistogram, MeanRelax, AnchorNudge, PhaseNudge, HeatExchange, ElemSym2, RunningMex, MirrorAdd, MirrorMax, MirrorMin, OnlineRankAdd, SignDiffuse, DoubleGE, MidrangeAdd, SpacedSeedSum, ArgPosAdd, LaneRunningMax, PancakeFlip, BitPlaneMajority, PredicatedSkip, BitTransitionCount, OpcodeMachine, KnockoutDepth, EnvelopeMax, CountMinSketch, BitTransposeGather, InternRank, MultiplicityParityXor, ProximityDegree, CumFreqRank, DistinctSubseqCount, CodonSkip, FragLen, DivisorZeta, MobiusInvert, FactoradicNormalize, NonAdjacentMax4, BoundedConfidenceMean, ExtremalReplace, Hsifte, LinearProbe, FlagPartition, Slew2, Rms3, Spn2, Xpar


class CircuitError(Exception):
    pass


CIRCUITS: list[type] = []

REJECTION_BUDGET_PER_ROW = 10000   # rejection draws per requested row before demanding sample()


class DomainCircuit:
    """Mixin marking a partial-algorithm circuit found under a DSL operator."""

    circuit_id: str = ""
    algorithm: str = ""
    scored_output_indices: tuple | None = None   # None => score all output positions
    _cond = None   # conditions.Condition; set it, or override condition() (+ sample() if rejection is too slow)

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.circuit_id and not cls.algorithm:
            return   # abstract base (e.g. ContrastCircuit) — not a concrete circuit
        if not cls.circuit_id or not cls.algorithm:
            raise CircuitError(f"{cls.__name__} must set circuit_id and algorithm")
        CIRCUITS.append(cls)

    def condition(self, input_seq: torch.Tensor) -> bool:
        """Whether this circuit's eval matches the true algorithm on input_seq."""
        if self._cond is None:
            raise NotImplementedError(f"{self.circuit_id}: set _cond or override condition()")
        return self._cond.holds(input_seq)

    def sample(self, batch: int, seed: int) -> torch.Tensor:
        """``[batch, input_length]`` inputs drawn UNIFORMLY from ``condition``'s domain:
        ``_cond.sample`` when set, else vectorized rejection over ``condition`` (the
        budget below fails loud when the accept-rate is too low)."""
        if self._cond is not None:
            return self._cond.sample(batch, seed)
        g = torch.Generator().manual_seed(conditions.stream_key(seed, self.circuit_id))
        budget, drawn, need, kept = batch * REJECTION_BUDGET_PER_ROW, 0, batch, []
        while need > 0:
            block = torch.randint(0, 10, (max(4 * need, 1024), self.input_length), generator=g)
            drawn += block.shape[0]
            good = block[torch.tensor([self.condition(row) for row in block])]
            if good.numel():
                kept.append(good); need -= good.shape[0]
            if drawn > budget:
                raise CircuitError(f"{self.circuit_id}: rejection accept-rate too low "
                                   f"({batch - need}/{batch} after {drawn} draws); override sample().")
        return torch.cat(kept)[:batch].to(torch.long)


def _parent_op(cls: type) -> type:
    """The DSL operator a circuit subclasses."""
    return next(b for b in cls.__mro__[1:] if issubclass(b, DSLExpr) and b is not DSLExpr)


def build(cls: type) -> DomainCircuit:
    """Instantiate a circuit from its algorithm id (pulls operator params + child via decode)."""
    parent = decode_algorithm_id(cls.algorithm)
    inst = cls.__new__(cls)
    inst.__dict__.update(parent.__dict__)
    return inst


def validate(cls: type, trials: int = 2000, seed: int = 0) -> int:
    """Draw ``trials`` inputs from the circuit's condition domain (its ``sample``) and raise if
    ``eval`` disagrees with the true algorithm on any; return the number checked. Sampling in-domain
    (not random 0-9) is what makes tight conditions like sum<=12 or all-even actually testable."""
    parent = decode_algorithm_id(cls.algorithm)
    if type(parent) is not _parent_op(cls):
        raise CircuitError(f"{cls.circuit_id}: algorithm '{cls.algorithm}' is not a {_parent_op(cls).__name__}")
    inst = build(cls)
    xs = inst.sample(trials, seed)
    for i in range(xs.shape[0]):
        x = xs[i]
        if not inst.condition(x):
            raise CircuitError(f"{cls.circuit_id}: sample() drew an off-condition input {x.tolist()}")
        if not torch.equal(inst.eval(x), parent.eval(x)):
            raise CircuitError(f"{cls.circuit_id}: eval != {cls.algorithm} on {x.tolist()} where condition holds")
    return int(xs.shape[0])


class ContrastCircuit(DomainCircuit):
    """A circuit defined by a minimal-edit clean/corrupt pair: the corruption toggles ONE
    causal predicate, so the circuit found is the machinery FOR that predicate. eval stays
    the TRUE parent operator (ground truth on both sides makes the per-row output-diff
    masks derivable at load time); ``base_circuit`` names the sibling filter circuit whose
    edges complete the differential union test. ``local_predicate`` says whether the toggled
    predicate is consumed only at the mask positions (set False when it feeds every output)."""

    base_circuit: str | None = None
    local_predicate: bool = True   # predicate consumed only at toggle-affected positions; False = d_unaffected reported, not gated

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Override; the FIRST body line must be ``seed = conditions.stream_key(seed, self.circuit_id)``
        so sibling scopes of one model draw independent rows (the clean/corrupt members of a pair
        stay correlated — only the CROSS-SCOPE stream is separated)."""
        raise NotImplementedError(f"{self.circuit_id}: override sample_pairs()")

    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self.sample_pairs(batch, seed)[0]   # clean side; domain-draw consumers unchanged

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        """Optional [out_len] bool claim of which outputs the toggle moves (None = no claim)."""
        return None


def validate_pairs(cls: type, trials: int = 512, seed: int = 0) -> int:
    """Draw pairs and raise unless every row has: clean in-domain, differing inputs, differing
    TRUE outputs (non-empty mask — the DoubleOdd-preimage trap), and a matching
    ``affected_outputs`` claim when one is made. Returns the number of pairs checked."""
    inst = build(cls)
    parent = decode_algorithm_id(cls.algorithm)
    clean, corrupt = inst.sample_pairs(trials, seed)
    for i in range(clean.shape[0]):
        c, k = clean[i], corrupt[i]
        if not inst.condition(c):
            raise CircuitError(f"{cls.circuit_id}: clean row {i} off-condition {c.tolist()}")
        if torch.equal(c, k):
            raise CircuitError(f"{cls.circuit_id}: pair {i} has identical inputs {c.tolist()}")
        diff = parent.eval(c) != parent.eval(k)
        if not bool(diff.any()):
            raise CircuitError(f"{cls.circuit_id}: pair {i} outputs identical (empty mask) {c.tolist()}")
        claim = inst.affected_outputs(c, k)
        if claim is not None and not torch.equal(claim.bool(), diff):
            raise CircuitError(f"{cls.circuit_id}: pair {i} diff mask != affected_outputs claim")
    return int(clean.shape[0])


def _edit_pos(clean: torch.Tensor, corrupt: torch.Tensor) -> int:
    """The single minimal-edit position (first differing input index)."""
    return int((clean != corrupt).nonzero()[0])


def _mask(length: int, positions) -> torch.Tensor:
    """Bool [length] mask true at ``positions`` (an int, index list, or slice)."""
    m = torch.zeros(length, dtype=torch.bool); m[positions] = True; return m


def _rejection_pairs(inst: "ContrastCircuit", batch: int, seed: int,
                     edit) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform clean rows; ``edit(row, g)`` may pin clean fields and returns the corrupt clone;
    keep only rows whose TRUE parent output differs (non-empty, unclaimed mask). Fails loud if the
    edit rarely moves the output — bounded, never hangs."""
    g = torch.Generator().manual_seed(int(seed))
    cleans, corrupts, stalls = [], [], 0
    while len(cleans) < batch:
        before = len(cleans)
        block = torch.randint(0, 10, (max(4 * (batch - len(cleans)), 512), inst.input_length), generator=g)
        for row in block:
            corrupt = edit(row, g)
            if bool((inst.eval(row) != inst.eval(corrupt)).any()):
                cleans.append(row.clone()); corrupts.append(corrupt)
                if len(cleans) == batch:
                    break
        stalls = stalls + 1 if len(cleans) == before else 0
        if stalls >= 20:                    # degenerate edit: 20 blocks with zero non-empty-mask pairs
            raise CircuitError(f"{inst.circuit_id}: edit rarely changes the output "
                               f"({len(cleans)}/{batch} pairs, {stalls} empty blocks) — degenerate contrast")
    return torch.stack(cleans), torch.stack(corrupts)


def _anchor_edit(lo: int, hi: int):
    """Edit fn (for a position-0 anchor op): pin the anchor to ``[lo, hi]``, bump it +1 in corrupt."""
    def edit(row: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        row[0] = min(max(int(row[0]), lo), hi)
        corrupt = row.clone(); corrupt[0] = int(row[0]) + 1
        return corrupt
    return edit


class SuppressionContrastCircuit(ContrastCircuit):
    """Cue-flip contrast on a TaskMux model (C3 suppression): clean = task-``cued_task``-cued,
    corrupt = task-``wrong_task``-cued, SAME payload. eval stays the TRUE TaskMux op, so the
    per-pair mask = positions where cued(payload) != wrong(payload) -- exactly the slots the
    wrong task could intrude on. The suppression circuit is the machinery that keeps the wrong
    task OFF those slots; the --suppression selection mode finds it as the fringe (beyond the
    cued task's executor) whose severing RELEASES the wrong answer. base_circuit = the cued
    task's own circuit (``task{cued}``)."""

    suppression = True
    cued_task: int = -1
    wrong_task: int = -1

    def condition(self, input_seq: torch.Tensor) -> bool:
        return int(input_seq[0]) == 10 + self.cued_task

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        cued, wrong = self.members[self.cued_task], self.members[self.wrong_task]
        cued_tok = torch.tensor([10 + self.cued_task]); wrong_tok = torch.tensor([10 + self.wrong_task])
        cleans, corrupts, stalls = [], [], 0
        while len(cleans) < batch:
            before = len(cleans)
            payload = torch.randint(0, 10, (max(4 * (batch - len(cleans)), 1024), 10), generator=g)
            differ = torch.tensor([bool((cued.eval(p) != wrong.eval(p)).any()) for p in payload])
            for p in payload[differ][:batch - len(cleans)]:
                cleans.append(torch.cat([cued_tok, p])); corrupts.append(torch.cat([wrong_tok, p]))
            stalls = stalls + 1 if len(cleans) == before else 0
            if stalls >= 20:                # tasks agree on every payload -> no suppression contrast
                raise CircuitError(f"{self.circuit_id}: tasks {self.cued_task}/{self.wrong_task} agree too "
                                   f"often ({len(cleans)}/{batch} differing payloads) — degenerate suppression")
        return torch.stack(cleans), torch.stack(corrupts)

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        p = clean_row[1:]
        return self.members[self.cued_task].eval(p) != self.members[self.wrong_task].eval(p)


def suppression_circuit(mux_algorithm_id: str, cued: int, wrong: int) -> type:
    """Manufacture (+ register, idempotent) a SuppressionContrastCircuit for the ordered task
    pair (cued, wrong) of a TaskMux model. circuit_id = ``suppress_t{cued}_over_t{wrong}``."""
    base = base_algorithm_id(mux_algorithm_id)
    parent = decode_algorithm_id(base)
    if not isinstance(parent, TaskMux):
        raise CircuitError(f"{mux_algorithm_id!r} is not a TaskMux model")
    n = len(parent.members)
    if not (0 <= cued < n and 0 <= wrong < n and cued != wrong):
        raise CircuitError(f"invalid ordered task pair ({cued},{wrong}) for a {n}-task mux")
    cid = f"suppress_t{cued}_over_t{wrong}"
    for c in CIRCUITS:
        if c.algorithm == base and c.circuit_id == cid:
            return c
    return type(f"Suppress_{base}_t{cued}_over_t{wrong}", (TaskMux, SuppressionContrastCircuit),
                {"circuit_id": cid, "algorithm": base, "cued_task": cued, "wrong_task": wrong,
                 "base_circuit": f"task{cued}", "__doc__": SuppressionContrastCircuit.__doc__})


def suppression_circuits_for(mux_algorithm_id: str) -> list[type]:
    """Every ordered-pair suppression circuit of a TaskMux model (registers them all)."""
    parent = decode_algorithm_id(base_algorithm_id(mux_algorithm_id))
    if not isinstance(parent, TaskMux):
        raise CircuitError(f"{mux_algorithm_id!r} is not a TaskMux model")
    n = len(parent.members)
    return [suppression_circuit(mux_algorithm_id, a, b)
            for a in range(n) for b in range(n) if a != b]


def task_circuit(mux_algorithm_id: str, task: int) -> type:
    """Manufacture (+ register, idempotent) the per-task DomainCircuit of a TaskMux model:
    circuit_id = ``task{task}``, cue token pinned to 10+task so the mux collapses to member `task`
    (eval stays the true TaskMux op on that domain). These are the base executors the ordered-pair
    suppression circuits sit on (base_circuit=``task{cued}``). Returns the hand-written class when
    one is already registered (e.g. mux5's Task0..4Circuit)."""
    base = base_algorithm_id(mux_algorithm_id)
    parent = decode_algorithm_id(base)
    if not isinstance(parent, TaskMux):
        raise CircuitError(f"{mux_algorithm_id!r} is not a TaskMux model")
    n = len(parent.members)
    if not 0 <= task < n:
        raise CircuitError(f"invalid task {task} for a {n}-task mux")
    cid = f"task{task}"
    for c in CIRCUITS:
        if c.algorithm == base and c.circuit_id == cid:
            return c
    safe = base.replace(".", "_").replace("-", "_")
    return type(f"Task{task}_{safe}Circuit", (TaskMux, DomainCircuit),
                {"circuit_id": cid, "algorithm": base, "_cond": conditions.task_pin(task, n),
                 "__doc__": f"task{task} of {base}: cue pinned to token {10 + task} -> member {task}."})


def task_circuits_for(mux_algorithm_id: str) -> list[type]:
    """Every per-task DomainCircuit of a TaskMux model (registers them all)."""
    parent = decode_algorithm_id(base_algorithm_id(mux_algorithm_id))
    if not isinstance(parent, TaskMux):
        raise CircuitError(f"{mux_algorithm_id!r} is not a TaskMux model")
    return [task_circuit(mux_algorithm_id, i) for i in range(len(parent.members))]


# --- Circuits: one per algorithm as you find them (an operator may carry several) ---


class PrefixMaxFirstMaxConstant(PrefixMax, DomainCircuit):
    """When x[0] is the global max, the running max is the constant x[0]."""

    circuit_id = "pmax_first_is_max_constant"
    algorithm = "pmax_id"

    def condition(self, input_seq: torch.Tensor) -> bool:
        x = self.child.eval(input_seq)
        return bool((x[0] >= x).all())

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        x = self.child.eval(input_seq)
        return torch.full_like(x, int(x[0]))


class RunningCountDigitAbsentZero(RunningCountDigit, DomainCircuit):
    """When digit d never appears, the running count stays zero everywhere."""

    circuit_id = "rcnt5_absent_zero"
    algorithm = "rcnt5_id"

    def condition(self, input_seq: torch.Tensor) -> bool:
        x = self.child.eval(input_seq)
        return bool((x != self.d).all())

    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        x = self.child.eval(input_seq)
        return torch.zeros_like(x)


class BwtCircuit(BWT, DomainCircuit):
    """Burrows-Wheeler rotation-sort — isolated under all-even {0,2,4,6,8}; DoubleOdd=id (K3 fixed-pt)."""
    circuit_id = 'bwt'
    algorithm = 'bwt_dblodd_id'
    _cond = conditions.ALL_EVEN
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return BWT(Id(self.input_length)).eval(input_seq)


class SarankCircuit(Sarank, DomainCircuit):
    """inverse suffix array (suffix ranks) — isolated under all-even (K3)."""
    circuit_id = 'sarank'
    algorithm = 'sarank_dblodd_id'
    _cond = conditions.ALL_EVEN
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return Sarank(Id(self.input_length)).eval(input_seq)


class ReverseOnlyCircuit(SegmentReverse, DomainCircuit):
    """reverse_only on segrev3_id: with no boundary digit 3, the whole input reverses."""
    circuit_id = 'reverse_only'
    algorithm = 'segrev3_id'
    _cond = conditions.exclude_digit(3)
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return Reverse(Id(self.input_length)).eval(input_seq)


class ThreeDetectorCircuit(SegmentReverse, ContrastCircuit):
    """three_detector on segrev3_id: toggle ONE boundary 3 -> non-3. The minimal edit
    isolates the boundary-detection machinery; the executor stays in base reverse_only."""
    circuit_id = 'three_detector'
    algorithm = 'segrev3_id'
    base_circuit = 'reverse_only'
    local_predicate = False   # the boundary's segment-membership info feeds every output position

    def condition(self, input_seq: torch.Tensor) -> bool:
        return bool((input_seq == 3).any())

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        clean = DomainCircuit.sample(self, batch, seed)   # rejection sampler: >=1 three per row
        g = torch.Generator().manual_seed(int(seed) + 1)
        non3 = torch.tensor([0, 1, 2, 4, 5, 6, 7, 8, 9])
        corrupt = clean.clone()
        for i in range(clean.shape[0]):
            js = (clean[i] == 3).nonzero().flatten()
            j = int(js[torch.randint(js.numel(), (1,), generator=g)])
            corrupt[i, j] = int(non3[torch.randint(9, (1,), generator=g)])
        return clean, corrupt


class Areset13AccumCircuit(AccumulateReset, DomainCircuit):
    """areset13_accum on areset13_id: sum<=12 keeps the reset silent -> plain prefix sum."""
    circuit_id = 'areset13_accum'
    algorithm = 'areset13_id'
    _cond = conditions.sum_at_most(12)
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return PrefixSumMod(Id(self.input_length)).eval(input_seq)


def _areset_crossing_pairs(inst: AccumulateReset, batch: int, seed: int,
                           compensate: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairs straddling the wrap: acc residue a in [4, T-1] at k-1; clean x[k]=T-a crosses
    (acc -> 0), corrupt x[k]=T-1-a lands on T-1. compensate adds +1 at x[k+1] on the corrupt
    side (clean x[k+1] <= 8 leaves room), restoring the residue from k+1 on."""
    T, L = inst.threshold, inst.input_length
    g = torch.Generator().manual_seed(int(seed))
    cleans, corrupts = [], []
    for _ in range(batch):
        k = int(torch.randint(1, (L - 2) if compensate else (L - 1), (1,), generator=g))
        for _try in range(1000):
            pre = torch.randint(0, 10, (k,), generator=g)
            a = int(pre.sum()) % T
            if 4 <= a < T:
                break
        else:
            raise CircuitError(f"{inst.circuit_id}: no prefix with residue in [4,{T - 1}] after 1000 draws")
        suf = torch.randint(0, 10, (L - k - 1,), generator=g)
        if compensate:
            suf[0] = int(torch.randint(0, 9, (1,), generator=g))
        clean = torch.cat([pre, torch.tensor([T - a]), suf])
        corrupt = clean.clone()
        corrupt[k] = T - 1 - a
        if compensate:
            corrupt[k + 1] += 1
        cleans.append(clean); corrupts.append(corrupt)
    return torch.stack(cleans), torch.stack(corrupts)


class Areset13CrossingToggleCircuit(AccumulateReset, ContrastCircuit):
    """areset13_crossing_toggle on areset13_id: clean crosses the threshold at k, corrupt lands
    on 12. Residues stay 1 apart mod 13 and never collide through %10 -> gap-free mask {k..9}."""
    circuit_id = 'areset13_crossing_toggle'
    algorithm = 'areset13_id'
    base_circuit = 'areset13_accum'

    def condition(self, input_seq: torch.Tensor) -> bool:
        return bool(input_seq.sum() >= self.threshold)

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        return _areset_crossing_pairs(self, batch, seed, compensate=False)

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        k = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(self.input_length, dtype=torch.bool); m[k:] = True
        return m


class Areset13CrossingCompensatedCircuit(AccumulateReset, ContrastCircuit):
    """areset13_crossing_compensated on areset13_id: crossing toggle at k plus +1 at x[k+1] on
    the corrupt side, re-syncing the accumulator -> mask exactly {k} (the wrap readout site)."""
    circuit_id = 'areset13_crossing_compensated'
    algorithm = 'areset13_id'
    base_circuit = 'areset13_accum'

    def condition(self, input_seq: torch.Tensor) -> bool:
        return bool(input_seq.sum() >= self.threshold)

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        return _areset_crossing_pairs(self, batch, seed, compensate=True)

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        k = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(self.input_length, dtype=torch.bool); m[k] = True
        return m


class Areset13DeltaControlCircuit(AccumulateReset, ContrastCircuit):
    """areset13_delta_control on areset13_id: delta=1 at k with BOTH sides staying <= 12, so the
    reset never fires -> pure value perturbation through the accumulator, mask {k..9}."""
    circuit_id = 'areset13_delta_control'
    algorithm = 'areset13_id'
    base_circuit = 'areset13_accum'
    _cond = conditions.sum_at_most(11)   # clean <= 11 keeps the +1 corrupt side <= 12

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        clean = DomainCircuit.sample(self, batch, seed)
        g = torch.Generator().manual_seed(int(seed) + 1)
        ks = torch.randint(1, self.input_length - 1, (batch,), generator=g)
        rows = torch.arange(batch)
        v = clean[rows, ks]
        corrupt = clean.clone()
        corrupt[rows, ks] = torch.where(v <= 8, v + 1, v - 1)   # 9 steps down; sums stay <= 12 either way
        return clean, corrupt

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        k = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(self.input_length, dtype=torch.bool); m[k:] = True
        return m


class SingleRecordCircuit(RecordCount, DomainCircuit):
    """single_record on reccount_id: isolated via conditions.BINARY."""
    circuit_id = 'single_record'
    algorithm = 'reccount_id'
    _cond = conditions.BINARY


class SingleThreeCircuit(RunningCountDigit, DomainCircuit):
    """single_three on rcnt3_id: isolated via conditions.at_most_k_of(3, 1)."""
    circuit_id = 'single_three'
    algorithm = 'rcnt3_id'
    _cond = conditions.at_most_k_of(3, 1)


class OddOnlyCircuit(PrefixSumOdd, DomainCircuit):
    """odd_only on psumodd_id: all-odd input -> every digit counts -> plain prefix sum."""
    circuit_id = 'odd_only'
    algorithm = 'psumodd_id'
    _cond = conditions.ALL_ODD
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return PrefixSumMod(Id(self.input_length)).eval(input_seq)


class OneOddCircuit(DoubleOdd, DomainCircuit):
    """one_odd on dblodd_id: isolated via conditions.ONE_ODD."""
    circuit_id = 'one_odd'
    algorithm = 'dblodd_id'
    _cond = conditions.ONE_ODD


class AllOddCircuit(DoubleOdd, DomainCircuit):
    """all_odd on dblodd_id: all-odd input -> every digit doubles -> uniform x2 mod 10."""
    circuit_id = 'all_odd'
    algorithm = 'dblodd_id'
    _cond = conditions.ALL_ODD
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return MapMul(2, Id(self.input_length)).eval(input_seq)


class DbloddParityToggleCircuit(DoubleOdd, ContrastCircuit):
    """dblodd_parity_toggle on dblodd_id: flip ONE digit's parity at a varying p; the partner
    map dodges the output-preimage pairs 1<->2 and 8<->9 (empty mask). Mask: exactly {p}."""
    circuit_id = 'dblodd_parity_toggle'
    algorithm = 'dblodd_id'
    base_circuit = 'all_odd'
    _cond = conditions.FULL
    _PARTNER = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6, 7, 0])   # parity-toggling, output-distinct partner per digit

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        clean = DomainCircuit.sample(self, batch, seed)
        g = torch.Generator().manual_seed(int(seed) + 1)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        rows = torch.arange(batch)
        corrupt = clean.clone()
        corrupt[rows, p] = self._PARTNER[clean[rows, p]]
        return clean, corrupt

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        return clean_row != corrupt_row   # positionwise map: output mask == input edit site


class ProdLocateCircuit(RunningProduct, DomainCircuit):
    """prod_locate on prod_id: isolated via conditions.exactly_one_in([0,2,3,4,5,6,7,8,9])."""
    circuit_id = 'prod_locate'
    algorithm = 'prod_id'
    _cond = conditions.exactly_one_in([0, 2, 3, 4, 5, 6, 7, 8, 9])


class CntoddParityCircuit(CountOdd, DomainCircuit):
    """cntodd_parity on cntodd_id: isolated via conditions.ONE_ODD."""
    circuit_id = 'cntodd_parity'
    algorithm = 'cntodd_id'
    _cond = conditions.ONE_ODD


class RecordindBinaryCircuit(RecordIndicator, DomainCircuit):
    """recordind_binary on recordind_id: isolated via conditions.BINARY."""
    circuit_id = 'recordind_binary'
    algorithm = 'recordind_id'
    _cond = conditions.BINARY


class NovelOnerepeatCircuit(NoveltyIndicator, DomainCircuit):
    """novel_onerepeat on novel_id: isolated via conditions.ONE_REPEAT."""
    circuit_id = 'novel_onerepeat'
    algorithm = 'novel_id'
    _cond = conditions.ONE_REPEAT


class FixedpointOnehitCircuit(FixedPointIndicator, DomainCircuit):
    """fixedpoint_onehit on fixedpoint_id: isolated via conditions.ONE_FIXEDPOINT."""
    circuit_id = 'fixedpoint_onehit'
    algorithm = 'fixedpoint_id'
    _cond = conditions.ONE_FIXEDPOINT


class Reset3SinglerampCircuit(ResetRamp, DomainCircuit):
    """reset3_singleramp on reset3_id: isolated via conditions.at_most_k_of(3, 1)."""
    circuit_id = 'reset3_singleramp'
    algorithm = 'reset3_id'
    _cond = conditions.at_most_k_of(3, 1)


class Tresetmax3RunmaxCircuit(TokenResetMax, DomainCircuit):
    """tresetmax3_runmax on tresetmax3_id: no reset token 3 -> plain running max."""
    circuit_id = 'tresetmax3_runmax'
    algorithm = 'tresetmax3_id'
    _cond = conditions.exclude_digit(3)
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return PrefixMax(Id(self.input_length)).eval(input_seq)


class Walk12DowncountCircuit(SignedWalk, DomainCircuit):
    """walk12_downcount on walk12_id: isolated via conditions.exclude_digit(1)."""
    circuit_id = 'walk12_downcount'
    algorithm = 'walk12_id'
    _cond = conditions.exclude_digit(1)


class Sr12SetlatchCircuit(SRLatch, DomainCircuit):
    """sr12_setlatch on sr12_id: isolated via conditions.exclude_digit(2)."""
    circuit_id = 'sr12_setlatch'
    algorithm = 'sr12_id'
    _cond = conditions.exclude_digit(2)


class Sr12ResetToggleCircuit(SRLatch, ContrastCircuit):
    """sr12_reset_toggle on sr12_id: pin state to 1 (set at s<p, no reset between), then clean
    x[p]=reset vs corrupt x[p]=hold. Later drivers resync -> mask is the window {p..q-1}."""
    circuit_id = 'sr12_reset_toggle'
    algorithm = 'sr12_id'
    base_circuit = 'sr12_setlatch'

    def condition(self, input_seq: torch.Tensor) -> bool:
        state = 0   # holds iff some reset fires while the latch is 1 (idempotency trap otherwise)
        for v in input_seq.tolist():
            if v == self.reset_tok and state == 1:
                return True
            state = 1 if v == self.set_tok else (0 if v == self.reset_tok else state)
        return False

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        hold = torch.tensor([d for d in range(10) if d not in (self.set_tok, self.reset_tok)])
        non_reset = torch.tensor([d for d in range(10) if d != self.reset_tok])
        clean = torch.randint(0, 10, (batch, L), generator=g)
        ps = torch.randint(1, L, (batch,), generator=g)
        corrupt_vals = hold[torch.randint(hold.numel(), (batch,), generator=g)]
        for r in range(batch):
            p = int(ps[r])
            s = int(torch.randint(p, (1,), generator=g))
            clean[r, s] = self.set_tok
            for t in range(s + 1, p):   # scrub resets so the state is still 1 at p-1
                if int(clean[r, t]) == self.reset_tok:
                    clean[r, t] = int(non_reset[torch.randint(non_reset.numel(), (1,), generator=g)])
            clean[r, p] = self.reset_tok
        corrupt = clean.clone()
        corrupt[torch.arange(batch), ps] = corrupt_vals
        return clean, corrupt

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        p = int((clean_row != corrupt_row).nonzero()[0])
        drivers = ((clean_row == self.set_tok) | (clean_row == self.reset_tok)).nonzero().flatten()
        q = int(drivers[drivers > p][0]) if bool((drivers > p).any()) else self.input_length
        m = torch.zeros(self.input_length, dtype=torch.bool); m[p:q] = True
        return m


class Holdge5OneholdCircuit(SampleHold, DomainCircuit):
    """holdge5_onehold on holdge5_id: isolated via conditions.exactly_one_in([5,6,7,8,9])."""
    circuit_id = 'holdge5_onehold'
    algorithm = 'holdge5_id'
    _cond = conditions.exactly_one_in([5, 6, 7, 8, 9])


class Dep9f0RampCircuit(Deplete, DomainCircuit):
    """dep9f0_ramp on dep9f0_id: isolated via conditions.sum_at_most(9)."""
    circuit_id = 'dep9f0_ramp'
    algorithm = 'dep9f0_id'
    _cond = conditions.sum_at_most(9)


class Anc0CaesarCircuit(AnchorSubtract, DomainCircuit):
    """anc0_caesar on anc0_id: isolated via conditions.pin(0, [3])."""
    circuit_id = 'anc0_caesar'
    algorithm = 'anc0_id'
    _cond = conditions.pin(0, [3])


class Fclamp0ClampCircuit(AnchorClamp, DomainCircuit):
    """fclamp0_clamp on fclamp0_id: isolated via conditions.pin(0, [3])."""
    circuit_id = 'fclamp0_clamp'
    algorithm = 'fclamp0_id'
    _cond = conditions.pin(0, [3])


class Drot0RotateCircuit(DataRotate, DomainCircuit):
    """drot0_rotate on drot0_id: isolated via conditions.pin(0, [3])."""
    circuit_id = 'drot0_rotate'
    algorithm = 'drot0_id'
    _cond = conditions.pin(0, [3])


class Refl0MirrorCircuit(DataReflect, DomainCircuit):
    """refl0_mirror on refl0_id: isolated via conditions.pin(0, [3])."""
    circuit_id = 'refl0_mirror'
    algorithm = 'refl0_id'
    _cond = conditions.pin(0, [3])


class Stride0GatherCircuit(StrideGather, DomainCircuit):
    """stride0_gather on stride0_id: isolated via conditions.pin(0, [3])."""
    circuit_id = 'stride0_gather'
    algorithm = 'stride0_id'
    _cond = conditions.pin(0, [3])


class Pref0RevkCircuit(PrefixReverseK, DomainCircuit):
    """pref0_revk on pref0_id: isolated via conditions.pin(0, [5])."""
    circuit_id = 'pref0_revk'
    algorithm = 'pref0_id'
    _cond = conditions.pin(0, [5])


class Psort0SortkCircuit(PrefixSortK, DomainCircuit):
    """psort0_sortk on psort0_id: isolated via conditions.pin(0, [5])."""
    circuit_id = 'psort0_sortk'
    algorithm = 'psort0_id'
    _cond = conditions.pin(0, [5])


class Cmpswap09SwapCircuit(CompareSwap, DomainCircuit):
    """cmpswap09_swap on cmpswap0p9_id: isolated via conditions.compare_positions(0, 9)."""
    circuit_id = 'cmpswap09_swap'
    algorithm = 'cmpswap0p9_id'
    _cond = conditions.compare_positions(0, 9)


class Cmpswap09TieStraddleCircuit(CompareSwap, ContrastCircuit):
    """cmpswap09_tie_straddle on cmpswap0p9_id: x[9]=b; clean x[0]=b+1 (swap fires), corrupt
    x[0]=b (tie holds). The copy confound cancels at 0 (both outputs = b) -> mask exactly {9}."""
    circuit_id = 'cmpswap09_tie_straddle'
    algorithm = 'cmpswap0p9_id'
    base_circuit = 'cmpswap09_swap'

    def condition(self, input_seq: torch.Tensor) -> bool:
        return int(input_seq[self.i]) == int(input_seq[self.j]) + 1

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        b = torch.randint(0, 9, (batch,), generator=g)
        clean[:, self.j] = b
        clean[:, self.i] = b + 1
        corrupt = clean.clone()
        corrupt[:, self.i] = b
        return clean, corrupt

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(self.input_length, dtype=torch.bool); m[self.j] = True
        return m


class Bmax1DelayCircuit(WindowMax, DomainCircuit):
    """bmax1_delay on bmax1_id: isolated via conditions.monotone(False)."""
    circuit_id = 'bmax1_delay'
    algorithm = 'bmax1_id'
    _cond = conditions.monotone(False)


class PmstepEvenaddCircuit(ParityRoute, DomainCircuit):
    """pmstep_evenadd on pmstep_id: isolated via conditions.ALL_EVEN."""
    circuit_id = 'pmstep_evenadd'
    algorithm = 'pmstep_id'
    _cond = conditions.ALL_EVEN


class Carryk1LocalCircuit(RippleCarry, DomainCircuit):
    """carryk1_local on carryk1_id: isolated via conditions.pin(9, [0..8]) (no overflow)."""
    circuit_id = 'carryk1_local'
    algorithm = 'carryk1_id'
    _cond = conditions.pin(9, [0, 1, 2, 3, 4, 5, 6, 7, 8])


class ArgminmaxArgmaxCircuit(ArgMinMax, DomainCircuit):
    """argminmax_argmax on argminmax_id: full domain, scoring only the argmax half."""
    circuit_id = 'argminmax_argmax'
    algorithm = 'argminmax_id'
    _cond = conditions.FULL
    scored_output_indices = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


class CjumpForwardCircuit(ConditionalJump, DomainCircuit):
    """cjump_forward on cjump10t5_id: isolated via conditions.domain([6,7,8,9])."""
    circuit_id = 'cjump_forward'
    algorithm = 'cjump10t5_id'
    _cond = conditions.domain([6, 7, 8, 9])


# --- C3 task circuits: pin the mux cue -> collapse to one member (base for the suppression factory) ---

MUX5 = 'mux5.psum-id.cmpswap0p9-id.jump10-id.pmstep-id.segrev3-id'


class Task0Circuit(TaskMux, DomainCircuit):
    "task0 on mux5: pin cue to token 10 -> collapses to member 0; eval = TaskMux on this domain."
    circuit_id = 'task0'
    algorithm = MUX5
    _cond = conditions.task_pin(0, 5)


class Task1Circuit(TaskMux, DomainCircuit):
    "task1 on mux5: pin cue to token 11 -> collapses to member 1."
    circuit_id = 'task1'
    algorithm = MUX5
    _cond = conditions.task_pin(1, 5)


class Task2Circuit(TaskMux, DomainCircuit):
    "task2 on mux5: pin cue to token 12 -> collapses to member 2."
    circuit_id = 'task2'
    algorithm = MUX5
    _cond = conditions.task_pin(2, 5)


class Task3Circuit(TaskMux, DomainCircuit):
    "task3 on mux5: pin cue to token 13 -> collapses to member 3."
    circuit_id = 'task3'
    algorithm = MUX5
    _cond = conditions.task_pin(3, 5)


class Task4Circuit(TaskMux, DomainCircuit):
    "task4 on mux5: pin cue to token 14 -> collapses to member 4."
    circuit_id = 'task4'
    algorithm = MUX5
    _cond = conditions.task_pin(4, 5)


# --- C1 contrast pairs: one minimal-edit clean/corrupt pair per model (mask = output-diff fingerprint) ---


class RcntDigitToggleCircuit(RunningCountDigit, ContrastCircuit):
    """rcnt_digit_toggle on rcnt3_id: toggle a 3<->non-3 at p; the count steps by 1 from p on -> {p..9}."""
    circuit_id = 'rcnt_digit_toggle'
    algorithm = 'rcnt3_id'
    base_circuit = 'single_three'
    _cond = conditions.FULL
    _NON3 = torch.tensor([0, 1, 2, 4, 5, 6, 7, 8, 9])

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(0, self.input_length, (batch,), generator=g); rows = torch.arange(batch)
        clean[rows, p] = 3
        corrupt = clean.clone()
        corrupt[rows, p] = self._NON3[torch.randint(0, 9, (batch,), generator=g)]
        return clean, corrupt

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        return _mask(self.input_length, slice(_edit_pos(clean_row, corrupt_row), None))


class CntoddParityToggleCircuit(CountOdd, ContrastCircuit):
    """cntodd_parity_toggle on cntodd_id: flip one digit's parity (even<->+1 odd); odd-count steps -> {p..9}."""
    circuit_id = 'cntodd_parity_toggle'
    algorithm = 'cntodd_id'
    base_circuit = 'cntodd_parity'
    _cond = conditions.FULL
    _EVEN = torch.tensor([0, 2, 4, 6, 8])

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(0, self.input_length, (batch,), generator=g); rows = torch.arange(batch)
        clean[rows, p] = self._EVEN[torch.randint(0, 5, (batch,), generator=g)]
        corrupt = clean.clone()
        corrupt[rows, p] = clean[rows, p] + 1
        return clean, corrupt

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        return _mask(self.input_length, slice(_edit_pos(clean_row, corrupt_row), None))


class Walk12UpdownFlipCircuit(SignedWalk, ContrastCircuit):
    """walk12_updown_flip on walk12_id: flip up-token 1 <-> down-token 2 at p; counter offsets by 2 -> {p..9}."""
    circuit_id = 'walk12_updown_flip'
    algorithm = 'walk12_id'
    base_circuit = 'walk12_downcount'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(0, self.input_length, (batch,), generator=g); rows = torch.arange(batch)
        clean[rows, p] = 1
        corrupt = clean.clone(); corrupt[rows, p] = 2
        return clean, corrupt

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        return _mask(self.input_length, slice(_edit_pos(clean_row, corrupt_row), None))


class PsumoddParityToggleCircuit(PrefixSumOdd, ContrastCircuit):
    """psumodd_parity_toggle on psumodd_id: odd v (contributes) <-> v-1 even (silent) at p; sum offsets -> {p..9}."""
    circuit_id = 'psumodd_parity_toggle'
    algorithm = 'psumodd_id'
    base_circuit = 'odd_only'
    _cond = conditions.FULL
    _ODD = torch.tensor([1, 3, 5, 7, 9])

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(0, self.input_length, (batch,), generator=g); rows = torch.arange(batch)
        clean[rows, p] = self._ODD[torch.randint(0, 5, (batch,), generator=g)]
        corrupt = clean.clone(); corrupt[rows, p] = clean[rows, p] - 1
        return clean, corrupt

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        return _mask(self.input_length, slice(_edit_pos(clean_row, corrupt_row), None))


class ProdZeroToggleCircuit(RunningProduct, ContrastCircuit):
    """prod_zero_toggle on prod_id: over units {1,3,7,9}, annihilate (0) vs no-op (1) at p -> {p..9}."""
    circuit_id = 'prod_zero_toggle'
    algorithm = 'prod_id'
    base_circuit = 'prod_locate'
    _cond = conditions.FULL
    _UNITS = torch.tensor([1, 3, 7, 9])

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = self._UNITS[torch.randint(0, 4, (batch, self.input_length), generator=g)]
        p = torch.randint(0, self.input_length, (batch,), generator=g); rows = torch.arange(batch)
        clean[rows, p] = 0
        corrupt = clean.clone(); corrupt[rows, p] = 1
        return clean, corrupt

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        return _mask(self.input_length, slice(_edit_pos(clean_row, corrupt_row), None))


def _record_straddle_pairs(length: int, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefix max exactly M; at p clean=M+1 (new record) vs corrupt=M (tie); suffix<=M (no later record)."""
    g = torch.Generator().manual_seed(int(seed))
    cleans, corrupts = [], []
    for _ in range(batch):
        M = int(torch.randint(0, 9, (1,), generator=g))
        p = int(torch.randint(1, length, (1,), generator=g))
        row = torch.randint(0, M + 1, (length,), generator=g)
        row[int(torch.randint(0, p, (1,), generator=g))] = M
        clean = row.clone(); clean[p] = M + 1
        corrupt = row.clone(); corrupt[p] = M
        cleans.append(clean); corrupts.append(corrupt)
    return torch.stack(cleans), torch.stack(corrupts)


class ReccountRecordToggleCircuit(RecordCount, ContrastCircuit):
    """reccount_record_toggle on reccount_id: new record (M+1) vs tie (M) at p; the count stays +1 -> {p..9}."""
    circuit_id = 'reccount_record_toggle'
    algorithm = 'reccount_id'
    base_circuit = 'single_record'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        return _record_straddle_pairs(self.input_length, batch, seed)

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        return _mask(self.input_length, slice(_edit_pos(clean_row, corrupt_row), None))


class RecordindRecordToggleCircuit(RecordIndicator, ContrastCircuit):
    """recordind_record_toggle on recordind_id: new record (M+1) vs tie (M) at p; suffix<=M -> mask {p}."""
    circuit_id = 'recordind_record_toggle'
    algorithm = 'recordind_id'
    base_circuit = 'recordind_binary'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        return _record_straddle_pairs(self.input_length, batch, seed)

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        return _mask(self.input_length, _edit_pos(clean_row, corrupt_row))


class PmaxNewmaxToggleCircuit(PrefixMax, ContrastCircuit):
    """pmax_newmax_toggle on pmax_id: new max (M+1) vs tie (M) at p; the later M+1 at q resyncs -> {p..q-1}."""
    circuit_id = 'pmax_newmax_toggle'
    algorithm = 'pmax_id'
    base_circuit = 'pmax_first_is_max_constant'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        L = self.input_length
        g = torch.Generator().manual_seed(int(seed))
        cleans, corrupts = [], []
        for _ in range(batch):
            M = int(torch.randint(0, 9, (1,), generator=g))
            p = int(torch.randint(1, L - 2, (1,), generator=g))
            q = int(torch.randint(p + 1, L, (1,), generator=g))
            row = torch.randint(0, 10, (L,), generator=g)
            row[:q] = torch.randint(0, M + 1, (q,), generator=g)
            row[int(torch.randint(0, p, (1,), generator=g))] = M
            row[q] = M + 1
            clean = row.clone(); clean[p] = M + 1
            corrupt = row.clone(); corrupt[p] = M
            cleans.append(clean); corrupts.append(corrupt)
        return torch.stack(cleans), torch.stack(corrupts)

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        p = _edit_pos(clean_row, corrupt_row); v = int(clean_row[p])
        after = (clean_row[p + 1:] >= v).nonzero().flatten()
        q = int(after[0]) + p + 1 if after.numel() else self.input_length
        return _mask(self.input_length, slice(p, q))


def _next_token_after(row: torch.Tensor, p: int, pred) -> int:
    """First index > p where ``pred(row[i])`` holds, else len(row) (window resync point)."""
    hits = pred(row[p + 1:]).nonzero().flatten()
    return int(hits[0]) + p + 1 if hits.numel() else int(row.shape[0])


class Reset3ResetToggleCircuit(ResetRamp, ContrastCircuit):
    """reset3_reset_toggle on reset3_id: reset token 3 in vs out at p; the ramp offsets until the next 3 -> {p..q-1}."""
    circuit_id = 'reset3_reset_toggle'
    algorithm = 'reset3_id'
    base_circuit = 'reset3_singleramp'
    _cond = conditions.FULL
    _NON3 = torch.tensor([0, 1, 2, 4, 5, 6, 7, 8, 9])

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        L = self.input_length
        g = torch.Generator().manual_seed(int(seed))
        cleans, corrupts = [], []
        for _ in range(batch):
            p = int(torch.randint(1, L - 2, (1,), generator=g))
            q = int(torch.randint(p + 1, L, (1,), generator=g))
            row = self._NON3[torch.randint(0, 9, (L,), generator=g)]
            clean = row.clone(); clean[p] = 3; clean[q] = 3
            corrupt = row.clone(); corrupt[q] = 3
            cleans.append(clean); corrupts.append(corrupt)
        return torch.stack(cleans), torch.stack(corrupts)

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        p = _edit_pos(clean_row, corrupt_row)
        return _mask(self.input_length, slice(p, _next_token_after(clean_row, p, lambda x: x == 3)))


class Tresetmax3ResetToggleCircuit(TokenResetMax, ContrastCircuit):
    """tresetmax3_reset_toggle on tresetmax3_id: reset token 3 vs a c<=prefix-max at p; max diverges until the next 3 -> {p..q-1}."""
    circuit_id = 'tresetmax3_reset_toggle'
    algorithm = 'tresetmax3_id'
    base_circuit = 'tresetmax3_runmax'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        L = self.input_length
        g = torch.Generator().manual_seed(int(seed))
        cleans, corrupts = [], []
        for _ in range(batch):
            p = int(torch.randint(1, L - 2, (1,), generator=g))
            q = int(torch.randint(p + 1, L, (1,), generator=g))
            R = int(torch.randint(4, 10, (1,), generator=g))          # prefix reset-max
            row = torch.randint(1, 3, (L,), generator=g)              # {1,2} baseline (< R)
            row[int(torch.randint(0, p, (1,), generator=g))] = R
            clean = row.clone(); clean[p] = 3
            corrupt = row.clone(); corrupt[p] = int(torch.randint(1, 3, (1,), generator=g))
            clean[q] = 3; corrupt[q] = 3
            cleans.append(clean); corrupts.append(corrupt)
        return torch.stack(cleans), torch.stack(corrupts)

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        p = _edit_pos(clean_row, corrupt_row)
        return _mask(self.input_length, slice(p, _next_token_after(clean_row, p, lambda x: x == 3)))


class Holdge5ThresholdStraddleCircuit(SampleHold, ContrastCircuit):
    """holdge5_threshold_straddle on holdge5_id: qualifier 5 vs 4 at p (held reg 0!=5); latch diverges until the next >=5 -> {p..q-1}."""
    circuit_id = 'holdge5_threshold_straddle'
    algorithm = 'holdge5_id'
    base_circuit = 'holdge5_onehold'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        L = self.input_length
        g = torch.Generator().manual_seed(int(seed))
        cleans, corrupts = [], []
        for _ in range(batch):
            p = int(torch.randint(0, L - 2, (1,), generator=g))
            q = int(torch.randint(p + 1, L, (1,), generator=g))
            row = torch.randint(0, 5, (L,), generator=g)              # <5 baseline (non-qualifying)
            clean = row.clone(); clean[p] = 5
            corrupt = row.clone(); corrupt[p] = 4
            qual = int(torch.randint(5, 10, (1,), generator=g))
            clean[q] = qual; corrupt[q] = qual
            cleans.append(clean); corrupts.append(corrupt)
        return torch.stack(cleans), torch.stack(corrupts)

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        p = _edit_pos(clean_row, corrupt_row)
        return _mask(self.input_length, slice(p, _next_token_after(clean_row, p, lambda x: x >= 5)))


class Dep9f0ClipToggleCircuit(Deplete, ContrastCircuit):
    """dep9f0_clip_toggle on dep9f0_id: at full acc, deplete 9 (clips to 0) vs 7 (lands on 2); 0-run holds the gap until a clip at q -> {p..q-1}."""
    circuit_id = 'dep9f0_clip_toggle'
    algorithm = 'dep9f0_id'
    base_circuit = 'dep9f0_ramp'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        L = self.input_length
        g = torch.Generator().manual_seed(int(seed))
        cleans, corrupts = [], []
        for _ in range(batch):
            p = int(torch.randint(1, L - 3, (1,), generator=g))
            q = int(torch.randint(p + 1, L - 1, (1,), generator=g))
            row = torch.zeros(L, dtype=torch.long)                   # prefix/window 0 keeps acc pinned
            row[q] = int(torch.randint(2, 10, (1,), generator=g))    # >=2 clips the corrupt side to 0
            row[q + 1:] = torch.randint(0, 10, (L - q - 1,), generator=g)
            clean = row.clone(); clean[p] = 9
            corrupt = row.clone(); corrupt[p] = 7
            cleans.append(clean); corrupts.append(corrupt)
        return torch.stack(cleans), torch.stack(corrupts)

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        p = _edit_pos(clean_row, corrupt_row)
        return _mask(self.input_length, slice(p, _next_token_after(clean_row, p, lambda x: x != 0)))


class PmstepParityToggleCircuit(ParityRoute, ContrastCircuit):
    """pmstep_parity_toggle on pmstep_id: even v <-> odd v+1 at p; the parity-routed map is a bijection so the mask is exactly {p}."""
    circuit_id = 'pmstep_parity_toggle'
    algorithm = 'pmstep_id'
    base_circuit = 'pmstep_evenadd'
    _cond = conditions.FULL
    _EVEN = torch.tensor([0, 2, 4, 6, 8])

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(0, self.input_length, (batch,), generator=g); rows = torch.arange(batch)
        clean[rows, p] = self._EVEN[torch.randint(0, 5, (batch,), generator=g)]
        corrupt = clean.clone(); corrupt[rows, p] = clean[rows, p] + 1
        return clean, corrupt

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        return _mask(self.input_length, _edit_pos(clean_row, corrupt_row))


class FixedpointHitToggleCircuit(FixedPointIndicator, ContrastCircuit):
    """fixedpoint_hit_toggle on fixedpoint_id: x[p]=p (hit) vs p+-1 (miss); positionwise indicator -> mask {p}."""
    circuit_id = 'fixedpoint_hit_toggle'
    algorithm = 'fixedpoint_id'
    base_circuit = 'fixedpoint_onehit'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(0, self.input_length, (batch,), generator=g); rows = torch.arange(batch)
        clean[rows, p] = p
        corrupt = clean.clone()
        corrupt[rows, p] = torch.where(p < self.input_length - 1, p + 1, p - 1)
        return clean, corrupt

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        return _mask(self.input_length, _edit_pos(clean_row, corrupt_row))


class NovelRepeatToggleCircuit(NoveltyIndicator, ContrastCircuit):
    """novel_repeat_toggle on novel_id: seen value (repeat) vs globally-absent 9 (novel) at p; absent elsewhere -> mask {p}."""
    circuit_id = 'novel_repeat_toggle'
    algorithm = 'novel_id'
    base_circuit = 'novel_onerepeat'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        L = self.input_length
        g = torch.Generator().manual_seed(int(seed))
        cleans, corrupts = [], []
        for _ in range(batch):
            row = torch.randint(0, 9, (L,), generator=g)             # digits 0..8; 9 is globally absent
            p = int(torch.randint(1, L, (1,), generator=g))
            j = int(torch.randint(0, p, (1,), generator=g))
            clean = row.clone(); clean[p] = clean[j]                 # repeat of an earlier value
            corrupt = clean.clone(); corrupt[p] = 9                  # novel (absent) value
            cleans.append(clean); corrupts.append(corrupt)
        return torch.stack(cleans), torch.stack(corrupts)

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        return _mask(self.input_length, _edit_pos(clean_row, corrupt_row))


class Bmax1LocalStraddleCircuit(WindowMax, ContrastCircuit):
    """bmax1_local_straddle on bmax1_id: window-winner M+1 vs M at p, with x[p+1]>=M+1 frozen; only out[p] moves -> mask {p}."""
    circuit_id = 'bmax1_local_straddle'
    algorithm = 'bmax1_id'
    base_circuit = 'bmax1_delay'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        L = self.input_length
        g = torch.Generator().manual_seed(int(seed))
        cleans, corrupts = [], []
        for _ in range(batch):
            p = int(torch.randint(1, L - 1, (1,), generator=g))
            M = int(torch.randint(1, 9, (1,), generator=g))
            row = torch.randint(0, 10, (L,), generator=g)
            row[p - 1] = int(torch.randint(0, M, (1,), generator=g))          # < M (window loses to x[p])
            row[p + 1] = int(torch.randint(M + 1, 10, (1,), generator=g))     # >= M+1 freezes out[p+1]
            clean = row.clone(); clean[p] = M + 1
            corrupt = row.clone(); corrupt[p] = M
            cleans.append(clean); corrupts.append(corrupt)
        return torch.stack(cleans), torch.stack(corrupts)

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        return _mask(self.input_length, _edit_pos(clean_row, corrupt_row))


class Anc0AnchorPerturbCircuit(AnchorSubtract, ContrastCircuit):
    """anc0_anchor_perturb on anc0_id: shift the anchor x[0] by +-1; every relative output moves except out[0]=0 -> {1..9}."""
    circuit_id = 'anc0_anchor_perturb'
    algorithm = 'anc0_id'
    base_circuit = 'anc0_caesar'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        corrupt = clean.clone()
        a = clean[:, 0]
        corrupt[:, 0] = torch.where(a < 9, a + 1, a - 1)
        return clean, corrupt

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        return _mask(self.input_length, slice(1, None))


class Fclamp0AnchorPerturbCircuit(AnchorClamp, ContrastCircuit):
    """fclamp0_anchor_perturb on fclamp0_id: raise the floor anchor x[0]=b by 1; mask = {0} U {i: x[i]<=b}."""
    circuit_id = 'fclamp0_anchor_perturb'
    algorithm = 'fclamp0_id'
    base_circuit = 'fclamp0_clamp'
    local_predicate = False   # the anchor bound feeds every clamped position
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        clean[:, 0] = torch.randint(0, 9, (batch,), generator=g)     # b in [0,8] so b+1 is a digit
        corrupt = clean.clone(); corrupt[:, 0] = clean[:, 0] + 1
        return clean, corrupt

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        return clean_row <= int(clean_row[0])


class Carryk1CarryGateCircuit(RippleCarry, ContrastCircuit):
    """carryk1_carry_gate on carryk1_id: in a trailing 9-run, x[j]=9 (carry propagates) vs 8 (absorbs); mask exactly {j-1, j}."""
    circuit_id = 'carryk1_carry_gate'
    algorithm = 'carryk1_id'
    base_circuit = 'carryk1_local'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        L = self.input_length
        g = torch.Generator().manual_seed(int(seed))
        cleans, corrupts = [], []
        for _ in range(batch):
            j = int(torch.randint(1, L, (1,), generator=g))
            row = torch.randint(0, 10, (L,), generator=g)
            row[j + 1:] = 9                                          # trailing 9-run to carry through
            row[j - 1] = int(torch.randint(0, 9, (1,), generator=g)) # <=8 stops the carry
            clean = row.clone(); clean[j] = 9
            corrupt = row.clone(); corrupt[j] = 8
            cleans.append(clean); corrupts.append(corrupt)
        return torch.stack(cleans), torch.stack(corrupts)

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        j = _edit_pos(clean_row, corrupt_row)
        return _mask(self.input_length, [j - 1, j])


class ArgminmaxMaxSwapCircuit(ArgMinMax, ContrastCircuit):
    """argminmax_max_swap on argminmax_id: swap the unique max (9) and min (0); both indicators move -> {p,q}u{p+10,q+10}."""
    circuit_id = 'argminmax_max_swap'
    algorithm = 'argminmax_id'
    base_circuit = 'argminmax_argmax'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        L = self.input_length
        g = torch.Generator().manual_seed(int(seed))
        cleans, corrupts = [], []
        for _ in range(batch):
            perm = torch.randperm(L, generator=g)
            p, q = int(perm[0]), int(perm[1])
            row = torch.randint(1, 9, (L,), generator=g)             # 1..8 -> 9 and 0 stay unique extrema
            row[p] = 9; row[q] = 0
            clean = row.clone()
            corrupt = row.clone(); corrupt[p], corrupt[q] = 0, 9
            cleans.append(clean); corrupts.append(corrupt)
        return torch.stack(cleans), torch.stack(corrupts)

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        L = self.input_length
        idx = (clean_row != corrupt_row).nonzero().flatten().tolist()
        return _mask(2 * L, idx + [i + L for i in idx])


class Segrev3SecondBoundaryToggleCircuit(SegmentReverse, ContrastCircuit):
    """segrev3_second_boundary_toggle on segrev3_id: keep a 3 at p, toggle the 3 at q>p off; the kept boundary fences 0..p -> {p+1..9}."""
    circuit_id = 'segrev3_second_boundary_toggle'
    algorithm = 'segrev3_id'
    base_circuit = 'reverse_only'
    local_predicate = False   # segment-membership past the kept boundary feeds the whole tail
    _cond = conditions.FULL
    _NON3 = torch.tensor([0, 1, 2, 4, 5, 6, 7, 8, 9])

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        L = self.input_length
        g = torch.Generator().manual_seed(int(seed))
        cleans, corrupts, tries = [], [], 0
        cap = batch * REJECTION_BUDGET_PER_ROW
        while len(cleans) < batch:
            tries += 1
            p = int(torch.randint(1, L // 2, (1,), generator=g))
            q = int(torch.randint(p + 1, L, (1,), generator=g))
            row = self._NON3[torch.randint(0, 9, (L,), generator=g)]
            clean = row.clone(); clean[p] = 3; clean[q] = 3
            corrupt = clean.clone(); corrupt[q] = int(self._NON3[torch.randint(0, 9, (1,), generator=g)])
            diff = self.eval(clean) != self.eval(corrupt)
            if torch.equal(diff, _mask(L, slice(p + 1, None))):       # keep only the exact fenced mask
                cleans.append(clean); corrupts.append(corrupt)
            if tries > cap:
                raise CircuitError(f"{self.circuit_id}: fenced-mask accept-rate too low "
                                   f"({len(cleans)}/{batch} after {tries} tries)")
        return torch.stack(cleans), torch.stack(corrupts)

    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor:
        p = int((corrupt_row == 3).nonzero()[0])                     # corrupt keeps only the anchor 3
        return _mask(self.input_length, slice(p + 1, None))


class Drot0AmountPerturbCircuit(DataRotate, ContrastCircuit):
    """drot0_amount_perturb on drot0_id: rotation amount x[0] r <-> r+1 (anchor is double-duty data); near-full, mask unclaimed."""
    circuit_id = 'drot0_amount_perturb'
    algorithm = 'drot0_id'
    base_circuit = 'drot0_rotate'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        return _rejection_pairs(self, batch, seed, _anchor_edit(0, 8))


class Refl0PivotPerturbCircuit(DataReflect, ContrastCircuit):
    """refl0_pivot_perturb on refl0_id: mirror pivot x[0] p <-> p+1; near-full with fixed-point holes, mask unclaimed."""
    circuit_id = 'refl0_pivot_perturb'
    algorithm = 'refl0_id'
    base_circuit = 'refl0_mirror'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        return _rejection_pairs(self, batch, seed, _anchor_edit(0, 8))


class Stride0StridePerturbCircuit(StrideGather, ContrastCircuit):
    """stride0_stride_perturb on stride0_id: gather stride x[0] k <-> k+1 (regimes differ by k); confounded, mask unclaimed."""
    circuit_id = 'stride0_stride_perturb'
    algorithm = 'stride0_id'
    base_circuit = 'stride0_gather'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        return _rejection_pairs(self, batch, seed, _anchor_edit(0, 8))


class Pref0KPerturbCircuit(PrefixReverseK, ContrastCircuit):
    """pref0_k_perturb on pref0_id: prefix-reverse length x[0] k <-> k+1 (k in 2..8); mask ~ {0..k}, unclaimed."""
    circuit_id = 'pref0_k_perturb'
    algorithm = 'pref0_id'
    base_circuit = 'pref0_revk'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        return _rejection_pairs(self, batch, seed, _anchor_edit(2, 8))


class Psort0RankMoveCircuit(PrefixSortK, ContrastCircuit):
    """psort0_rank_move on psort0_id: pin sort length k=x[0] in [3,9], perturb one prefix payload so its rank moves; mask unclaimed."""
    circuit_id = 'psort0_rank_move'
    algorithm = 'psort0_id'
    base_circuit = 'psort0_sortk'
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        def edit(row: torch.Tensor, g: torch.Generator) -> torch.Tensor:
            row[0] = int(torch.randint(3, 10, (1,), generator=g))
            j = int(torch.randint(1, int(row[0]), (1,), generator=g))
            corrupt = row.clone(); corrupt[j] = int(torch.randint(0, 10, (1,), generator=g))
            return corrupt
        return _rejection_pairs(self, batch, seed, edit)


class CjumpBranchStraddleCircuit(ConditionalJump, ContrastCircuit):
    """cjump_branch_straddle on cjump10t5_id: at a cell, 6 (forward) vs 5 (backward) straddles the branch; trajectory-confounded, mask unclaimed."""
    circuit_id = 'cjump_branch_straddle'
    algorithm = 'cjump10t5_id'
    base_circuit = 'cjump_forward'
    local_predicate = False   # the branch's only consequence is the whole downstream trajectory
    _cond = conditions.FULL

    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        L = self.input_length
        def edit(row: torch.Tensor, g: torch.Generator) -> torch.Tensor:
            c = int(torch.randint(0, L, (1,), generator=g)); row[c] = 6
            corrupt = row.clone(); corrupt[c] = 5
            return corrupt
        return _rejection_pairs(self, batch, seed, edit)


class Pmaxv9No9Circuit(Pmaxv9, DomainCircuit):
    """pmaxv9_no9 on pmaxv9_id: isolated via exclude_digit(9) (integrated)."""
    circuit_id = 'pmaxv9_no9'
    algorithm = 'pmaxv9_id'
    _cond = conditions.exclude_digit(9)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Pmaxv9One9Circuit(Pmaxv9, DomainCircuit):
    """pmaxv9_one9 on pmaxv9_id: isolated via exactly_one_in([9]) (integrated)."""
    circuit_id = 'pmaxv9_one9'
    algorithm = 'pmaxv9_id'
    _cond = conditions.exactly_one_in([9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Pmaxv9DisqualifyCircuit(Pmaxv9, ContrastCircuit):
    """pmaxv9_disqualify on pmaxv9_id: contrast vs pmaxv9_no9 [threshold-straddle] (integrated)."""
    circuit_id = 'pmaxv9_disqualify'
    algorithm = 'pmaxv9_id'
    base_circuit = 'pmaxv9_no9'
    local_predicate = False
    _cond = conditions.exclude_digit(9)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 9, (batch, self.input_length), generator=g)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        rows = torch.arange(batch)
        before = torch.arange(self.input_length).unsqueeze(0) < p.unsqueeze(1)
        clean = torch.where(before, clean.clamp(max=7), clean)
        clean[rows, p] = 8
        corrupt = clean.clone()
        corrupt[rows, p] = 9
        return clean, corrupt


class RwGe1recCircuit(DoubleOdd, DomainCircuit):
    """rw_ge1rec on dblodd_walk10_recordind_id: isolated via Condition('ge1rec', lambda x: bool((x[1:] > torch.cummax(x, 0).values[:-1]).any()), lambda batch, seed: (lambda blk: blk[[bool((r[1:] > torch.cummax(r, 0).values[:-1]).any()) for r in blk]][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'rw_ge1rec'))))) (integrated)."""
    circuit_id = 'rw_ge1rec'
    algorithm = 'dblodd_walk10_recordind_id'
    _cond = conditions.Condition('ge1rec', lambda x: bool((x[1:] > torch.cummax(x, 0).values[:-1]).any()), lambda batch, seed: (lambda blk: blk[[bool((r[1:] > torch.cummax(r, 0).values[:-1]).any()) for r in blk]][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'rw_ge1rec')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class RwMakerecCircuit(DoubleOdd, ContrastCircuit):
    """rw_makerec on dblodd_walk10_recordind_id: contrast vs rw_ge1rec [threshold-straddle] (integrated)."""
    circuit_id = 'rw_makerec'
    algorithm = 'dblodd_walk10_recordind_id'
    base_circuit = 'rw_ge1rec'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            for i in idx.tolist():
                Ms = torch.cummax(clean[i], 0).values
                opts = [j for j in range(1, L) if int(clean[i, j]) <= int(Ms[j - 1]) < 9]
                if not opts:
                    clean[i] = torch.randint(0, 10, (L,), generator=g)
                    continue
                j = opts[int(torch.randint(0, len(opts), (1,), generator=g))]
                Mb = int(Ms[j - 1])
                corrupt[i] = clean[i]
                corrupt[i, j] = Mb + 1 + int(torch.randint(0, 9 - Mb, (1,), generator=g))
                need[i] = False
        assert not int(need.any())
        return clean, corrupt


class RwBreakrecCircuit(DoubleOdd, ContrastCircuit):
    """rw_breakrec on dblodd_walk10_recordind_id: contrast vs rw_ge1rec [threshold-straddle] (integrated)."""
    circuit_id = 'rw_breakrec'
    algorithm = 'dblodd_walk10_recordind_id'
    base_circuit = 'rw_ge1rec'
    local_predicate = False
    _cond = conditions.Condition('ge1rec', lambda x: bool((x[1:] > torch.cummax(x, 0).values[:-1]).any()), lambda batch, seed: (lambda blk: blk[[bool((r[1:] > torch.cummax(r, 0).values[:-1]).any()) for r in blk]][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'rw_breakrec')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        for i in range(batch):
            Ms = torch.cummax(clean[i], 0).values
            recs = [j for j in range(1, L) if int(clean[i, j]) > int(Ms[j - 1])]
            j = recs[int(torch.randint(0, len(recs), (1,), generator=g))]
            corrupt[i] = clean[i]
            corrupt[i, j] = int(torch.randint(0, int(Ms[j - 1]) + 1, (1,), generator=g))
        return clean, corrupt


class RwUpstreamCircuit(DoubleOdd, ContrastCircuit):
    """rw_upstream on dblodd_walk10_recordind_id: contrast vs rw_ge1rec [threshold-straddle] (integrated)."""
    circuit_id = 'rw_upstream'
    algorithm = 'dblodd_walk10_recordind_id'
    base_circuit = 'rw_ge1rec'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            for i in idx.tolist():
                Ms = torch.cummax(clean[i], 0).values
                done = False
                for _t in range(8):
                    q = 1 + int(torch.randint(0, L - 3, (1,), generator=g))
                    Mb = int(Ms[q - 1])
                    later = [int(clean[i, j]) for j in range(q + 1, L) if int(clean[i, j]) > int(Ms[j - 1])]
                    hiv = min(min(later) - 1 if later else 9, 9)
                    if int(clean[i, q]) <= Mb and Mb + 1 <= hiv:
                        corrupt[i] = clean[i]
                        corrupt[i, q] = Mb + 1 + int(torch.randint(0, hiv - Mb, (1,), generator=g))
                        need[i] = False
                        done = True
                        break
                if not done:
                    clean[i] = torch.randint(0, 10, (L,), generator=g)
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) >= int((clean_row != corrupt_row).nonzero()[0])


class Dfa3FullCircuit(SmallDFATrace, DomainCircuit):
    """dfa3_full on dfa3t010020001_id: isolated via FULL (integrated)."""
    circuit_id = 'dfa3_full'
    algorithm = 'dfa3t010020001_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Dfa3LowmidCircuit(SmallDFATrace, DomainCircuit):
    """dfa3_lowmid on dfa3t010020001_id: isolated via domain([0, 1, 2, 3, 4, 5, 6]) (integrated)."""
    circuit_id = 'dfa3_lowmid'
    algorithm = 'dfa3t010020001_id'
    _cond = conditions.domain([0, 1, 2, 3, 4, 5, 6])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Dfa3NolowCircuit(SmallDFATrace, DomainCircuit):
    """dfa3_nolow on dfa3t010020001_id: isolated via domain([3, 4, 5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'dfa3_nolow'
    algorithm = 'dfa3t010020001_id'
    _cond = conditions.domain([3, 4, 5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Dfa3ClassflipCircuit(SmallDFATrace, ContrastCircuit):
    """dfa3_classflip on dfa3t010020001_id: contrast vs dfa3_full [digit-toggle] (integrated)."""
    circuit_id = 'dfa3_classflip'
    algorithm = 'dfa3t010020001_id'
    base_circuit = 'dfa3_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        cl = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            p = torch.randint(0, L, (idx.numel(),), generator=g)
            new = torch.randint(0, 10, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                if int(cl[new[k]]) == int(cl[clean[i, p[k]]]):
                    continue
                corrupt[i] = clean[i]
                corrupt[i, p[k]] = new[k]
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Dfa3ResetinjectCircuit(SmallDFATrace, ContrastCircuit):
    """dfa3_resetinject on dfa3t010020001_id: contrast vs dfa3_full [digit-toggle] (integrated)."""
    circuit_id = 'dfa3_resetinject'
    algorithm = 'dfa3t010020001_id'
    base_circuit = 'dfa3_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            p = torch.randint(0, L, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                if int((clean[i] >= 3).sum()) == 0:
                    clean[i] = torch.randint(0, 10, (L,), generator=g)
                    continue
                if int(clean[i, p[k]]) < 3:
                    continue
                corrupt[i] = clean[i]
                corrupt[i, p[k]] = int(torch.randint(0, 3, (1,), generator=g))
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class IndechoAlldistinctCircuit(Indecho, DomainCircuit):
    """indecho_alldistinct on indecho_id: isolated via Condition('all_distinct', lambda x: len(set(x.tolist())) == 10, lambda batch, seed: torch.argsort(torch.rand(batch, 10, generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'indecho_alldistinct'))), dim=1)) (integrated)."""
    circuit_id = 'indecho_alldistinct'
    algorithm = 'indecho_id'
    _cond = conditions.Condition('all_distinct', lambda x: len(set(x.tolist())) == 10, lambda batch, seed: torch.argsort(torch.rand(batch, 10, generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'indecho_alldistinct'))), dim=1))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class IndechoOnerepeatCircuit(Indecho, DomainCircuit):
    """indecho_onerepeat on indecho_id: isolated via ONE_REPEAT (integrated)."""
    circuit_id = 'indecho_onerepeat'
    algorithm = 'indecho_id'
    _cond = conditions.ONE_REPEAT
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class IndechoMatchbreakCircuit(Indecho, ContrastCircuit):
    """indecho_matchbreak on indecho_id: contrast vs indecho_alldistinct [digit-toggle] (integrated)."""
    circuit_id = 'indecho_matchbreak'
    algorithm = 'indecho_id'
    base_circuit = 'indecho_alldistinct'
    local_predicate = True
    _cond = conditions.ONE_REPEAT
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        for i in range(batch):
            for k in range(64):
                vals = clean[i].tolist()
                dup = next(v for v in vals if vals.count(v) == 2)
                j1 = vals.index(dup)
                j2 = 9 - vals[::-1].index(dup)
                if j2 - j1 >= 2:
                    break
                clean[i] = DomainCircuit.sample(self, 1, int(seed) * 641 + i * 64 + k + 1)[0]
            assert j2 - j1 >= 2
            fresh = [v for v in range(10) if v not in vals]
            corrupt[i] = clean[i]
            corrupt[i, j1] = fresh[int(torch.randint(0, len(fresh), (1,), generator=g))]
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        j1 = int((clean_row != corrupt_row).nonzero()[0])
        v = int(clean_row[j1])
        m = torch.zeros(10, dtype=torch.bool)
        m[j1] = True
        for j in range(j1 + 1, 10):
            if int(clean_row[j]) == v:
                m[j] = True
        return m


class CosetprodFullCircuit(GroupCosetProductS3, DomainCircuit):
    """cosetprod_full on cosetprod_id: isolated via FULL (integrated)."""
    circuit_id = 'cosetprod_full'
    algorithm = 'cosetprod_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CosetprodEditCircuit(GroupCosetProductS3, ContrastCircuit):
    """cosetprod_edit on cosetprod_id: contrast vs cosetprod_full [digit-toggle] (integrated)."""
    circuit_id = 'cosetprod_edit'
    algorithm = 'cosetprod_id'
    base_circuit = 'cosetprod_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Kanc2Class0Circuit(Kanc2, DomainCircuit):
    """kanc2_class0 on kanc2_id: isolated via ALL_EVEN (integrated)."""
    circuit_id = 'kanc2_class0'
    algorithm = 'kanc2_id'
    _cond = conditions.ALL_EVEN
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Kanc2RegimeflipCircuit(Kanc2, ContrastCircuit):
    """kanc2_regimeflip on kanc2_id: contrast vs kanc2_class0 [digit-toggle] (integrated)."""
    circuit_id = 'kanc2_regimeflip'
    algorithm = 'kanc2_id'
    base_circuit = 'kanc2_class0'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        M = torch.tensor([3, 5, 7, 9, 1, 3, 7, 5, 9, 1]); B = torch.tensor([1, 2, 5, 0, 3, 8, 6, 4, 9, 7])
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(1, self.input_length, (batch,), generator=g)
        rows = torch.arange(batch)
        new = torch.randint(0, 10, (batch,), generator=g)
        for _ in range(64):
            old, x0 = clean[rows, p], clean[:, 0]
            yo = (old + M[old % 2] * x0 + B[old % 2]) % 10
            yn = (new + M[new % 2] * x0 + B[new % 2]) % 10
            bad = ((new % 2) == (old % 2)) | (yo == yn)
            if not int(bad.any()): break
            new[bad] = torch.randint(0, 10, (int(bad.sum()),), generator=g)
        assert not int(bad.any())
        corrupt = clean.clone()
        corrupt[rows, p] = new
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return clean_row != corrupt_row


class Kanc2InclassCircuit(Kanc2, ContrastCircuit):
    """kanc2_inclass on kanc2_id: contrast vs kanc2_class0 [value-perturb-control] (integrated)."""
    circuit_id = 'kanc2_inclass'
    algorithm = 'kanc2_id'
    base_circuit = 'kanc2_class0'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(1, self.input_length, (batch,), generator=g)
        rows = torch.arange(batch)
        old = clean[rows, p]
        new = (old + 2) % 10
        assert int(((new % 2) == (old % 2)).all()) and int((new != old).all())
        corrupt = clean.clone()
        corrupt[rows, p] = new
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return clean_row != corrupt_row


class Kanc2AnchorshiftCircuit(Kanc2, ContrastCircuit):
    """kanc2_anchorshift on kanc2_id: contrast vs kanc2_class0 [digit-toggle] (integrated)."""
    circuit_id = 'kanc2_anchorshift'
    algorithm = 'kanc2_id'
    base_circuit = 'kanc2_class0'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        corrupt = clean.clone()
        corrupt[:, 0] = (clean[:, 0] + 3) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.ones(10, dtype=torch.bool)


class Kanc4Class0Circuit(Kanc4, DomainCircuit):
    """kanc4_class0 on kanc4_id: isolated via domain([0, 4, 8]) (integrated)."""
    circuit_id = 'kanc4_class0'
    algorithm = 'kanc4_id'
    _cond = conditions.domain([0, 4, 8])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Kanc4RegimeflipCircuit(Kanc4, ContrastCircuit):
    """kanc4_regimeflip on kanc4_id: contrast vs kanc4_class0 [digit-toggle] (integrated)."""
    circuit_id = 'kanc4_regimeflip'
    algorithm = 'kanc4_id'
    base_circuit = 'kanc4_class0'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        M = torch.tensor([3, 5, 7, 9, 1, 3, 7, 5, 9, 1]); B = torch.tensor([1, 2, 5, 0, 3, 8, 6, 4, 9, 7])
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(1, self.input_length, (batch,), generator=g)
        rows = torch.arange(batch)
        new = torch.randint(0, 10, (batch,), generator=g)
        for _ in range(64):
            old, x0 = clean[rows, p], clean[:, 0]
            yo = (old + M[old % 4] * x0 + B[old % 4]) % 10
            yn = (new + M[new % 4] * x0 + B[new % 4]) % 10
            bad = ((new % 4) == (old % 4)) | (yo == yn)
            if not int(bad.any()): break
            new[bad] = torch.randint(0, 10, (int(bad.sum()),), generator=g)
        assert not int(bad.any())
        corrupt = clean.clone()
        corrupt[rows, p] = new
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return clean_row != corrupt_row


class Kanc4InclassCircuit(Kanc4, ContrastCircuit):
    """kanc4_inclass on kanc4_id: contrast vs kanc4_class0 [value-perturb-control] (integrated)."""
    circuit_id = 'kanc4_inclass'
    algorithm = 'kanc4_id'
    base_circuit = 'kanc4_class0'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(1, self.input_length, (batch,), generator=g)
        rows = torch.arange(batch)
        old = clean[rows, p]
        new = torch.where(old < 6, old + 4, old - 4)
        assert int(((new % 4) == (old % 4)).all()) and int((new != old).all())
        corrupt = clean.clone()
        corrupt[rows, p] = new
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return clean_row != corrupt_row


class Kanc4AnchorshiftCircuit(Kanc4, ContrastCircuit):
    """kanc4_anchorshift on kanc4_id: contrast vs kanc4_class0 [digit-toggle] (integrated)."""
    circuit_id = 'kanc4_anchorshift'
    algorithm = 'kanc4_id'
    base_circuit = 'kanc4_class0'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        corrupt = clean.clone()
        corrupt[:, 0] = (clean[:, 0] + 3) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.ones(10, dtype=torch.bool)


class Kanc8Class0Circuit(Kanc8, DomainCircuit):
    """kanc8_class0 on kanc8_id: isolated via domain([0, 8]) (integrated)."""
    circuit_id = 'kanc8_class0'
    algorithm = 'kanc8_id'
    _cond = conditions.domain([0, 8])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Kanc8RegimeflipCircuit(Kanc8, ContrastCircuit):
    """kanc8_regimeflip on kanc8_id: contrast vs kanc8_class0 [digit-toggle] (integrated)."""
    circuit_id = 'kanc8_regimeflip'
    algorithm = 'kanc8_id'
    base_circuit = 'kanc8_class0'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        M = torch.tensor([3, 5, 7, 9, 1, 3, 7, 5, 9, 1]); B = torch.tensor([1, 2, 5, 0, 3, 8, 6, 4, 9, 7])
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(1, self.input_length, (batch,), generator=g)
        rows = torch.arange(batch)
        new = torch.randint(0, 10, (batch,), generator=g)
        for _ in range(64):
            old, x0 = clean[rows, p], clean[:, 0]
            yo = (old + M[old % 8] * x0 + B[old % 8]) % 10
            yn = (new + M[new % 8] * x0 + B[new % 8]) % 10
            bad = ((new % 8) == (old % 8)) | (yo == yn)
            if not int(bad.any()): break
            new[bad] = torch.randint(0, 10, (int(bad.sum()),), generator=g)
        assert not int(bad.any())
        corrupt = clean.clone()
        corrupt[rows, p] = new
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return clean_row != corrupt_row


class Kanc8InclassCircuit(Kanc8, ContrastCircuit):
    """kanc8_inclass on kanc8_id: contrast vs kanc8_class0 [value-perturb-control] (integrated)."""
    circuit_id = 'kanc8_inclass'
    algorithm = 'kanc8_id'
    base_circuit = 'kanc8_class0'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(1, self.input_length, (batch,), generator=g)
        rows = torch.arange(batch)
        pal = torch.tensor([0, 1, 8, 9])
        clean[rows, p] = pal[torch.randint(0, 4, (batch,), generator=g)]
        old = clean[rows, p]
        new = old ^ 8
        assert int(((new % 8) == (old % 8)).all()) and int((new != old).all())
        corrupt = clean.clone()
        corrupt[rows, p] = new
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return clean_row != corrupt_row


class Kanc8AnchorshiftCircuit(Kanc8, ContrastCircuit):
    """kanc8_anchorshift on kanc8_id: contrast vs kanc8_class0 [digit-toggle] (integrated)."""
    circuit_id = 'kanc8_anchorshift'
    algorithm = 'kanc8_id'
    base_circuit = 'kanc8_class0'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        corrupt = clean.clone()
        corrupt[:, 0] = (clean[:, 0] + 3) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.ones(10, dtype=torch.bool)


class Kanc10FullCircuit(Kanc10, DomainCircuit):
    """kanc10_full on kanc10_id: isolated via FULL (integrated)."""
    circuit_id = 'kanc10_full'
    algorithm = 'kanc10_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Kanc10RegimeflipCircuit(Kanc10, ContrastCircuit):
    """kanc10_regimeflip on kanc10_id: contrast vs kanc10_full [digit-toggle] (integrated)."""
    circuit_id = 'kanc10_regimeflip'
    algorithm = 'kanc10_id'
    base_circuit = 'kanc10_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        M = torch.tensor([3, 5, 7, 9, 1, 3, 7, 5, 9, 1]); B = torch.tensor([1, 2, 5, 0, 3, 8, 6, 4, 9, 7])
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(1, self.input_length, (batch,), generator=g)
        rows = torch.arange(batch)
        new = torch.randint(0, 10, (batch,), generator=g)
        for _ in range(64):
            old, x0 = clean[rows, p], clean[:, 0]
            yo = (old + M[old % 10] * x0 + B[old % 10]) % 10
            yn = (new + M[new % 10] * x0 + B[new % 10]) % 10
            bad = ((new % 10) == (old % 10)) | (yo == yn)
            if not int(bad.any()): break
            new[bad] = torch.randint(0, 10, (int(bad.sum()),), generator=g)
        assert not int(bad.any())
        corrupt = clean.clone()
        corrupt[rows, p] = new
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return clean_row != corrupt_row


class Kanc10AnchorshiftCircuit(Kanc10, ContrastCircuit):
    """kanc10_anchorshift on kanc10_id: contrast vs kanc10_full [digit-toggle] (integrated)."""
    circuit_id = 'kanc10_anchorshift'
    algorithm = 'kanc10_id'
    base_circuit = 'kanc10_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        corrupt = clean.clone()
        corrupt[:, 0] = (clean[:, 0] + 3) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.ones(10, dtype=torch.bool)


class Sanc2AllevenCircuit(Sanc2, DomainCircuit):
    """sanc2_alleven on sanc2_id: isolated via ALL_EVEN (integrated)."""
    circuit_id = 'sanc2_alleven'
    algorithm = 'sanc2_id'
    _cond = conditions.ALL_EVEN
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Sanc2RegimeflipCircuit(Sanc2, ContrastCircuit):
    """sanc2_regimeflip on sanc2_id: contrast vs sanc2_alleven [digit-toggle] (integrated)."""
    circuit_id = 'sanc2_regimeflip'
    algorithm = 'sanc2_id'
    base_circuit = 'sanc2_alleven'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        corrupt = clean.clone()
        old = clean[rows, p]
        corrupt[rows, p] = torch.where(old % 2 == 0, old + 1, old - 1)
        for _ in range(64):
            bad = torch.tensor([torch.equal(self.eval(clean[i]), self.eval(corrupt[i]))
                                for i in range(batch)])
            if not int(bad.any()): break
            idx = bad.nonzero().flatten()
            p[idx] = torch.randint(0, self.input_length, (idx.numel(),), generator=g)
            corrupt[idx] = clean[idx]
            old = clean[idx, p[idx]]
            corrupt[idx, p[idx]] = torch.where(old % 2 == 0, old + 1, old - 1)
        assert not int(bad.any())
        return clean, corrupt


class Sanc2InparityCircuit(Sanc2, ContrastCircuit):
    """sanc2_inparity on sanc2_id: contrast vs sanc2_alleven [value-perturb-control] (integrated)."""
    circuit_id = 'sanc2_inparity'
    algorithm = 'sanc2_id'
    base_circuit = 'sanc2_alleven'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(1, self.input_length, (batch,), generator=g)
        rows = torch.arange(batch)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + 2) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return clean_row != corrupt_row


class T1Ge1matchCircuit(DoubleOdd, DomainCircuit):
    """t1_ge1match on dblodd_walk10_cmpsh1_id: isolated via Condition('ge1match', lambda x: bool((x[1:] == x[:-1]).any()), lambda batch, seed: (lambda blk: blk[[bool((r[1:] == r[:-1]).any()) for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 't1_ge1match'))))) (integrated)."""
    circuit_id = 't1_ge1match'
    algorithm = 'dblodd_walk10_cmpsh1_id'
    _cond = conditions.Condition('ge1match', lambda x: bool((x[1:] == x[:-1]).any()), lambda batch, seed: (lambda blk: blk[[bool((r[1:] == r[:-1]).any()) for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 't1_ge1match')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class T1MakematchCircuit(DoubleOdd, ContrastCircuit):
    """t1_makematch on dblodd_walk10_cmpsh1_id: contrast vs t1_ge1match [digit-toggle] (integrated)."""
    circuit_id = 't1_makematch'
    algorithm = 'dblodd_walk10_cmpsh1_id'
    base_circuit = 't1_ge1match'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(2, 9, (batch,), generator=g)
        for k in range(64):
            bad = (clean[rows, p] == clean[rows, p - 1])
            if not int(bad.any()): break
            p[bad] = torch.randint(2, 9, (int(bad.sum()),), generator=g)
            if k >= 8:
                clean[bad] = torch.randint(0, 10, (int(bad.sum()), self.input_length), generator=g)
        bad = (clean[rows, p] == clean[rows, p - 1])
        assert not int(bad.any())
        corrupt = clean.clone()
        corrupt[rows, p] = clean[rows, p - 1]
        return clean, corrupt


class T1UpstreamShiftCircuit(DoubleOdd, ContrastCircuit):
    """t1_upstream_shift on dblodd_walk10_cmpsh1_id: contrast vs t1_ge1match [digit-toggle] (integrated)."""
    circuit_id = 't1_upstream_shift'
    algorithm = 'dblodd_walk10_cmpsh1_id'
    base_circuit = 't1_ge1match'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(2, 8, (batch,), generator=g)
        for k in range(64):
            bad = (clean[rows, p] == clean[rows, p - 1]) | (clean[rows, p + 1] == clean[rows, p]) | (clean[rows, p + 1] == clean[rows, p - 1])
            if not int(bad.any()): break
            p[bad] = torch.randint(2, 8, (int(bad.sum()),), generator=g)
            if k >= 8:
                clean[bad] = torch.randint(0, 10, (int(bad.sum()), self.input_length), generator=g)
        bad = (clean[rows, p] == clean[rows, p - 1]) | (clean[rows, p + 1] == clean[rows, p]) | (clean[rows, p + 1] == clean[rows, p - 1])
        assert not int(bad.any())
        corrupt = clean.clone()
        corrupt[rows, p] = clean[rows, p - 1]
        return clean, corrupt


class Tmap5LowCircuit(Tmap5, DomainCircuit):
    """tmap5_low on tmap5_id: isolated via domain([0, 1, 2, 3, 4]) (integrated)."""
    circuit_id = 'tmap5_low'
    algorithm = 'tmap5_id'
    _cond = conditions.domain([0, 1, 2, 3, 4])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Tmap5HighCircuit(Tmap5, DomainCircuit):
    """tmap5_high on tmap5_id: isolated via domain([5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'tmap5_high'
    algorithm = 'tmap5_id'
    _cond = conditions.domain([5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Tmap5CrossThresholdCircuit(Tmap5, ContrastCircuit):
    """tmap5_cross_threshold on tmap5_id: contrast vs tmap5_low [threshold-straddle] (integrated)."""
    circuit_id = 'tmap5_cross_threshold'
    algorithm = 'tmap5_id'
    base_circuit = 'tmap5_low'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        rows = torch.arange(batch)
        clean[rows, p] = 4
        corrupt = clean.clone()
        corrupt[rows, p] = 5
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return clean_row != corrupt_row


class AddfixOnefpCircuit(Addfix, DomainCircuit):
    """addfix_onefp on addfix_id: isolated via ONE_FIXEDPOINT (integrated)."""
    circuit_id = 'addfix_onefp'
    algorithm = 'addfix_id'
    _cond = conditions.ONE_FIXEDPOINT
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class AddfixNofpCircuit(Addfix, DomainCircuit):
    """addfix_nofp on addfix_id: isolated via Condition('no_fixedpoint', lambda x: bool((x != torch.arange(10)).all()), conditions.position_domain_sampler({i: [v for v in range(10) if v != i] for i in range(10)})) (integrated)."""
    circuit_id = 'addfix_nofp'
    algorithm = 'addfix_id'
    _cond = conditions.Condition('no_fixedpoint', lambda x: bool((x != torch.arange(10)).all()), conditions.position_domain_sampler({i: [v for v in range(10) if v != i] for i in range(10)}))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class AddfixBinaryCircuit(Addfix, DomainCircuit):
    """addfix_binary on addfix_id: isolated via BINARY (integrated)."""
    circuit_id = 'addfix_binary'
    algorithm = 'addfix_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class AddfixMonodecCircuit(Addfix, DomainCircuit):
    """addfix_monodec on addfix_id: isolated via monotone() (integrated)."""
    circuit_id = 'addfix_monodec'
    algorithm = 'addfix_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class AddfixPlantCircuit(Addfix, ContrastCircuit):
    """addfix_plant on addfix_id: contrast vs addfix_nofp [digit-toggle] (integrated)."""
    circuit_id = 'addfix_plant'
    algorithm = 'addfix_id'
    base_circuit = 'addfix_nofp'
    local_predicate = False
    _cond = conditions.Condition('no_fixedpoint', lambda x: bool((x != torch.arange(10)).all()), conditions.position_domain_sampler({i: [v for v in range(10) if v != i] for i in range(10)}))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        clean = DomainCircuit.sample(self, batch, seed)
        g = torch.Generator().manual_seed(int(seed) + 1)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        for _ in range(64):
            bad = clean[rows, p] == (p + 1) % 10
            if not int(bad.any()): break
            p[bad] = torch.randint(0, self.input_length, (int(bad.sum()),), generator=g)
        assert not int((clean[rows, p] == (p + 1) % 10).any())
        corrupt = clean.clone()
        corrupt[rows, p] = p
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        return torch.arange(10) >= p


class AddfixSlideCircuit(Addfix, ContrastCircuit):
    """addfix_slide on addfix_id: contrast vs addfix_onefp [compensated-two-site] (integrated)."""
    circuit_id = 'addfix_slide'
    algorithm = 'addfix_id'
    base_circuit = 'addfix_onefp'
    local_predicate = True
    _cond = conditions.ONE_FIXEDPOINT
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        clean = DomainCircuit.sample(self, batch, seed)
        g = torch.Generator().manual_seed(int(seed) + 1)
        rows = torch.arange(batch)
        p0 = (clean == torch.arange(10)).long().argmax(dim=1)
        q = torch.randint(0, self.input_length, (batch,), generator=g)
        for _ in range(64):
            bad = (q == p0) | ((q < p0) & (clean[rows, q] == (q + 1) % 10))
            if not int(bad.any()): break
            q[bad] = torch.randint(0, self.input_length, (int(bad.sum()),), generator=g)
        assert not int(((q == p0) | ((q < p0) & (clean[rows, q] == (q + 1) % 10))).any())
        v = (p0 + 2 + torch.randint(0, 7, (batch,), generator=g)) % 10
        corrupt = clean.clone()
        corrupt[rows, p0] = v
        corrupt[rows, q] = q
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        d = (clean_row != corrupt_row).nonzero().flatten()
        return (torch.arange(10) >= int(d.min())) & (torch.arange(10) <= int(d.max()))


class Ca1FullCircuit(CellularAutomaton, DomainCircuit):
    """ca1_full on ca1a1b1c1_id: isolated via FULL (integrated)."""
    circuit_id = 'ca1_full'
    algorithm = 'ca1a1b1c1_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Ca1ImpulseCircuit(CellularAutomaton, DomainCircuit):
    """ca1_impulse on ca1a1b1c1_id: isolated via exactly_one_in([1, 2, 3, 4, 5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'ca1_impulse'
    algorithm = 'ca1a1b1c1_id'
    _cond = conditions.exactly_one_in([1, 2, 3, 4, 5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Ca1WindowhitCircuit(CellularAutomaton, ContrastCircuit):
    """ca1_windowhit on ca1a1b1c1_id: contrast vs ca1_full [digit-toggle] (integrated)."""
    circuit_id = 'ca1_windowhit'
    algorithm = 'ca1a1b1c1_id'
    base_circuit = 'ca1_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(1, 9, (batch,), generator=g)
        d = torch.tensor([1, 3, 7, 9])[torch.randint(0, 4, (batch,), generator=g)]
        rows = torch.arange(batch)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        return (torch.arange(10) - p).abs() <= 1


class Ca1DeltaclassCircuit(CellularAutomaton, ContrastCircuit):
    """ca1_deltaclass on ca1a1b1c1_id: contrast vs ca1_full [digit-toggle] (integrated)."""
    circuit_id = 'ca1_deltaclass'
    algorithm = 'ca1a1b1c1_id'
    base_circuit = 'ca1_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(1, 9, (batch,), generator=g)
        rows = torch.arange(batch)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + 5) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool)
        for o in [-1, 0, 1]:
            m[p + o] = True
        return m


class Ca2FullCircuit(CellularAutomaton, DomainCircuit):
    """ca2_full on ca2a1b1c1_id: isolated via FULL (integrated)."""
    circuit_id = 'ca2_full'
    algorithm = 'ca2a1b1c1_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Ca2ImpulseCircuit(CellularAutomaton, DomainCircuit):
    """ca2_impulse on ca2a1b1c1_id: isolated via exactly_one_in([1, 2, 3, 4, 5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'ca2_impulse'
    algorithm = 'ca2a1b1c1_id'
    _cond = conditions.exactly_one_in([1, 2, 3, 4, 5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Ca2WindowhitCircuit(CellularAutomaton, ContrastCircuit):
    """ca2_windowhit on ca2a1b1c1_id: contrast vs ca2_full [digit-toggle] (integrated)."""
    circuit_id = 'ca2_windowhit'
    algorithm = 'ca2a1b1c1_id'
    base_circuit = 'ca2_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(2, 8, (batch,), generator=g)
        d = torch.tensor([1, 3, 7, 9])[torch.randint(0, 4, (batch,), generator=g)]
        rows = torch.arange(batch)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        return (torch.arange(10) - p).abs() <= 2


class Ca2DeltaclassCircuit(CellularAutomaton, ContrastCircuit):
    """ca2_deltaclass on ca2a1b1c1_id: contrast vs ca2_full [digit-toggle] (integrated)."""
    circuit_id = 'ca2_deltaclass'
    algorithm = 'ca2a1b1c1_id'
    base_circuit = 'ca2_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(2, 8, (batch,), generator=g)
        rows = torch.arange(batch)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + 5) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool)
        for o in [-2, 0, 2]:
            m[p + o] = True
        return m


class Ca3FullCircuit(CellularAutomaton, DomainCircuit):
    """ca3_full on ca3a1b1c1_id: isolated via FULL (integrated)."""
    circuit_id = 'ca3_full'
    algorithm = 'ca3a1b1c1_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Ca3ImpulseCircuit(CellularAutomaton, DomainCircuit):
    """ca3_impulse on ca3a1b1c1_id: isolated via exactly_one_in([1, 2, 3, 4, 5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'ca3_impulse'
    algorithm = 'ca3a1b1c1_id'
    _cond = conditions.exactly_one_in([1, 2, 3, 4, 5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Ca3WindowhitCircuit(CellularAutomaton, ContrastCircuit):
    """ca3_windowhit on ca3a1b1c1_id: contrast vs ca3_full [digit-toggle] (integrated)."""
    circuit_id = 'ca3_windowhit'
    algorithm = 'ca3a1b1c1_id'
    base_circuit = 'ca3_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(3, 7, (batch,), generator=g)
        d = torch.tensor([1, 3, 7, 9])[torch.randint(0, 4, (batch,), generator=g)]
        rows = torch.arange(batch)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        return (torch.arange(10) - p).abs() <= 3


class Ca3DeltaclassCircuit(CellularAutomaton, ContrastCircuit):
    """ca3_deltaclass on ca3a1b1c1_id: contrast vs ca3_full [digit-toggle] (integrated)."""
    circuit_id = 'ca3_deltaclass'
    algorithm = 'ca3a1b1c1_id'
    base_circuit = 'ca3_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(3, 7, (batch,), generator=g)
        rows = torch.arange(batch)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + 5) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool)
        for o in [-3, -2, 0, 2, 3]:
            m[p + o] = True
        return m


class Ca5FullCircuit(CellularAutomaton, DomainCircuit):
    """ca5_full on ca5a1b1c1_id: isolated via FULL (integrated)."""
    circuit_id = 'ca5_full'
    algorithm = 'ca5a1b1c1_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Ca5ImpulseCircuit(CellularAutomaton, DomainCircuit):
    """ca5_impulse on ca5a1b1c1_id: isolated via exactly_one_in([1, 2, 3, 4, 5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'ca5_impulse'
    algorithm = 'ca5a1b1c1_id'
    _cond = conditions.exactly_one_in([1, 2, 3, 4, 5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Ca5DeltaclassCircuit(CellularAutomaton, ContrastCircuit):
    """ca5_deltaclass on ca5a1b1c1_id: contrast vs ca5_full [digit-toggle] (integrated)."""
    circuit_id = 'ca5_deltaclass'
    algorithm = 'ca5a1b1c1_id'
    base_circuit = 'ca5_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.tensor([2, 4, 6, 8])[torch.randint(0, 4, (batch,), generator=g)]
        rows = torch.arange(batch)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool)
        m[p] = True
        m[(p + 5) % 10] = True
        return m


class Chain1FullCircuit(ChainedIndex, DomainCircuit):
    """chain1_full on chain10d1_id: isolated via FULL (integrated)."""
    circuit_id = 'chain1_full'
    algorithm = 'chain10d1_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Chain1BinaryCircuit(ChainedIndex, DomainCircuit):
    """chain1_binary on chain10d1_id: isolated via BINARY (integrated)."""
    circuit_id = 'chain1_binary'
    algorithm = 'chain10d1_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Chain1PathhitCircuit(ChainedIndex, ContrastCircuit):
    """chain1_pathhit on chain10d1_id: contrast vs chain1_full [digit-toggle] (integrated)."""
    circuit_id = 'chain1_pathhit'
    algorithm = 'chain10d1_id'
    base_circuit = 'chain1_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        for k in range(128):
            bad = torch.tensor([torch.equal(self.eval(clean[i]), self.eval(corrupt[i]))
                                for i in range(batch)])
            if not int(bad.any()): break
            idx = bad.nonzero().flatten()
            p[idx] = torch.randint(0, self.input_length, (idx.numel(),), generator=g)
            d2 = torch.randint(1, 10, (idx.numel(),), generator=g)
            corrupt[idx] = clean[idx]
            corrupt[idx, p[idx]] = (clean[idx, p[idx]] + d2) % 10
        bad = torch.tensor([torch.equal(self.eval(clean[i]), self.eval(corrupt[i]))
                            for i in range(batch)])
        assert not int(bad.any())
        return clean, corrupt


class Chain2FullCircuit(ChainedIndex, DomainCircuit):
    """chain2_full on chain10d2_id: isolated via FULL (integrated)."""
    circuit_id = 'chain2_full'
    algorithm = 'chain10d2_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Chain2BinaryCircuit(ChainedIndex, DomainCircuit):
    """chain2_binary on chain10d2_id: isolated via BINARY (integrated)."""
    circuit_id = 'chain2_binary'
    algorithm = 'chain10d2_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Chain2PathhitCircuit(ChainedIndex, ContrastCircuit):
    """chain2_pathhit on chain10d2_id: contrast vs chain2_full [digit-toggle] (integrated)."""
    circuit_id = 'chain2_pathhit'
    algorithm = 'chain10d2_id'
    base_circuit = 'chain2_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        for k in range(128):
            bad = torch.tensor([torch.equal(self.eval(clean[i]), self.eval(corrupt[i]))
                                for i in range(batch)])
            if not int(bad.any()): break
            idx = bad.nonzero().flatten()
            p[idx] = torch.randint(0, self.input_length, (idx.numel(),), generator=g)
            d2 = torch.randint(1, 10, (idx.numel(),), generator=g)
            corrupt[idx] = clean[idx]
            corrupt[idx, p[idx]] = (clean[idx, p[idx]] + d2) % 10
        bad = torch.tensor([torch.equal(self.eval(clean[i]), self.eval(corrupt[i]))
                            for i in range(batch)])
        assert not int(bad.any())
        return clean, corrupt


class Chain3FullCircuit(ChainedIndex, DomainCircuit):
    """chain3_full on chain10d3_id: isolated via FULL (integrated)."""
    circuit_id = 'chain3_full'
    algorithm = 'chain10d3_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Chain3BinaryCircuit(ChainedIndex, DomainCircuit):
    """chain3_binary on chain10d3_id: isolated via BINARY (integrated)."""
    circuit_id = 'chain3_binary'
    algorithm = 'chain10d3_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Chain3PathhitCircuit(ChainedIndex, ContrastCircuit):
    """chain3_pathhit on chain10d3_id: contrast vs chain3_full [digit-toggle] (integrated)."""
    circuit_id = 'chain3_pathhit'
    algorithm = 'chain10d3_id'
    base_circuit = 'chain3_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        for k in range(128):
            bad = torch.tensor([torch.equal(self.eval(clean[i]), self.eval(corrupt[i]))
                                for i in range(batch)])
            if not int(bad.any()): break
            idx = bad.nonzero().flatten()
            p[idx] = torch.randint(0, self.input_length, (idx.numel(),), generator=g)
            d2 = torch.randint(1, 10, (idx.numel(),), generator=g)
            corrupt[idx] = clean[idx]
            corrupt[idx, p[idx]] = (clean[idx, p[idx]] + d2) % 10
        bad = torch.tensor([torch.equal(self.eval(clean[i]), self.eval(corrupt[i]))
                            for i in range(batch)])
        assert not int(bad.any())
        return clean, corrupt


class Sjf1BusyCircuit(Sjf1, DomainCircuit):
    """sjf1_busy on sjf1_id: isolated via monotone() (integrated)."""
    circuit_id = 'sjf1_busy'
    algorithm = 'sjf1_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Sjf1StarveCircuit(Sjf1, DomainCircuit):
    """sjf1_starve on sjf1_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'sjf1_starve'
    algorithm = 'sjf1_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Sjf1ShadowCircuit(Sjf1, ContrastCircuit):
    """sjf1_shadow on sjf1_id: contrast vs sjf1_busy [value-perturb-control] (integrated)."""
    circuit_id = 'sjf1_shadow'
    algorithm = 'sjf1_id'
    base_circuit = 'sjf1_busy'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(1, self.input_length, (batch,), generator=g)
        for _ in range(64):
            Mb = torch.cummax(clean, dim=1).values[rows, p - 1]
            bad = Mb < 2
            if not int(bad.any()): break
            clean[bad] = torch.randint(0, 10, (int(bad.sum()), self.input_length), generator=g)
        assert not int(bad.any())
        Mb = torch.cummax(clean, dim=1).values[rows, p - 1]
        a = torch.randint(0, 9, (batch,), generator=g) % Mb
        b = (a + 1 + torch.randint(0, 9, (batch,), generator=g) % (Mb - 1)) % Mb
        clean[rows, p] = a
        corrupt = clean.clone()
        corrupt[rows, p] = b
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) == int((clean_row != corrupt_row).nonzero()[0])


class Sjf1DeferCircuit(Sjf1, ContrastCircuit):
    """sjf1_defer on sjf1_id: contrast vs sjf1_busy [threshold-straddle] (integrated)."""
    circuit_id = 'sjf1_defer'
    algorithm = 'sjf1_id'
    base_circuit = 'sjf1_busy'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(1, self.input_length, (batch,), generator=g)
        for _ in range(64):
            Mb = torch.cummax(clean, dim=1).values[rows, p - 1]
            bad = (Mb < 1) | (Mb > 8)
            if not int(bad.any()): break
            clean[bad] = torch.randint(0, 10, (int(bad.sum()), self.input_length), generator=g)
        assert not int(bad.any())
        Mb = torch.cummax(clean, dim=1).values[rows, p - 1]
        clean[rows, p] = torch.randint(0, 9, (batch,), generator=g) % Mb
        corrupt = clean.clone()
        corrupt[rows, p] = 9
        return clean, corrupt


class Sjf1DeferSparseCircuit(Sjf1, ContrastCircuit):
    """sjf1_defer_sparse on sjf1_id: contrast vs sjf1_busy [threshold-straddle] (integrated)."""
    circuit_id = 'sjf1_defer_sparse'
    algorithm = 'sjf1_id'
    base_circuit = 'sjf1_busy'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        rows = torch.arange(batch)
        Mb = torch.randint(2, 9, (batch,), generator=g)
        p = torch.randint(1, 7, (batch,), generator=g)
        q = p + torch.randint(2, 4, (batch,), generator=g)
        clean = torch.randint(0, 9, (batch, L), generator=g) % (Mb + 1).unsqueeze(1)
        clean[:, 0] = Mb
        clean[rows, p] = torch.randint(0, 9, (batch,), generator=g) % Mb
        xq = Mb + 1 + torch.randint(0, 9, (batch,), generator=g) % (9 - Mb)
        clean[rows, q] = xq
        after = torch.arange(L).unsqueeze(0) > q.unsqueeze(1)
        clean = torch.where(after, clean % (xq + 1).unsqueeze(1), clean)
        corrupt = clean.clone()
        corrupt[rows, p] = 9
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        M = int(clean_row[:p].max())
        m = torch.zeros(10, dtype=torch.bool)
        m[p] = True
        for j in range(p + 1, 10):
            if int(clean_row[j]) > M:
                m[j] = True
                break
        return m


class Psumk1FullCircuit(PrefixSumMod, DomainCircuit):
    """psumk1_full on psum_id: isolated via FULL (integrated)."""
    circuit_id = 'psumk1_full'
    algorithm = 'psum_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Psumk1ControlCircuit(PrefixSumMod, ContrastCircuit):
    """psumk1_control on psum_id: contrast vs psumk1_full [digit-toggle] (integrated)."""
    circuit_id = 'psumk1_control'
    algorithm = 'psum_id'
    base_circuit = 'psumk1_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        delta = (int(corrupt_row[p]) - int(clean_row[p])) % 10
        coeff = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        m = torch.zeros(10, dtype=torch.bool)
        for i in range(p, 10):
            m[i] = (delta * coeff[i - p]) % 10 != 0
        return m


class Psumk2FullCircuit(PrefixSumMod, DomainCircuit):
    """psumk2_full on psum_psum_id: isolated via FULL (integrated)."""
    circuit_id = 'psumk2_full'
    algorithm = 'psum_psum_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Psumk2ImpulseCircuit(PrefixSumMod, DomainCircuit):
    """psumk2_impulse on psum_psum_id: isolated via exactly_one_in([1, 2, 3, 4, 5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'psumk2_impulse'
    algorithm = 'psum_psum_id'
    _cond = conditions.exactly_one_in([1, 2, 3, 4, 5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Psumk2SuffixholesCircuit(PrefixSumMod, ContrastCircuit):
    """psumk2_suffixholes on psum_psum_id: contrast vs psumk2_full [digit-toggle] (integrated)."""
    circuit_id = 'psumk2_suffixholes'
    algorithm = 'psum_psum_id'
    base_circuit = 'psumk2_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        delta = (int(corrupt_row[p]) - int(clean_row[p])) % 10
        coeff = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        m = torch.zeros(10, dtype=torch.bool)
        for i in range(p, 10):
            m[i] = (delta * coeff[i - p]) % 10 != 0
        return m


class Psumk3FullCircuit(PrefixSumMod, DomainCircuit):
    """psumk3_full on psum_psum_psum_id: isolated via FULL (integrated)."""
    circuit_id = 'psumk3_full'
    algorithm = 'psum_psum_psum_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Psumk3ImpulseCircuit(PrefixSumMod, DomainCircuit):
    """psumk3_impulse on psum_psum_psum_id: isolated via exactly_one_in([1, 2, 3, 4, 5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'psumk3_impulse'
    algorithm = 'psum_psum_psum_id'
    _cond = conditions.exactly_one_in([1, 2, 3, 4, 5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Psumk3SuffixholesCircuit(PrefixSumMod, ContrastCircuit):
    """psumk3_suffixholes on psum_psum_psum_id: contrast vs psumk3_full [digit-toggle] (integrated)."""
    circuit_id = 'psumk3_suffixholes'
    algorithm = 'psum_psum_psum_id'
    base_circuit = 'psumk3_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        delta = (int(corrupt_row[p]) - int(clean_row[p])) % 10
        coeff = [1, 3, 6, 0, 5, 1, 8, 6, 5, 5]
        m = torch.zeros(10, dtype=torch.bool)
        for i in range(p, 10):
            m[i] = (delta * coeff[i - p]) % 10 != 0
        return m


class DuelWholeCircuit(Duelsum, DomainCircuit):
    """duel_whole on duelsum_id: isolated via FULL (integrated)."""
    circuit_id = 'duel_whole'
    algorithm = 'duelsum_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DuelOddledCircuit(Duelsum, DomainCircuit):
    """duel_oddled on duelsum_id: isolated via Condition('odd_leads', lambda x: bool((torch.cumsum(torch.where(x % 2 == 1, x, torch.zeros_like(x)), 0) > torch.cumsum(torch.where(x % 2 == 0, x, torch.zeros_like(x)), 0)).all()), lambda batch, seed: (lambda blk: blk[[bool((torch.cumsum(torch.where(r % 2 == 1, r, torch.zeros_like(r)), 0) > torch.cumsum(torch.where(r % 2 == 0, r, torch.zeros_like(r)), 0)).all()) for r in blk]][:batch])(torch.randint(0, 10, (16 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'duel_oddled'))))) (integrated)."""
    circuit_id = 'duel_oddled'
    algorithm = 'duelsum_id'
    _cond = conditions.Condition('odd_leads', lambda x: bool((torch.cumsum(torch.where(x % 2 == 1, x, torch.zeros_like(x)), 0) > torch.cumsum(torch.where(x % 2 == 0, x, torch.zeros_like(x)), 0)).all()), lambda batch, seed: (lambda blk: blk[[bool((torch.cumsum(torch.where(r % 2 == 1, r, torch.zeros_like(r)), 0) > torch.cumsum(torch.where(r % 2 == 0, r, torch.zeros_like(r)), 0)).all()) for r in blk]][:batch])(torch.randint(0, 10, (16 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'duel_oddled')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DuelEvensplitCircuit(Duelsum, ContrastCircuit):
    """duel_evensplit on duelsum_id: contrast vs duel_whole [value-perturb-control] (integrated)."""
    circuit_id = 'duel_evensplit'
    algorithm = 'duelsum_id'
    base_circuit = 'duel_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            redraw = idx[(clean[idx] % 2 == 0).sum(dim=1) == 0]
            if redraw.numel():
                clean[redraw] = torch.randint(0, 10, (redraw.numel(), L), generator=g)
            p = torch.randint(0, L, (idx.numel(),), generator=g)
            d = 2 + 2 * torch.randint(0, 4, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                if int(clean[i, p[k]]) % 2 != 0:
                    continue
                corrupt[i] = clean[i]
                corrupt[i, p[k]] = (int(clean[i, p[k]]) + int(d[k])) % 10
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class DuelParityswapCircuit(Duelsum, ContrastCircuit):
    """duel_parityswap on duelsum_id: contrast vs duel_whole [position-swap] (integrated)."""
    circuit_id = 'duel_parityswap'
    algorithm = 'duelsum_id'
    base_circuit = 'duel_whole'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            for i in idx.tolist():
                pq = [(a, b) for a in range(L) for b in range(a + 1, L)
                      if int(clean[i, a]) % 2 == int(clean[i, b]) % 2 and int(clean[i, a]) != int(clean[i, b])]
                if not pq:
                    clean[i] = torch.randint(0, 10, (L,), generator=g)
                    continue
                a, b = pq[int(torch.randint(0, len(pq), (1,), generator=g))]
                corrupt[i] = clean[i]
                va, vb = int(clean[i, a]), int(clean[i, b])
                corrupt[i, a], corrupt[i, b] = vb, va
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class NetrmMaxsideCircuit(Netrm, DomainCircuit):
    """netrm_maxside on netrm_id: isolated via Condition('netrm_maxside', lambda x: bool((2 * x >= torch.cummax(x, 0).values + torch.cummin(x, 0).values).all()), lambda batch, seed: (lambda blk: blk[(2 * blk >= torch.cummax(blk, 1).values + torch.cummin(blk, 1).values).all(1)][:batch])(torch.randint(0, 10, (640 * batch + 4096, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'netrm_maxside'))))) (integrated)."""
    circuit_id = 'netrm_maxside'
    algorithm = 'netrm_id'
    _cond = conditions.Condition('netrm_maxside', lambda x: bool((2 * x >= torch.cummax(x, 0).values + torch.cummin(x, 0).values).all()), lambda batch, seed: (lambda blk: blk[(2 * blk >= torch.cummax(blk, 1).values + torch.cummin(blk, 1).values).all(1)][:batch])(torch.randint(0, 10, (640 * batch + 4096, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'netrm_maxside')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class NetrmX0is0Circuit(Netrm, DomainCircuit):
    """netrm_x0is0 on netrm_id: isolated via pin(0, [0]) (integrated)."""
    circuit_id = 'netrm_x0is0'
    algorithm = 'netrm_id'
    _cond = conditions.pin(0, [0])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class NetrmFlipCircuit(Netrm, ContrastCircuit):
    """netrm_flip on netrm_id: contrast vs netrm_maxside [threshold-straddle] (integrated)."""
    circuit_id = 'netrm_flip'
    algorithm = 'netrm_id'
    base_circuit = 'netrm_maxside'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        for r in range(batch):
            done = False
            for _ in range(200):
                M = int(clean[r, 0]); m = int(clean[r, 0])
                opts = []
                for j in range(1, L):
                    lo = [v for v in range(m + 1, M) if 2 * v < M + m]
                    hi = [v for v in range(m + 1, M) if 2 * v >= M + m]
                    if lo and hi:
                        opts.append((j, lo, hi))
                    M = max(M, int(clean[r, j])); m = min(m, int(clean[r, j]))
                if opts:
                    j, lo, hi = opts[int(torch.randint(0, len(opts), (1,), generator=g))]
                    if int(torch.randint(0, 2, (1,), generator=g)):
                        lo, hi = hi, lo
                    clean[r, j] = hi[int(torch.randint(0, len(hi), (1,), generator=g))]
                    corrupt[r] = clean[r]
                    corrupt[r, j] = lo[int(torch.randint(0, len(lo), (1,), generator=g))]
                    done = True
                    break
                clean[r] = torch.randint(0, 10, (L,), generator=g)
            assert done
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) == int((clean_row != corrupt_row).nonzero()[0])


class NetrmReleaseCircuit(Netrm, ContrastCircuit):
    """netrm_release on netrm_id: contrast vs netrm_maxside [threshold-straddle] (integrated)."""
    circuit_id = 'netrm_release'
    algorithm = 'netrm_id'
    base_circuit = 'netrm_maxside'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        for r in range(batch):
            done = False
            for _ in range(200):
                M = int(clean[r, 0]); m = int(clean[r, 0])
                opts = []
                for j in range(1, L):
                    lo = [v for v in range(m + 1, M) if 2 * v < M + m]
                    hi = [v for v in range(m + 1, M) if 0 <= 2 * v - (M + m) <= 2]
                    if lo and hi:
                        opts.append((j, lo, hi))
                    M = max(M, int(clean[r, j])); m = min(m, int(clean[r, j]))
                if opts:
                    j, lo, hi = opts[int(torch.randint(0, len(opts), (1,), generator=g))]
                    clean[r, j] = hi[int(torch.randint(0, len(hi), (1,), generator=g))]
                    corrupt[r] = clean[r]
                    corrupt[r, j] = lo[int(torch.randint(0, len(lo), (1,), generator=g))]
                    done = True
                    break
                clean[r] = torch.randint(0, 10, (L,), generator=g)
            assert done
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) == int((clean_row != corrupt_row).nonzero()[0])


class NetrmMaxraiseCircuit(Netrm, ContrastCircuit):
    """netrm_maxraise on netrm_id: contrast vs netrm_maxside [digit-toggle] (integrated)."""
    circuit_id = 'netrm_maxraise'
    algorithm = 'netrm_id'
    base_circuit = 'netrm_maxside'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 9, (batch, L), generator=g)
        rows = torch.arange(batch)
        for _ in range(64):
            M = clean.max(dim=1).values
            bad = (clean == M.unsqueeze(1)).sum(dim=1) != 1
            if not int(bad.any()): break
            clean[bad] = torch.randint(0, 9, (int(bad.sum()), L), generator=g)
        assert not int(bad.any())
        M = clean.max(dim=1).values
        p = (clean == M.unsqueeze(1)).long().argmax(dim=1)
        corrupt = clean.clone()
        corrupt[rows, p] = M + 1 + torch.randint(0, 9, (batch,), generator=g) % (9 - M)
        return clean, corrupt


class ConvfixCleanCircuit(Convfix, DomainCircuit):
    """convfix_clean on convfix_id: isolated via Condition('conv_codeword', lambda x: all((((int(x[i]) >> 1) & 1) == ((int(x[i]) & 1) ^ ((int(x[i - 1]) & 1) if i else 0))) for i in range(10)), lambda batch, seed: (lambda g: (lambda sb: (lambda pb: torch.tensor([[0, 4, 8, 0, 4, 8], [2, 6, 2, 6, 2, 6], [1, 5, 9, 1, 5, 9], [3, 7, 3, 7, 3, 7]])[sb * 2 + pb, torch.randint(0, 6, (batch, 10), generator=g)])(sb ^ torch.cat([torch.zeros(batch, 1, dtype=torch.long), sb[:, :-1]], dim=1)))(torch.randint(0, 2, (batch, 10), generator=g)))(torch.Generator().manual_seed(conditions.stream_key(seed, 'convfix_clean')))) (integrated)."""
    circuit_id = 'convfix_clean'
    algorithm = 'convfix_id'
    _cond = conditions.Condition('conv_codeword', lambda x: all((((int(x[i]) >> 1) & 1) == ((int(x[i]) & 1) ^ ((int(x[i - 1]) & 1) if i else 0))) for i in range(10)), lambda batch, seed: (lambda g: (lambda sb: (lambda pb: torch.tensor([[0, 4, 8, 0, 4, 8], [2, 6, 2, 6, 2, 6], [1, 5, 9, 1, 5, 9], [3, 7, 3, 7, 3, 7]])[sb * 2 + pb, torch.randint(0, 6, (batch, 10), generator=g)])(sb ^ torch.cat([torch.zeros(batch, 1, dtype=torch.long), sb[:, :-1]], dim=1)))(torch.randint(0, 2, (batch, 10), generator=g)))(torch.Generator().manual_seed(conditions.stream_key(seed, 'convfix_clean'))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class ConvfixMsgflipCircuit(Convfix, ContrastCircuit):
    """convfix_msgflip on convfix_id: contrast vs convfix_clean [compensated-two-site] (integrated)."""
    circuit_id = 'convfix_msgflip'
    algorithm = 'convfix_id'
    base_circuit = 'convfix_clean'
    local_predicate = True
    _cond = conditions.Condition('conv_codeword', lambda x: all((((int(x[i]) >> 1) & 1) == ((int(x[i]) & 1) ^ ((int(x[i - 1]) & 1) if i else 0))) for i in range(10)), lambda batch, seed: (lambda g: (lambda sb: (lambda pb: torch.tensor([[0, 4, 8, 0, 4, 8], [2, 6, 2, 6, 2, 6], [1, 5, 9, 1, 5, 9], [3, 7, 3, 7, 3, 7]])[sb * 2 + pb, torch.randint(0, 6, (batch, 10), generator=g)])(sb ^ torch.cat([torch.zeros(batch, 1, dtype=torch.long), sb[:, :-1]], dim=1)))(torch.randint(0, 2, (batch, 10), generator=g)))(torch.Generator().manual_seed(conditions.stream_key(seed, 'convfix_msgflip'))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        q = torch.randint(0, L - 1, (batch,), generator=g)
        enc = torch.tensor([[0, 4, 8, 0, 4, 8], [2, 6, 2, 6, 2, 6], [1, 5, 9, 1, 5, 9], [3, 7, 3, 7, 3, 7]])
        for i in range(batch):
            s = [int(d) & 1 for d in clean[i].tolist()]
            qi = int(q[i])
            s[qi] ^= 1
            for j in (qi, qi + 1):
                pj = s[j] ^ (s[j - 1] if j else 0)
                corrupt[i, j] = enc[2 * s[j] + pj, int(torch.randint(0, 6, (1,), generator=g))]
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) == int((clean_row != corrupt_row).nonzero()[0]) + 1


class ConvfixMsgflip2Circuit(Convfix, ContrastCircuit):
    """convfix_msgflip2 on convfix_id: contrast vs convfix_clean [compensated-two-site] (integrated)."""
    circuit_id = 'convfix_msgflip2'
    algorithm = 'convfix_id'
    base_circuit = 'convfix_clean'
    local_predicate = True
    _cond = conditions.Condition('conv_codeword', lambda x: all((((int(x[i]) >> 1) & 1) == ((int(x[i]) & 1) ^ ((int(x[i - 1]) & 1) if i else 0))) for i in range(10)), lambda batch, seed: (lambda g: (lambda sb: (lambda pb: torch.tensor([[0, 4, 8, 0, 4, 8], [2, 6, 2, 6, 2, 6], [1, 5, 9, 1, 5, 9], [3, 7, 3, 7, 3, 7]])[sb * 2 + pb, torch.randint(0, 6, (batch, 10), generator=g)])(sb ^ torch.cat([torch.zeros(batch, 1, dtype=torch.long), sb[:, :-1]], dim=1)))(torch.randint(0, 2, (batch, 10), generator=g)))(torch.Generator().manual_seed(conditions.stream_key(seed, 'convfix_msgflip2'))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        q1 = torch.randint(0, L - 4, (batch,), generator=g)
        q2 = q1 + 2 + torch.randint(0, 2, (batch,), generator=g)
        enc = torch.tensor([[0, 4, 8, 0, 4, 8], [2, 6, 2, 6, 2, 6], [1, 5, 9, 1, 5, 9], [3, 7, 3, 7, 3, 7]])
        for i in range(batch):
            s = [int(d) & 1 for d in clean[i].tolist()]
            a, b = int(q1[i]), int(q2[i])
            s[a] ^= 1
            s[b] ^= 1
            for j in (a, a + 1, b, b + 1):
                pj = s[j] ^ (s[j - 1] if j else 0)
                corrupt[i, j] = enc[2 * s[j] + pj, int(torch.randint(0, 6, (1,), generator=g))]
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        d = (clean_row != corrupt_row).nonzero().flatten().tolist()
        m = torch.zeros(10, dtype=torch.bool)
        m[d[0] + 1] = True
        m[d[2] + 1] = True
        return m


class ConvfixUncorrectableCircuit(Convfix, ContrastCircuit):
    """convfix_uncorrectable on convfix_id: contrast vs convfix_clean [digit-toggle] (integrated)."""
    circuit_id = 'convfix_uncorrectable'
    algorithm = 'convfix_id'
    base_circuit = 'convfix_clean'
    local_predicate = True
    _cond = conditions.Condition('conv_codeword', lambda x: all((((int(x[i]) >> 1) & 1) == ((int(x[i]) & 1) ^ ((int(x[i - 1]) & 1) if i else 0))) for i in range(10)), lambda batch, seed: (lambda g: (lambda sb: (lambda pb: torch.tensor([[0, 4, 8, 0, 4, 8], [2, 6, 2, 6, 2, 6], [1, 5, 9, 1, 5, 9], [3, 7, 3, 7, 3, 7]])[sb * 2 + pb, torch.randint(0, 6, (batch, 10), generator=g)])(sb ^ torch.cat([torch.zeros(batch, 1, dtype=torch.long), sb[:, :-1]], dim=1)))(torch.randint(0, 2, (batch, 10), generator=g)))(torch.Generator().manual_seed(conditions.stream_key(seed, 'convfix_uncorrectable'))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        rows = torch.arange(batch)
        q = torch.randint(0, L - 2, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, q] = clean[rows, q] ^ 1
        corrupt[rows, q + 1] = clean[rows, q + 1] ^ 1
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        q = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool)
        m[q + 1] = True
        m[q + 2] = True
        return m


class UsumBinaryCircuit(Usum, DomainCircuit):
    """usum_binary on usum_id: isolated via BINARY (integrated)."""
    circuit_id = 'usum_binary'
    algorithm = 'usum_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class UsumOnerepeatCircuit(Usum, DomainCircuit):
    """usum_onerepeat on usum_id: isolated via ONE_REPEAT (integrated)."""
    circuit_id = 'usum_onerepeat'
    algorithm = 'usum_id'
    _cond = conditions.ONE_REPEAT
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class UsumAlldistinctCircuit(Usum, DomainCircuit):
    """usum_alldistinct on usum_id: isolated via Condition('all_distinct', lambda x: len(set(x.tolist())) == 10, lambda batch, seed: torch.argsort(torch.rand(batch, 10, generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'usum_alldistinct'))), dim=1)) (integrated)."""
    circuit_id = 'usum_alldistinct'
    algorithm = 'usum_id'
    _cond = conditions.Condition('all_distinct', lambda x: len(set(x.tolist())) == 10, lambda batch, seed: torch.argsort(torch.rand(batch, 10, generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'usum_alldistinct'))), dim=1))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class UsumMonoincCircuit(Usum, DomainCircuit):
    """usum_monoinc on usum_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'usum_monoinc'
    algorithm = 'usum_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class UsumSumle9Circuit(Usum, DomainCircuit):
    """usum_sumle9 on usum_id: isolated via sum_at_most(9) (integrated)."""
    circuit_id = 'usum_sumle9'
    algorithm = 'usum_id'
    _cond = conditions.sum_at_most(9)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class UsumSwapnovelCircuit(Usum, ContrastCircuit):
    """usum_swapnovel on usum_id: contrast vs usum_alldistinct [position-swap] (integrated)."""
    circuit_id = 'usum_swapnovel'
    algorithm = 'usum_id'
    base_circuit = 'usum_alldistinct'
    local_predicate = True
    _cond = conditions.Condition('all_distinct', lambda x: len(set(x.tolist())) == 10, lambda batch, seed: torch.argsort(torch.rand(batch, 10, generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'usum_swapnovel'))), dim=1))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        clean = DomainCircuit.sample(self, batch, seed)
        g = torch.Generator().manual_seed(int(seed) + 1)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length - 1, (batch,), generator=g)
        q = p + 1 + torch.randint(0, 9, (batch,), generator=g) % (9 - p)
        corrupt = clean.clone()
        corrupt[rows, p] = clean[rows, q]
        corrupt[rows, q] = clean[rows, p]
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        d = (clean_row != corrupt_row).nonzero().flatten()
        return (torch.arange(10) >= int(d.min())) & (torch.arange(10) < int(d.max()))


class UsumRefreshCircuit(Usum, ContrastCircuit):
    """usum_refresh on usum_id: contrast vs usum_onerepeat [digit-toggle] (integrated)."""
    circuit_id = 'usum_refresh'
    algorithm = 'usum_id'
    base_circuit = 'usum_onerepeat'
    local_predicate = False
    _cond = conditions.ONE_REPEAT
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        for i in range(batch):
            for k in range(64):
                vals = clean[i].tolist()
                missing = [v for v in range(10) if v not in vals]
                if missing and missing[0] != 0:
                    break
                clean[i] = DomainCircuit.sample(self, 1, int(seed) * 131 + i * 64 + k + 1)[0]
            vals = clean[i].tolist()
            dup = next(v for v in vals if vals.count(v) == 2)
            j2 = 9 - vals[::-1].index(dup)
            corrupt[i] = clean[i]
            corrupt[i, j2] = [v for v in range(10) if v not in vals][0]
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) >= int((clean_row != corrupt_row).nonzero()[0])


class Perase0Pin3Circuit(Perase0, DomainCircuit):
    """perase0_pin3 on perase0_id: isolated via pin(0, [3]) (integrated)."""
    circuit_id = 'perase0_pin3'
    algorithm = 'perase0_id'
    _cond = conditions.pin(0, [3])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Perase0PtrStepCircuit(Perase0, ContrastCircuit):
    """perase0_ptr_step on perase0_id: contrast vs perase0_pin3 [digit-toggle] (integrated)."""
    circuit_id = 'perase0_ptr_step'
    algorithm = 'perase0_id'
    base_circuit = 'perase0_pin3'
    local_predicate = True
    _cond = conditions.pin(0, [3])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        clean[:, 0] = 3
        kp = 5 + torch.randint(0, 5, (batch,), generator=g)
        clean[rows, 3] = 1 + torch.randint(0, 9, (batch,), generator=g)
        clean[rows, kp] = 1 + torch.randint(0, 9, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[:, 0] = kp
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        m = torch.zeros(10, dtype=torch.bool)
        m[0] = True
        m[3] = True
        m[int(corrupt_row[0])] = True
        return m


class Lagf3Ge1threeCircuit(Lagf3, DomainCircuit):
    """lagf3_ge1three on lagf3_id: isolated via Condition('lagf3_active', lambda x: bool((x == 3).any()) and not bool((x[(torch.arange(10) - torch.cumsum((x == 3).long(), 0)) % 10] == x).all()), lambda batch, seed: (lambda blk: blk[[bool((r == 3).any()) and not bool((r[(torch.arange(10) - torch.cumsum((r == 3).long(), 0)) % 10] == r).all()) for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'lagf3_ge1three'))))) (integrated)."""
    circuit_id = 'lagf3_ge1three'
    algorithm = 'lagf3_id'
    _cond = conditions.Condition('lagf3_active', lambda x: bool((x == 3).any()) and not bool((x[(torch.arange(10) - torch.cumsum((x == 3).long(), 0)) % 10] == x).all()), lambda batch, seed: (lambda blk: blk[[bool((r == 3).any()) and not bool((r[(torch.arange(10) - torch.cumsum((r == 3).long(), 0)) % 10] == r).all()) for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'lagf3_ge1three')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Lagf3NothreeCircuit(Lagf3, DomainCircuit):
    """lagf3_nothree on lagf3_id: isolated via exclude_digit(3) (integrated)."""
    circuit_id = 'lagf3_nothree'
    algorithm = 'lagf3_id'
    _cond = conditions.exclude_digit(3)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Lagf3Plant3Circuit(Lagf3, ContrastCircuit):
    """lagf3_plant3 on lagf3_id: contrast vs lagf3_ge1three [digit-toggle] (integrated)."""
    circuit_id = 'lagf3_plant3'
    algorithm = 'lagf3_id'
    base_circuit = 'lagf3_ge1three'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            p = torch.randint(0, L, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                if int(clean[i, p[k]]) == 3:
                    continue
                corrupt[i] = clean[i]
                corrupt[i, p[k]] = 3
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Lagf3Move3Circuit(Lagf3, ContrastCircuit):
    """lagf3_move3 on lagf3_id: contrast vs lagf3_ge1three [position-swap] (integrated)."""
    circuit_id = 'lagf3_move3'
    algorithm = 'lagf3_id'
    base_circuit = 'lagf3_ge1three'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            p = torch.randint(0, L, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                threes = (clean[i] == 3).nonzero().flatten()
                if not threes.numel() or int(clean[i, p[k]]) == 3:
                    continue
                q = int(threes[int(torch.randint(0, threes.numel(), (1,), generator=g))])
                corrupt[i] = clean[i]
                corrupt[i, q] = int(clean[i, p[k]])
                corrupt[i, p[k]] = 3
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Lagf3ValeditCircuit(Lagf3, ContrastCircuit):
    """lagf3_valedit on lagf3_id: contrast vs lagf3_ge1three [value-perturb-control] (integrated)."""
    circuit_id = 'lagf3_valedit'
    algorithm = 'lagf3_id'
    base_circuit = 'lagf3_ge1three'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            p = torch.randint(0, L, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                new = int(torch.randint(0, 10, (1,), generator=g))
                if int(clean[i, p[k]]) == 3 or new == 3 or new == int(clean[i, p[k]]):
                    continue
                corrupt[i] = clean[i]
                corrupt[i, p[k]] = new
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class NearidxMaxsideCircuit(Nearidx, DomainCircuit):
    """nearidx_maxside on nearidx_id: isolated via Condition('netrm_maxside', lambda x: bool((2 * x >= torch.cummax(x, 0).values + torch.cummin(x, 0).values).all()), lambda batch, seed: (lambda blk: blk[(2 * blk >= torch.cummax(blk, 1).values + torch.cummin(blk, 1).values).all(1)][:batch])(torch.randint(0, 10, (640 * batch + 4096, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'nearidx_maxside'))))) (integrated)."""
    circuit_id = 'nearidx_maxside'
    algorithm = 'nearidx_id'
    _cond = conditions.Condition('netrm_maxside', lambda x: bool((2 * x >= torch.cummax(x, 0).values + torch.cummin(x, 0).values).all()), lambda batch, seed: (lambda blk: blk[(2 * blk >= torch.cummax(blk, 1).values + torch.cummin(blk, 1).values).all(1)][:batch])(torch.randint(0, 10, (640 * batch + 4096, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'nearidx_maxside')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class NearidxFlipCircuit(Nearidx, ContrastCircuit):
    """nearidx_flip on nearidx_id: contrast vs nearidx_maxside [threshold-straddle] (integrated)."""
    circuit_id = 'nearidx_flip'
    algorithm = 'nearidx_id'
    base_circuit = 'nearidx_maxside'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        for r in range(batch):
            done = False
            for _ in range(200):
                M = int(clean[r, 0]); m = int(clean[r, 0])
                opts = []
                for j in range(1, L):
                    lo = [v for v in range(m + 1, M) if 2 * v < M + m]
                    hi = [v for v in range(m + 1, M) if 2 * v >= M + m]
                    if lo and hi:
                        opts.append((j, lo, hi))
                    M = max(M, int(clean[r, j])); m = min(m, int(clean[r, j]))
                if opts:
                    j, lo, hi = opts[int(torch.randint(0, len(opts), (1,), generator=g))]
                    if int(torch.randint(0, 2, (1,), generator=g)):
                        lo, hi = hi, lo
                    clean[r, j] = hi[int(torch.randint(0, len(hi), (1,), generator=g))]
                    corrupt[r] = clean[r]
                    corrupt[r, j] = lo[int(torch.randint(0, len(lo), (1,), generator=g))]
                    done = True
                    break
                clean[r] = torch.randint(0, 10, (L,), generator=g)
            assert done
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) == int((clean_row != corrupt_row).nonzero()[0])


class NearidxNoreleaseCircuit(Nearidx, ContrastCircuit):
    """nearidx_norelease on nearidx_id: contrast vs nearidx_maxside [threshold-straddle] (integrated)."""
    circuit_id = 'nearidx_norelease'
    algorithm = 'nearidx_id'
    base_circuit = 'nearidx_maxside'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        for r in range(batch):
            done = False
            for _ in range(200):
                M = int(clean[r, 0]); m = int(clean[r, 0])
                opts = []
                for j in range(1, L):
                    lo = [v for v in range(m + 1, M) if 2 * v < M + m]
                    hi = [v for v in range(m + 1, M) if 0 <= 2 * v - (M + m) <= 2]
                    if lo and hi:
                        opts.append((j, lo, hi))
                    M = max(M, int(clean[r, j])); m = min(m, int(clean[r, j]))
                if opts:
                    j, lo, hi = opts[int(torch.randint(0, len(opts), (1,), generator=g))]
                    clean[r, j] = hi[int(torch.randint(0, len(hi), (1,), generator=g))]
                    corrupt[r] = clean[r]
                    corrupt[r, j] = lo[int(torch.randint(0, len(lo), (1,), generator=g))]
                    done = True
                    break
                clean[r] = torch.randint(0, 10, (L,), generator=g)
            assert done
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) == int((clean_row != corrupt_row).nonzero()[0])


class Ts3One3Circuit(Tresetsum3, DomainCircuit):
    """ts3_one3 on tresetsum3_id: isolated via exactly_one_in([3]) (integrated)."""
    circuit_id = 'ts3_one3'
    algorithm = 'tresetsum3_id'
    _cond = conditions.exactly_one_in([3])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Ts3BinaryCircuit(Tresetsum3, DomainCircuit):
    """ts3_binary on tresetsum3_id: isolated via BINARY (integrated)."""
    circuit_id = 'ts3_binary'
    algorithm = 'tresetsum3_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Ts3Le13Circuit(Tresetsum3, DomainCircuit):
    """ts3_le13 on tresetsum3_id: isolated via at_most_k_of(3, 1) (integrated)."""
    circuit_id = 'ts3_le13'
    algorithm = 'tresetsum3_id'
    _cond = conditions.at_most_k_of(3, 1)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Ts3Sumle9Circuit(Tresetsum3, DomainCircuit):
    """ts3_sumle9 on tresetsum3_id: isolated via sum_at_most(9) (integrated)."""
    circuit_id = 'ts3_sumle9'
    algorithm = 'tresetsum3_id'
    _cond = conditions.sum_at_most(9)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Ts3Remove3Circuit(Tresetsum3, ContrastCircuit):
    """ts3_remove3 on tresetsum3_id: contrast vs ts3_one3 [digit-toggle] (integrated)."""
    circuit_id = 'ts3_remove3'
    algorithm = 'tresetsum3_id'
    base_circuit = 'ts3_one3'
    local_predicate = False
    _cond = conditions.exactly_one_in([3])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        for i in range(batch):
            p = int((clean[i] == 3).nonzero()[0])
            s = int(clean[i, :p].sum()) % 10
            ok = [v for v in range(10) if v != 3 and (s + v) % 10 != 0]
            corrupt[i] = clean[i]
            corrupt[i, p] = ok[int(torch.randint(0, len(ok), (1,), generator=g))]
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) >= int((clean_row != corrupt_row).nonzero()[0])


class Ts3Move3Circuit(Tresetsum3, ContrastCircuit):
    """ts3_move3 on tresetsum3_id: contrast vs ts3_one3 [position-swap] (integrated)."""
    circuit_id = 'ts3_move3'
    algorithm = 'tresetsum3_id'
    base_circuit = 'ts3_one3'
    local_predicate = False
    _cond = conditions.exactly_one_in([3])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
            q = torch.randint(0, L, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                p = int((clean[i] == 3).nonzero()[0])
                if int(q[k]) == p:
                    continue
                corrupt[i] = clean[i]
                corrupt[i, p] = int(clean[i, q[k]])
                corrupt[i, q[k]] = 3
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class OepassFullCircuit(Oepass0, DomainCircuit):
    """oepass_full on oepass0_id: isolated via FULL (integrated)."""
    circuit_id = 'oepass_full'
    algorithm = 'oepass0_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class OepassSortedCircuit(Oepass0, DomainCircuit):
    """oepass_sorted on oepass0_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'oepass_sorted'
    algorithm = 'oepass0_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class OepassRevsortedCircuit(Oepass0, DomainCircuit):
    """oepass_revsorted on oepass0_id: isolated via monotone() (integrated)."""
    circuit_id = 'oepass_revsorted'
    algorithm = 'oepass0_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class OepassPairhitCircuit(Oepass0, ContrastCircuit):
    """oepass_pairhit on oepass0_id: contrast vs oepass_full [digit-toggle] (integrated)."""
    circuit_id = 'oepass_pairhit'
    algorithm = 'oepass0_id'
    base_circuit = 'oepass_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt


class PgeFullCircuit(Pge, DomainCircuit):
    """pge_full on pge_id: isolated via FULL (integrated)."""
    circuit_id = 'pge_full'
    algorithm = 'pge_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PgePlateauCircuit(Pge, DomainCircuit):
    """pge_plateau on pge_id: isolated via monotone() (integrated)."""
    circuit_id = 'pge_plateau'
    algorithm = 'pge_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PgeRaiseCircuit(Pge, ContrastCircuit):
    """pge_raise on pge_id: contrast vs pge_full [threshold-straddle] (integrated)."""
    circuit_id = 'pge_raise'
    algorithm = 'pge_id'
    base_circuit = 'pge_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            p = torch.randint(1, L, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                if int(clean[i, p[k]]) == 9 or int(clean[i, p[k]]) > int(clean[i, :p[k]].max()):
                    continue
                corrupt[i] = clean[i]
                corrupt[i, p[k]] = 9
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class PgeUnshadowCircuit(Pge, ContrastCircuit):
    """pge_unshadow on pge_id: contrast vs pge_full [digit-toggle] (integrated)."""
    circuit_id = 'pge_unshadow'
    algorithm = 'pge_id'
    base_circuit = 'pge_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            for i in idx.tolist():
                if int(clean[i].max()) < 1:
                    clean[i] = torch.randint(0, 10, (L,), generator=g)
                    continue
                p = int((clean[i] == clean[i].max()).long().argmax())
                corrupt[i] = clean[i]
                corrupt[i, p] = 0
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class CgpPmaxonlyCircuit(Cgpmax09, DomainCircuit):
    """cgp_pmaxonly on cgpmax09_id: isolated via compare_positions(0, 9) (integrated)."""
    circuit_id = 'cgp_pmaxonly'
    algorithm = 'cgpmax09_id'
    _cond = conditions.compare_positions(0, 9)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CgpCopyonlyCircuit(Cgpmax09, DomainCircuit):
    """cgp_copyonly on cgpmax09_id: isolated via Condition('cmp_0_le_9', lambda x: int(x[0]) <= int(x[9]), lambda batch, seed: (lambda blk: blk[blk[:, 0] <= blk[:, 9]][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'cgp_copyonly'))))) (integrated)."""
    circuit_id = 'cgp_copyonly'
    algorithm = 'cgpmax09_id'
    _cond = conditions.Condition('cmp_0_le_9', lambda x: int(x[0]) <= int(x[9]), lambda batch, seed: (lambda blk: blk[blk[:, 0] <= blk[:, 9]][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'cgp_copyonly')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CgpMonodecCircuit(Cgpmax09, DomainCircuit):
    """cgp_monodec on cgpmax09_id: isolated via monotone() (integrated)."""
    circuit_id = 'cgp_monodec'
    algorithm = 'cgpmax09_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CgpMonoincCircuit(Cgpmax09, DomainCircuit):
    """cgp_monoinc on cgpmax09_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'cgp_monoinc'
    algorithm = 'cgpmax09_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CgpBinaryCircuit(Cgpmax09, DomainCircuit):
    """cgp_binary on cgpmax09_id: isolated via BINARY (integrated)."""
    circuit_id = 'cgp_binary'
    algorithm = 'cgpmax09_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CgpFlipCircuit(Cgpmax09, ContrastCircuit):
    """cgp_flip on cgpmax09_id: contrast vs cgp_copyonly [threshold-straddle] (integrated)."""
    circuit_id = 'cgp_flip'
    algorithm = 'cgpmax09_id'
    base_circuit = 'cgp_copyonly'
    local_predicate = False
    _cond = conditions.Condition('cmp_0_le_9', lambda x: int(x[0]) <= int(x[9]), lambda batch, seed: (lambda blk: blk[blk[:, 0] <= blk[:, 9]][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'cgp_flip')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
            for i in idx.tolist():
                if int(clean[i, 0]) < 1:
                    clean[i] = DomainCircuit.sample(self, 1, int(seed) * 977 + i + t + 2)[0]
                    continue
                corrupt[i] = clean[i]
                corrupt[i, 9] = int(torch.randint(0, int(clean[i, 0]), (1,), generator=g))
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class LanesumWholeCircuit(Lanesum, DomainCircuit):
    """lanesum_whole on lanesum_id: isolated via FULL (integrated)."""
    circuit_id = 'lanesum_whole'
    algorithm = 'lanesum_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class LanesumAltEvenCircuit(Lanesum, ContrastCircuit):
    """lanesum_alt_even on lanesum_id: contrast vs lanesum_whole [digit-toggle] (integrated)."""
    circuit_id = 'lanesum_alt_even'
    algorithm = 'lanesum_id'
    base_circuit = 'lanesum_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = 0 + 2 * torch.randint(0, 5, (batch,), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        return (torch.arange(10) >= p) & (torch.arange(10) % 2 == p % 2)


class LanesumAltOddCircuit(Lanesum, ContrastCircuit):
    """lanesum_alt_odd on lanesum_id: contrast vs lanesum_whole [digit-toggle] (integrated)."""
    circuit_id = 'lanesum_alt_odd'
    algorithm = 'lanesum_id'
    base_circuit = 'lanesum_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = 1 + 2 * torch.randint(0, 5, (batch,), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        return (torch.arange(10) >= p) & (torch.arange(10) % 2 == p % 2)


class AddpmaxWholeCircuit(Addpmax, DomainCircuit):
    """addpmax_whole on addpmax_id: isolated via FULL (integrated)."""
    circuit_id = 'addpmax_whole'
    algorithm = 'addpmax_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class AddpmaxBelowMaxCircuit(Addpmax, ContrastCircuit):
    """addpmax_below_max on addpmax_id: contrast vs addpmax_whole [value-perturb-control] (integrated)."""
    circuit_id = 'addpmax_below_max'
    algorithm = 'addpmax_id'
    base_circuit = 'addpmax_whole'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(1, self.input_length, (batch,), generator=g)
        for _ in range(64):
            Mb = torch.cummax(clean, dim=1).values[rows, p - 1]
            bad = Mb < 2
            if not int(bad.any()): break
            clean[bad] = torch.randint(0, 10, (int(bad.sum()), self.input_length), generator=g)
        assert not int(bad.any())
        Mb = torch.cummax(clean, dim=1).values[rows, p - 1]
        a = torch.randint(0, 9, (batch,), generator=g) % Mb
        b = (a + 1 + torch.randint(0, 9, (batch,), generator=g) % (Mb - 1)) % Mb
        clean[rows, p] = a
        corrupt = clean.clone()
        corrupt[rows, p] = b
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) == int((clean_row != corrupt_row).nonzero()[0])


class AddpmaxNewMaxCircuit(Addpmax, ContrastCircuit):
    """addpmax_new_max on addpmax_id: contrast vs addpmax_whole [threshold-straddle] (integrated)."""
    circuit_id = 'addpmax_new_max'
    algorithm = 'addpmax_id'
    base_circuit = 'addpmax_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(1, self.input_length, (batch,), generator=g)
        for _ in range(64):
            Mb = torch.cummax(clean, dim=1).values[rows, p - 1]
            bad = (Mb < 1) | (Mb > 8)
            if not int(bad.any()): break
            clean[bad] = torch.randint(0, 10, (int(bad.sum()), self.input_length), generator=g)
        assert not int(bad.any())
        Mb = torch.cummax(clean, dim=1).values[rows, p - 1]
        clean[rows, p] = torch.randint(0, 9, (batch,), generator=g) % (Mb + 1)
        corrupt = clean.clone()
        corrupt[rows, p] = Mb + 1 + torch.randint(0, 9, (batch,), generator=g) % (9 - Mb)
        return clean, corrupt


class MaxrunOnepairCircuit(Maxrun, DomainCircuit):
    """maxrun_onepair on maxrun_id: isolated via Condition('one_adjacent_pair', lambda x: int((x[1:] == x[:-1]).sum()) == 1, lambda batch, seed: (lambda blk: blk[[int((r[1:] == r[:-1]).sum()) == 1 for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'maxrun_onepair'))))) (integrated)."""
    circuit_id = 'maxrun_onepair'
    algorithm = 'maxrun_id'
    _cond = conditions.Condition('one_adjacent_pair', lambda x: int((x[1:] == x[:-1]).sum()) == 1, lambda batch, seed: (lambda blk: blk[[int((r[1:] == r[:-1]).sum()) == 1 for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'maxrun_onepair')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class MaxrunExtendCircuit(Maxrun, ContrastCircuit):
    """maxrun_extend on maxrun_id: contrast vs maxrun_onepair [digit-toggle] (integrated)."""
    circuit_id = 'maxrun_extend'
    algorithm = 'maxrun_id'
    base_circuit = 'maxrun_onepair'
    local_predicate = False
    _cond = conditions.Condition('one_adjacent_pair', lambda x: int((x[1:] == x[:-1]).sum()) == 1, lambda batch, seed: (lambda blk: blk[[int((r[1:] == r[:-1]).sum()) == 1 for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'maxrun_extend')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        for i in range(batch):
            for k in range(64):
                q = int((clean[i][1:] == clean[i][:-1]).long().argmax())
                if q <= 6:
                    break
                clean[i] = DomainCircuit.sample(self, 1, int(seed) * 331 + i * 64 + k + 1)[0]
            q = int((clean[i][1:] == clean[i][:-1]).long().argmax())
            assert q <= 6
            corrupt[i] = clean[i]
            corrupt[i, q + 2] = int(clean[i, q])
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) >= int((clean_row != corrupt_row).nonzero()[0])


class MaxrunBreakCircuit(Maxrun, ContrastCircuit):
    """maxrun_break on maxrun_id: contrast vs maxrun_onepair [digit-toggle] (integrated)."""
    circuit_id = 'maxrun_break'
    algorithm = 'maxrun_id'
    base_circuit = 'maxrun_onepair'
    local_predicate = False
    _cond = conditions.Condition('one_adjacent_pair', lambda x: int((x[1:] == x[:-1]).sum()) == 1, lambda batch, seed: (lambda blk: blk[[int((r[1:] == r[:-1]).sum()) == 1 for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'maxrun_break')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        for i in range(batch):
            q = int((clean[i][1:] == clean[i][:-1]).long().argmax())
            ban = {int(clean[i, q])}
            if q + 2 < 10:
                ban.add(int(clean[i, q + 2]))
            if q > 0:
                ban.add(int(clean[i, q - 1]))
            ok = [v for v in range(10) if v not in ban]
            corrupt[i] = clean[i]
            corrupt[i, q + 1] = ok[int(torch.randint(0, len(ok), (1,), generator=g))]
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) >= int((clean_row != corrupt_row).nonzero()[0])


class DualtieNotieCircuit(Dualtie, DomainCircuit):
    """dualtie_notie on dualtie_id: isolated via Condition('no_extreme_tie', lambda x: all(int(x[i]) != int(x[:i].max()) and int(x[i]) != int(x[:i].min()) for i in range(1, 10)), lambda batch, seed: (lambda blk: blk[[all(int(r[i]) != int(r[:i].max()) and int(r[i]) != int(r[:i].min()) for i in range(1, 10)) for r in blk]][:batch])(torch.randint(0, 10, (16 * batch + 256, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'dualtie_notie'))))) (integrated)."""
    circuit_id = 'dualtie_notie'
    algorithm = 'dualtie_id'
    _cond = conditions.Condition('no_extreme_tie', lambda x: all(int(x[i]) != int(x[:i].max()) and int(x[i]) != int(x[:i].min()) for i in range(1, 10)), lambda batch, seed: (lambda blk: blk[[all(int(r[i]) != int(r[:i].max()) and int(r[i]) != int(r[:i].min()) for i in range(1, 10)) for r in blk]][:batch])(torch.randint(0, 10, (16 * batch + 256, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'dualtie_notie')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DualtieOnetieCircuit(Dualtie, DomainCircuit):
    """dualtie_onetie on dualtie_id: isolated via Condition('one_extreme_tie', lambda x: sum(1 for i in range(1, 10) if int(x[i]) == int(x[:i].max()) or int(x[i]) == int(x[:i].min())) == 1, lambda batch, seed: (lambda blk: blk[[sum(1 for i in range(1, 10) if int(r[i]) == int(r[:i].max()) or int(r[i]) == int(r[:i].min())) == 1 for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'dualtie_onetie'))))) (integrated)."""
    circuit_id = 'dualtie_onetie'
    algorithm = 'dualtie_id'
    _cond = conditions.Condition('one_extreme_tie', lambda x: sum(1 for i in range(1, 10) if int(x[i]) == int(x[:i].max()) or int(x[i]) == int(x[:i].min())) == 1, lambda batch, seed: (lambda blk: blk[[sum(1 for i in range(1, 10) if int(r[i]) == int(r[:i].max()) or int(r[i]) == int(r[:i].min())) == 1 for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'dualtie_onetie')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DualtieTiePlantCircuit(Dualtie, ContrastCircuit):
    """dualtie_tie_plant on dualtie_id: contrast vs dualtie_notie [threshold-straddle] (integrated)."""
    circuit_id = 'dualtie_tie_plant'
    algorithm = 'dualtie_id'
    base_circuit = 'dualtie_notie'
    local_predicate = True
    _cond = conditions.Condition('no_extreme_tie', lambda x: all(int(x[i]) != int(x[:i].max()) and int(x[i]) != int(x[:i].min()) for i in range(1, 10)), lambda batch, seed: (lambda blk: blk[[all(int(r[i]) != int(r[:i].max()) and int(r[i]) != int(r[:i].min()) for i in range(1, 10)) for r in blk]][:batch])(torch.randint(0, 10, (16 * batch + 256, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'dualtie_tie_plant')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        for i in range(batch):
            for k in range(64):
                opts = [p for p in range(1, 10)
                        if int(clean[i, :p].min()) < int(clean[i, p]) < int(clean[i, :p].max())
                        and int(clean[i, p]) != int(clean[i, :p].max())]
                if opts:
                    break
                clean[i] = DomainCircuit.sample(self, 1, int(seed) * 613 + i * 64 + k + 1)[0]
            assert opts
            p = opts[int(torch.randint(0, len(opts), (1,), generator=g))]
            corrupt[i] = clean[i]
            corrupt[i, p] = int(clean[i, :p].max())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) == int((clean_row != corrupt_row).nonzero()[0])


class Cap59NofireCircuit(Cap59, DomainCircuit):
    """cap59_nofire on cap59_id: isolated via Condition('at_most_4_high', lambda x: int((x >= 5).sum()) <= 4, lambda batch, seed: (lambda blk: blk[(blk >= 5).sum(dim=1) <= 4][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'cap59_nofire'))))) (integrated)."""
    circuit_id = 'cap59_nofire'
    algorithm = 'cap59_id'
    _cond = conditions.Condition('at_most_4_high', lambda x: int((x >= 5).sum()) <= 4, lambda batch, seed: (lambda blk: blk[(blk >= 5).sum(dim=1) <= 4][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'cap59_nofire')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Cap59TrippedCircuit(Cap59, DomainCircuit):
    """cap59_tripped on cap59_id: isolated via Condition('at_least_5_high', lambda x: int((x >= 5).sum()) >= 5, lambda batch, seed: (lambda blk: blk[(blk >= 5).sum(dim=1) >= 5][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'cap59_tripped'))))) (integrated)."""
    circuit_id = 'cap59_tripped'
    algorithm = 'cap59_id'
    _cond = conditions.Condition('at_least_5_high', lambda x: int((x >= 5).sum()) >= 5, lambda batch, seed: (lambda blk: blk[(blk >= 5).sum(dim=1) >= 5][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'cap59_tripped')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Cap59TripStraddleCircuit(Cap59, ContrastCircuit):
    """cap59_trip_straddle on cap59_id: contrast vs cap59_nofire [threshold-straddle] (integrated)."""
    circuit_id = 'cap59_trip_straddle'
    algorithm = 'cap59_id'
    base_circuit = 'cap59_nofire'
    local_predicate = False
    _cond = conditions.Condition('at_most_4_high', lambda x: int((x >= 5).sum()) <= 4, lambda batch, seed: (lambda blk: blk[(blk >= 5).sum(dim=1) <= 4][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'cap59_trip_straddle')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        perm = torch.argsort(torch.rand(batch, L, generator=g), dim=1)
        clean = torch.randint(0, 5, (batch, L), generator=g)
        rows = torch.arange(batch).unsqueeze(1)
        clean[rows, perm[:, :4]] = 5 + torch.randint(0, 5, (batch, 4), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            for i in idx.tolist():
                lows = (clean[i] < 5).nonzero().flatten()
                q = int(lows[int(torch.randint(0, lows.numel(), (1,), generator=g))])
                corrupt[i] = clean[i]
                corrupt[i, q] = int(clean[i, q]) + 5
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Cap59ReleaseCircuit(Cap59, ContrastCircuit):
    """cap59_release on cap59_id: contrast vs cap59_nofire [threshold-straddle] (integrated)."""
    circuit_id = 'cap59_release'
    algorithm = 'cap59_id'
    base_circuit = 'cap59_nofire'
    local_predicate = False
    _cond = conditions.Condition('at_most_4_high', lambda x: int((x >= 5).sum()) <= 4, lambda batch, seed: (lambda blk: blk[(blk >= 5).sum(dim=1) <= 4][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'cap59_release')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 5, (batch, L), generator=g)
        perm = torch.argsort(torch.rand(batch, 6, generator=g), dim=1)
        rows = torch.arange(batch).unsqueeze(1)
        clean[rows, perm[:, :4]] = 5 + torch.randint(0, 5, (batch, 4), generator=g)
        q = perm[:, 4]
        corrupt = clean.clone()
        corrupt[torch.arange(batch), q] = clean[torch.arange(batch), q] + 5
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        q = int((clean_row != corrupt_row).nonzero()[0])
        c = torch.cumsum((corrupt_row >= 5).long(), dim=0)
        t = int((c >= 5).long().argmax())
        m = (torch.arange(10) >= t) & (clean_row != 9)
        m[q] = True
        return m


class Cgrev09RevonlyCircuit(Cgrev09, DomainCircuit):
    """cgrev09_revonly on cgrev09_id: isolated via compare_positions(0, 9) (integrated)."""
    circuit_id = 'cgrev09_revonly'
    algorithm = 'cgrev09_id'
    _cond = conditions.compare_positions(0, 9)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Cgrev09CopyonlyCircuit(Cgrev09, DomainCircuit):
    """cgrev09_copyonly on cgrev09_id: isolated via Condition('cmp_0_le_9', lambda x: int(x[0]) <= int(x[9]), lambda batch, seed: (lambda blk: blk[blk[:, 0] <= blk[:, 9]][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'cgrev09_copyonly'))))) (integrated)."""
    circuit_id = 'cgrev09_copyonly'
    algorithm = 'cgrev09_id'
    _cond = conditions.Condition('cmp_0_le_9', lambda x: int(x[0]) <= int(x[9]), lambda batch, seed: (lambda blk: blk[blk[:, 0] <= blk[:, 9]][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'cgrev09_copyonly')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Cgrev09FlipbranchCircuit(Cgrev09, ContrastCircuit):
    """cgrev09_flipbranch on cgrev09_id: contrast vs cgrev09_copyonly [threshold-straddle] (integrated)."""
    circuit_id = 'cgrev09_flipbranch'
    algorithm = 'cgrev09_id'
    base_circuit = 'cgrev09_copyonly'
    local_predicate = False
    _cond = conditions.Condition('cmp_0_le_9', lambda x: int(x[0]) <= int(x[9]), lambda batch, seed: (lambda blk: blk[blk[:, 0] <= blk[:, 9]][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'cgrev09_flipbranch')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        a = 1 + torch.randint(0, 9, (batch,), generator=g)
        clean[:, 0] = a
        clean[:, 9] = a
        corrupt = clean.clone()
        corrupt[:, 9] = a - 1
        return clean, corrupt


class T2WholeCircuit(DoubleOdd, DomainCircuit):
    """t2_whole on dblodd_psum_novel_id: isolated via FULL (integrated)."""
    circuit_id = 't2_whole'
    algorithm = 'dblodd_psum_novel_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class T2FreshCircuit(DoubleOdd, ContrastCircuit):
    """t2_fresh on dblodd_psum_novel_id: contrast vs t2_whole [digit-toggle] (integrated)."""
    circuit_id = 't2_fresh'
    algorithm = 'dblodd_psum_novel_id'
    base_circuit = 't2_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            for i in idx.tolist():
                vals = clean[i].tolist()
                reps = [p for p in range(1, L) if vals[p] in vals[:p]]
                if not reps:
                    clean[i] = torch.randint(0, 10, (L,), generator=g)
                    continue
                p = reps[int(torch.randint(0, len(reps), (1,), generator=g))]
                absent = [v for v in range(10) if v not in vals]
                new = absent[int(torch.randint(0, len(absent), (1,), generator=g))] if absent else None
                if new is None:
                    continue
                corrupt[i] = clean[i]
                corrupt[i, p] = new
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class T2StealCircuit(DoubleOdd, ContrastCircuit):
    """t2_steal on dblodd_psum_novel_id: contrast vs t2_whole [compensated-two-site] (integrated)."""
    circuit_id = 't2_steal'
    algorithm = 'dblodd_psum_novel_id'
    base_circuit = 't2_whole'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            for i in idx.tolist():
                vals = clean[i].tolist()
                reps = [p for p in range(1, L) if vals[p] in vals[:p]]
                if not reps:
                    clean[i] = torch.randint(0, 10, (L,), generator=g)
                    continue
                p = reps[int(torch.randint(0, len(reps), (1,), generator=g))]
                later = [v for v in set(vals) if vals.index(v) > p]
                new = later[int(torch.randint(0, len(later), (1,), generator=g))] if later else None
                if new is None:
                    continue
                corrupt[i] = clean[i]
                corrupt[i, p] = new
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class DupmaxOnerepeatCircuit(Dupmax, DomainCircuit):
    """dupmax_onerepeat on dupmax_id: isolated via ONE_REPEAT (integrated)."""
    circuit_id = 'dupmax_onerepeat'
    algorithm = 'dupmax_id'
    _cond = conditions.ONE_REPEAT
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DupmaxBinaryCircuit(Dupmax, DomainCircuit):
    """dupmax_binary on dupmax_id: isolated via BINARY (integrated)."""
    circuit_id = 'dupmax_binary'
    algorithm = 'dupmax_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DupmaxMonoincCircuit(Dupmax, DomainCircuit):
    """dupmax_monoinc on dupmax_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'dupmax_monoinc'
    algorithm = 'dupmax_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DupmaxMonodecCircuit(Dupmax, DomainCircuit):
    """dupmax_monodec on dupmax_id: isolated via monotone() (integrated)."""
    circuit_id = 'dupmax_monodec'
    algorithm = 'dupmax_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DupmaxBreakCircuit(Dupmax, ContrastCircuit):
    """dupmax_break on dupmax_id: contrast vs dupmax_onerepeat [digit-toggle] (integrated)."""
    circuit_id = 'dupmax_break'
    algorithm = 'dupmax_id'
    base_circuit = 'dupmax_onerepeat'
    local_predicate = False
    _cond = conditions.ONE_REPEAT
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        for i in range(batch):
            for k in range(64):
                vals = clean[i].tolist()
                dup = next(v for v in vals if vals.count(v) == 2)
                if dup >= 1:
                    break
                clean[i] = DomainCircuit.sample(self, 1, int(seed) * 449 + i * 64 + k + 1)[0]
            vals = clean[i].tolist()
            dup = next(v for v in vals if vals.count(v) == 2)
            assert dup >= 1
            j2 = 9 - vals[::-1].index(dup)
            fresh = [v for v in range(10) if v not in vals]
            corrupt[i] = clean[i]
            corrupt[i, j2] = fresh[int(torch.randint(0, len(fresh), (1,), generator=g))]
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) >= int((clean_row != corrupt_row).nonzero()[0])


class DupmaxRaiseCircuit(Dupmax, ContrastCircuit):
    """dupmax_raise on dupmax_id: contrast vs dupmax_onerepeat [threshold-straddle] (integrated)."""
    circuit_id = 'dupmax_raise'
    algorithm = 'dupmax_id'
    base_circuit = 'dupmax_onerepeat'
    local_predicate = False
    _cond = conditions.ONE_REPEAT
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        for i in range(batch):
            for k in range(96):
                vals = clean[i].tolist()
                dup = next(v for v in vals if vals.count(v) == 2)
                j2 = 9 - vals[::-1].index(dup)
                singles = [v for v in set(vals) if vals.count(v) == 1 and v > dup]
                opts = [(kk, w) for w in singles for kk in range(j2 + 1, 10)
                        if kk != vals.index(w) and vals[kk] != dup]
                if opts:
                    break
                clean[i] = DomainCircuit.sample(self, 1, int(seed) * 457 + i * 96 + k + 1)[0]
            assert opts
            kk, w = opts[int(torch.randint(0, len(opts), (1,), generator=g))]
            corrupt[i] = clean[i]
            corrupt[i, kk] = w
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        k = int((clean_row != corrupt_row).nonzero()[0])
        w = int(corrupt_row[k])
        m = int((clean_row == w).long().argmax())
        return torch.arange(10) >= max(m, k)


class Cmpsh1FullCircuit(CompareShift, DomainCircuit):
    """cmpsh1_full on cmpsh1_id: isolated via FULL (integrated)."""
    circuit_id = 'cmpsh1_full'
    algorithm = 'cmpsh1_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Cmpsh1MakematchCircuit(CompareShift, ContrastCircuit):
    """cmpsh1_makematch on cmpsh1_id: contrast vs cmpsh1_full [digit-toggle] (integrated)."""
    circuit_id = 'cmpsh1_makematch'
    algorithm = 'cmpsh1_id'
    base_circuit = 'cmpsh1_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(1, self.input_length, (batch,), generator=g)
        for k in range(64):
            bad = clean[rows, p] == clean[rows, p - 1]
            if not int(bad.any()): break
            p[bad] = torch.randint(1, self.input_length, (int(bad.sum()),), generator=g)
            if k >= 8:
                clean[bad] = torch.randint(0, 10, (int(bad.sum()), self.input_length), generator=g)
        assert not int((clean[rows, p] == clean[rows, p - 1]).any())
        corrupt = clean.clone()
        corrupt[rows, p] = clean[rows, p - 1]
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool)
        m[p] = True
        if p + 1 < 10:
            m[p + 1] = bool(clean_row[p + 1] == clean_row[p]) != bool(corrupt_row[p + 1] == corrupt_row[p])
        return m


class Cmpsh1BreakmatchCircuit(CompareShift, ContrastCircuit):
    """cmpsh1_breakmatch on cmpsh1_id: contrast vs cmpsh1_full [digit-toggle] (integrated)."""
    circuit_id = 'cmpsh1_breakmatch'
    algorithm = 'cmpsh1_id'
    base_circuit = 'cmpsh1_full'
    local_predicate = True
    _cond = conditions.Condition('ge1match', lambda x: bool((x[1:] == x[:-1]).any()), lambda batch, seed: (lambda blk: blk[[bool((r[1:] == r[:-1]).any()) for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'cmpsh1_breakmatch')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        for i in range(batch):
            ms = (clean[i][1:] == clean[i][:-1]).nonzero().flatten() + 1
            q = int(ms[int(torch.randint(0, ms.numel(), (1,), generator=g))])
            ban = {int(clean[i, q - 1])}
            if q + 1 < 10:
                ban.add(int(clean[i, q + 1]))
            ok = [v for v in range(10) if v not in ban]
            corrupt[i] = clean[i]
            corrupt[i, q] = ok[int(torch.randint(0, len(ok), (1,), generator=g))]
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool)
        m[p] = True
        if p + 1 < 10:
            m[p + 1] = bool(clean_row[p + 1] == clean_row[p]) != bool(corrupt_row[p + 1] == corrupt_row[p])
        return m


class Rmaj5MonodecCircuit(Rmaj5, DomainCircuit):
    """rmaj5_monodec on rmaj5_id: isolated via monotone() (integrated)."""
    circuit_id = 'rmaj5_monodec'
    algorithm = 'rmaj5_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Rmaj5SwingCircuit(Rmaj5, ContrastCircuit):
    """rmaj5_swing on rmaj5_id: contrast vs rmaj5_monodec [threshold-straddle] (integrated)."""
    circuit_id = 'rmaj5_swing'
    algorithm = 'rmaj5_id'
    base_circuit = 'rmaj5_monodec'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            p = torch.randint(0, L, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                if int(clean[i, p[k]]) not in (4, 5):
                    continue
                corrupt[i] = clean[i]
                corrupt[i, p[k]] = 9 - int(clean[i, p[k]])
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class MajmaskNominorityCircuit(Majmask, DomainCircuit):
    """majmask_nominority on majmask_id: isolated via Condition('no_minority', lambda x: not bool((torch.where(x % 2 == 0, torch.cumsum((x % 2 == 0).long(), 0) - torch.cumsum((x % 2 == 1).long(), 0), torch.cumsum((x % 2 == 1).long(), 0) - torch.cumsum((x % 2 == 0).long(), 0)) < 0).any()), lambda batch, seed: (lambda blk: blk[[not bool((torch.where(r % 2 == 0, torch.cumsum((r % 2 == 0).long(), 0) - torch.cumsum((r % 2 == 1).long(), 0), torch.cumsum((r % 2 == 1).long(), 0) - torch.cumsum((r % 2 == 0).long(), 0)) < 0).any()) for r in blk]][:batch])(torch.randint(0, 10, (32 * batch + 512, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'majmask_nominority'))))) (integrated)."""
    circuit_id = 'majmask_nominority'
    algorithm = 'majmask_id'
    _cond = conditions.Condition('no_minority', lambda x: not bool((torch.where(x % 2 == 0, torch.cumsum((x % 2 == 0).long(), 0) - torch.cumsum((x % 2 == 1).long(), 0), torch.cumsum((x % 2 == 1).long(), 0) - torch.cumsum((x % 2 == 0).long(), 0)) < 0).any()), lambda batch, seed: (lambda blk: blk[[not bool((torch.where(r % 2 == 0, torch.cumsum((r % 2 == 0).long(), 0) - torch.cumsum((r % 2 == 1).long(), 0), torch.cumsum((r % 2 == 1).long(), 0) - torch.cumsum((r % 2 == 0).long(), 0)) < 0).any()) for r in blk]][:batch])(torch.randint(0, 10, (32 * batch + 512, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'majmask_nominority')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class MajmaskFlipCircuit(Majmask, ContrastCircuit):
    """majmask_flip on majmask_id: contrast vs majmask_nominority [digit-toggle] (integrated)."""
    circuit_id = 'majmask_flip'
    algorithm = 'majmask_id'
    base_circuit = 'majmask_nominority'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        start = torch.randint(0, 2, (batch, 1), generator=g)
        par = (start + torch.arange(L)) % 2
        clean = 2 * torch.randint(0, 5, (batch, L), generator=g) + par
        clean = torch.min(clean, torch.tensor(9))
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            p = torch.randint(0, 4, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                old = int(clean[i, p[k]])
                corrupt[i] = clean[i]
                corrupt[i, p[k]] = old + 1 if old % 2 == 0 else old - 1
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class R1cGe1matchCircuit(ResetRamp, DomainCircuit):
    """r1c_ge1match on reset1_cmpsh1_id: isolated via Condition('ge1match', lambda x: bool((x[1:] == x[:-1]).any()), lambda batch, seed: (lambda blk: blk[[bool((r[1:] == r[:-1]).any()) for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'r1c_ge1match'))))) (integrated)."""
    circuit_id = 'r1c_ge1match'
    algorithm = 'reset1_cmpsh1_id'
    _cond = conditions.Condition('ge1match', lambda x: bool((x[1:] == x[:-1]).any()), lambda batch, seed: (lambda blk: blk[[bool((r[1:] == r[:-1]).any()) for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'r1c_ge1match')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class R1cMakematchCircuit(ResetRamp, ContrastCircuit):
    """r1c_makematch on reset1_cmpsh1_id: contrast vs r1c_ge1match [digit-toggle] (integrated)."""
    circuit_id = 'r1c_makematch'
    algorithm = 'reset1_cmpsh1_id'
    base_circuit = 'r1c_ge1match'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(1, self.input_length, (batch,), generator=g)
        for k in range(64):
            bad = clean[rows, p] == clean[rows, p - 1]
            if not int(bad.any()): break
            p[bad] = torch.randint(1, self.input_length, (int(bad.sum()),), generator=g)
            if k >= 8:
                clean[bad] = torch.randint(0, 10, (int(bad.sum()), self.input_length), generator=g)
        assert not int((clean[rows, p] == clean[rows, p - 1]).any())
        corrupt = clean.clone()
        corrupt[rows, p] = clean[rows, p - 1]
        return clean, corrupt


class R1cBreakmatchCircuit(ResetRamp, ContrastCircuit):
    """r1c_breakmatch on reset1_cmpsh1_id: contrast vs r1c_ge1match [digit-toggle] (integrated)."""
    circuit_id = 'r1c_breakmatch'
    algorithm = 'reset1_cmpsh1_id'
    base_circuit = 'r1c_ge1match'
    local_predicate = False
    _cond = conditions.Condition('ge1match', lambda x: bool((x[1:] == x[:-1]).any()), lambda batch, seed: (lambda blk: blk[[bool((r[1:] == r[:-1]).any()) for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'r1c_breakmatch')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        for i in range(batch):
            ms = (clean[i][1:] == clean[i][:-1]).nonzero().flatten() + 1
            q = int(ms[int(torch.randint(0, ms.numel(), (1,), generator=g))])
            ban = {int(clean[i, q - 1])}
            if q + 1 < 10:
                ban.add(int(clean[i, q + 1]))
            ok = [v for v in range(10) if v not in ban]
            corrupt[i] = clean[i]
            corrupt[i, q] = ok[int(torch.randint(0, len(ok), (1,), generator=g))]
        return clean, corrupt


class R1nWholeCircuit(ResetRamp, DomainCircuit):
    """r1n_whole on reset1_novel_id: isolated via FULL (integrated)."""
    circuit_id = 'r1n_whole'
    algorithm = 'reset1_novel_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class R1nFreshCircuit(ResetRamp, ContrastCircuit):
    """r1n_fresh on reset1_novel_id: contrast vs r1n_whole [digit-toggle] (integrated)."""
    circuit_id = 'r1n_fresh'
    algorithm = 'reset1_novel_id'
    base_circuit = 'r1n_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        for i in range(batch):
            for k in range(64):
                vals = clean[i].tolist()
                reps = [p for p in range(1, L) if vals[p] in vals[:p]]
                fresh = [v for v in range(10) if v not in vals]
                if reps and fresh:
                    break
                clean[i] = torch.randint(0, 10, (L,), generator=g)
            assert reps and fresh
            p = reps[int(torch.randint(0, len(reps), (1,), generator=g))]
            corrupt[i] = clean[i]
            corrupt[i, p] = fresh[int(torch.randint(0, len(fresh), (1,), generator=g))]
        return clean, corrupt


class R1nStaleCircuit(ResetRamp, ContrastCircuit):
    """r1n_stale on reset1_novel_id: contrast vs r1n_whole [digit-toggle] (integrated)."""
    circuit_id = 'r1n_stale'
    algorithm = 'reset1_novel_id'
    base_circuit = 'r1n_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        for i in range(batch):
            for k in range(64):
                vals = clean[i].tolist()
                novs = [p for p in range(1, L) if vals[p] not in vals[:p]]
                if novs:
                    break
                clean[i] = torch.randint(0, 10, (L,), generator=g)
            assert novs
            p = novs[int(torch.randint(0, len(novs), (1,), generator=g))]
            earlier = list(set(clean[i][:p].tolist()))
            corrupt[i] = clean[i]
            corrupt[i, p] = earlier[int(torch.randint(0, len(earlier), (1,), generator=g))]
        return clean, corrupt


class PsumNovelWholeCircuit(PrefixSumMod, DomainCircuit):
    """psum_novel_whole on psum_novel_id: isolated via FULL (integrated)."""
    circuit_id = 'psum_novel_whole'
    algorithm = 'psum_novel_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PsumNovelFreshCircuit(PrefixSumMod, ContrastCircuit):
    """psum_novel_fresh on psum_novel_id: contrast vs psum_novel_whole [digit-toggle] (integrated)."""
    circuit_id = 'psum_novel_fresh'
    algorithm = 'psum_novel_id'
    base_circuit = 'psum_novel_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        for i in range(batch):
            for k in range(64):
                vals = clean[i].tolist()
                reps = [p for p in range(1, L) if vals[p] in vals[:p]]
                fresh = [v for v in range(10) if v not in vals]
                if reps and fresh:
                    break
                clean[i] = torch.randint(0, 10, (L,), generator=g)
            assert reps and fresh
            p = reps[int(torch.randint(0, len(reps), (1,), generator=g))]
            corrupt[i] = clean[i]
            corrupt[i, p] = fresh[int(torch.randint(0, len(fresh), (1,), generator=g))]
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) >= int((clean_row != corrupt_row).nonzero()[0])


class PsumNovelStealCircuit(PrefixSumMod, ContrastCircuit):
    """psum_novel_steal on psum_novel_id: contrast vs psum_novel_whole [compensated-two-site] (integrated)."""
    circuit_id = 'psum_novel_steal'
    algorithm = 'psum_novel_id'
    base_circuit = 'psum_novel_whole'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        for i in range(batch):
            for k in range(64):
                vals = clean[i].tolist()
                opts = [(p, w) for p in range(1, L) if vals[p] in vals[:p]
                        for w in set(vals[p + 1:]) if w not in vals[:p + 1]]
                if opts:
                    break
                clean[i] = torch.randint(0, 10, (L,), generator=g)
            assert opts
            p, w = opts[int(torch.randint(0, len(opts), (1,), generator=g))]
            corrupt[i] = clean[i]
            corrupt[i, p] = w
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        w = int(corrupt_row[p])
        j = 10
        for i in range(p + 1, 10):
            if int(clean_row[i]) == w:
                j = i
                break
        return (torch.arange(10) >= p) & (torch.arange(10) < j)


class Diffsh1FullCircuit(DiffShift, DomainCircuit):
    """diffsh1_full on diffsh1_id: isolated via FULL (integrated)."""
    circuit_id = 'diffsh1_full'
    algorithm = 'diffsh1_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Diffsh1EditCircuit(DiffShift, ContrastCircuit):
    """diffsh1_edit on diffsh1_id: contrast vs diffsh1_full [digit-toggle] (integrated)."""
    circuit_id = 'diffsh1_edit'
    algorithm = 'diffsh1_id'
    base_circuit = 'diffsh1_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool)
        m[p] = True
        if p + 1 < 10:
            m[p + 1] = True
        return m


class Bcast0FullCircuit(AnchorBroadcast, DomainCircuit):
    """bcast0_full on bcast0_id: isolated via FULL (integrated)."""
    circuit_id = 'bcast0_full'
    algorithm = 'bcast0_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Bcast0AnchoreditCircuit(AnchorBroadcast, ContrastCircuit):
    """bcast0_anchoredit on bcast0_id: contrast vs bcast0_full [digit-toggle] (integrated)."""
    circuit_id = 'bcast0_anchoredit'
    algorithm = 'bcast0_id'
    base_circuit = 'bcast0_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[:, 0] = (clean[:, 0] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.ones(10, dtype=torch.bool)


class ZpeakMonotoneCircuit(Zpeak, DomainCircuit):
    """zpeak_monotone on zpeak_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'zpeak_monotone'
    algorithm = 'zpeak_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class ZpeakOnepeakCircuit(Zpeak, DomainCircuit):
    """zpeak_onepeak on zpeak_id: isolated via Condition('one_interior_peak', lambda x: sum(1 for i in range(1, 9) if int(x[i]) > int(x[i - 1]) and int(x[i]) > int(x[i + 1])) == 1 and not int(x[0]) > int(x[1]) and not int(x[9]) > int(x[8]), lambda batch, seed: (lambda blk: blk[[sum(1 for i in range(1, 9) if int(r[i]) > int(r[i - 1]) and int(r[i]) > int(r[i + 1])) == 1 and not int(r[0]) > int(r[1]) and not int(r[9]) > int(r[8]) for r in blk]][:batch])(torch.randint(0, 10, (96 * batch + 1024, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'zpeak_onepeak'))))) (integrated)."""
    circuit_id = 'zpeak_onepeak'
    algorithm = 'zpeak_id'
    _cond = conditions.Condition('one_interior_peak', lambda x: sum(1 for i in range(1, 9) if int(x[i]) > int(x[i - 1]) and int(x[i]) > int(x[i + 1])) == 1 and not int(x[0]) > int(x[1]) and not int(x[9]) > int(x[8]), lambda batch, seed: (lambda blk: blk[[sum(1 for i in range(1, 9) if int(r[i]) > int(r[i - 1]) and int(r[i]) > int(r[i + 1])) == 1 and not int(r[0]) > int(r[1]) and not int(r[9]) > int(r[8]) for r in blk]][:batch])(torch.randint(0, 10, (96 * batch + 1024, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'zpeak_onepeak')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class ZpeakDestroyCircuit(Zpeak, ContrastCircuit):
    """zpeak_destroy on zpeak_id: contrast vs zpeak_onepeak [threshold-straddle] (integrated)."""
    circuit_id = 'zpeak_destroy'
    algorithm = 'zpeak_id'
    base_circuit = 'zpeak_onepeak'
    local_predicate = True
    _cond = conditions.Condition('one_interior_peak', lambda x: sum(1 for i in range(1, 9) if int(x[i]) > int(x[i - 1]) and int(x[i]) > int(x[i + 1])) == 1 and not int(x[0]) > int(x[1]) and not int(x[9]) > int(x[8]), lambda batch, seed: (lambda blk: blk[[sum(1 for i in range(1, 9) if int(r[i]) > int(r[i - 1]) and int(r[i]) > int(r[i + 1])) == 1 and not int(r[0]) > int(r[1]) and not int(r[9]) > int(r[8]) for r in blk]][:batch])(torch.randint(0, 10, (96 * batch + 1024, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'zpeak_destroy')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        for i in range(batch):
            for k in range(64):
                r = clean[i].tolist()
                q = next(j for j in range(1, 9) if r[j] > r[j - 1] and r[j] > r[j + 1])
                if r[q] <= 8:
                    break
                clean[i] = DomainCircuit.sample(self, 1, int(seed) * 769 + i * 64 + k + 1)[0]
            r = clean[i].tolist()
            q = next(j for j in range(1, 9) if r[j] > r[j - 1] and r[j] > r[j + 1])
            assert r[q] <= 8
            corrupt[i] = clean[i]
            corrupt[i, q - 1] = r[q] + 1 + int(torch.randint(0, 9 - r[q], (1,), generator=g))
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        e = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool)
        m[e + 1] = True
        left = int(corrupt_row[e - 1]) if e > 0 else -1
        newpk = int(corrupt_row[e]) > left and int(corrupt_row[e]) > int(corrupt_row[e + 1])
        m[e] = (0 if newpk else int(corrupt_row[e])) != int(clean_row[e])
        return m


class Mulanc0Caesar3Circuit(Mulanc0, DomainCircuit):
    """mulanc0_caesar3 on mulanc0_id: isolated via pin(0, [3]) (integrated)."""
    circuit_id = 'mulanc0_caesar3'
    algorithm = 'mulanc0_id'
    _cond = conditions.pin(0, [3])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Mulanc0AnchorshiftSparseCircuit(Mulanc0, ContrastCircuit):
    """mulanc0_anchorshift_sparse on mulanc0_id: contrast vs mulanc0_caesar3 [digit-toggle] (integrated)."""
    circuit_id = 'mulanc0_anchorshift_sparse'
    algorithm = 'mulanc0_id'
    base_circuit = 'mulanc0_caesar3'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        units = torch.tensor([1, 3, 7, 9])
        clean = 2 * torch.randint(0, 5, (batch, L), generator=g)
        clean[:, 0] = units[torch.randint(0, 4, (batch,), generator=g)]
        o1 = 2 + torch.randint(0, 6, (batch,), generator=g)
        o2 = o1 + 2 + torch.randint(0, 8, (batch,), generator=g) % (8 - o1)
        rows = torch.arange(batch)
        odd = 2 * torch.randint(0, 5, (batch, 2), generator=g) + 1
        clean[rows, o1] = odd[:, 0]
        clean[rows, o2] = odd[:, 1]
        corrupt = clean.clone()
        corrupt[:, 0] = (clean[:, 0] + 5) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        m = clean_row % 2 == 1
        m[0] = True
        return m


class Mulanc0AnchorshiftFullCircuit(Mulanc0, ContrastCircuit):
    """mulanc0_anchorshift_full on mulanc0_id: contrast vs mulanc0_caesar3 [digit-toggle] (integrated)."""
    circuit_id = 'mulanc0_anchorshift_full'
    algorithm = 'mulanc0_id'
    base_circuit = 'mulanc0_caesar3'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        corrupt = clean.clone()
        corrupt[:, 0] = (clean[:, 0] + 5) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        m = clean_row % 2 == 1
        m[0] = True
        return m


class RecdistWholeCircuit(Recdist, DomainCircuit):
    """recdist_whole on recdist_id: isolated via FULL (integrated)."""
    circuit_id = 'recdist_whole'
    algorithm = 'recdist_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class RecdistPlantRecordCircuit(Recdist, ContrastCircuit):
    """recdist_plant_record on recdist_id: contrast vs recdist_whole [threshold-straddle] (integrated)."""
    circuit_id = 'recdist_plant_record'
    algorithm = 'recdist_id'
    base_circuit = 'recdist_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 9, (batch, L), generator=g)
        rows = torch.arange(batch)
        q = torch.randint(1, L, (batch,), generator=g)
        for _ in range(64):
            Mb = torch.cummax(clean, dim=1).values[rows, q - 1]
            bad = clean[rows, q] > Mb
            if not int(bad.any()): break
            q[bad] = torch.randint(1, L, (int(bad.sum()),), generator=g)
        assert not int(bad.any())
        corrupt = clean.clone()
        corrupt[rows, q] = 9
        return clean, corrupt


class PmrepNorepeatCircuit(Pmrep, DomainCircuit):
    """pmrep_norepeat on pmrep_id: isolated via Condition('no_adjacent_repeat', lambda x: not bool((x[1:] == x[:-1]).any()), lambda batch, seed: (lambda blk: blk[[not bool((r[1:] == r[:-1]).any()) for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'pmrep_norepeat'))))) (integrated)."""
    circuit_id = 'pmrep_norepeat'
    algorithm = 'pmrep_id'
    _cond = conditions.Condition('no_adjacent_repeat', lambda x: not bool((x[1:] == x[:-1]).any()), lambda batch, seed: (lambda blk: blk[[not bool((r[1:] == r[:-1]).any()) for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'pmrep_norepeat')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PmrepOnerepeatCircuit(Pmrep, DomainCircuit):
    """pmrep_onerepeat on pmrep_id: isolated via Condition('one_adjacent_pair', lambda x: int((x[1:] == x[:-1]).sum()) == 1, lambda batch, seed: (lambda blk: blk[[int((r[1:] == r[:-1]).sum()) == 1 for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'pmrep_onerepeat'))))) (integrated)."""
    circuit_id = 'pmrep_onerepeat'
    algorithm = 'pmrep_id'
    _cond = conditions.Condition('one_adjacent_pair', lambda x: int((x[1:] == x[:-1]).sum()) == 1, lambda batch, seed: (lambda blk: blk[[int((r[1:] == r[:-1]).sum()) == 1 for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'pmrep_onerepeat')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PmrepMakerepCircuit(Pmrep, ContrastCircuit):
    """pmrep_makerep on pmrep_id: contrast vs pmrep_norepeat [digit-toggle] (integrated)."""
    circuit_id = 'pmrep_makerep'
    algorithm = 'pmrep_id'
    base_circuit = 'pmrep_norepeat'
    local_predicate = True
    _cond = conditions.Condition('no_adjacent_repeat', lambda x: not bool((x[1:] == x[:-1]).any()), lambda batch, seed: (lambda blk: blk[[not bool((r[1:] == r[:-1]).any()) for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'pmrep_makerep')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
            p = torch.randint(1, L, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                corrupt[i] = clean[i]
                corrupt[i, p[k]] = int(clean[i, p[k] - 1])
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Rcnt1Cmpsh1Ge1matchCircuit(RunningCountDigit, DomainCircuit):
    """rcnt1_cmpsh1_ge1match on rcnt1_cmpsh1_id: isolated via Condition('ge1match', lambda x: bool((x[1:] == x[:-1]).any()), lambda batch, seed: (lambda blk: blk[[bool((r[1:] == r[:-1]).any()) for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'rcnt1_cmpsh1_ge1match'))))) (integrated)."""
    circuit_id = 'rcnt1_cmpsh1_ge1match'
    algorithm = 'rcnt1_cmpsh1_id'
    _cond = conditions.Condition('ge1match', lambda x: bool((x[1:] == x[:-1]).any()), lambda batch, seed: (lambda blk: blk[[bool((r[1:] == r[:-1]).any()) for r in blk]][:batch])(torch.randint(0, 10, (8 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'rcnt1_cmpsh1_ge1match')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Rcnt1Cmpsh1MakematchCircuit(RunningCountDigit, ContrastCircuit):
    """rcnt1_cmpsh1_makematch on rcnt1_cmpsh1_id: contrast vs rcnt1_cmpsh1_ge1match [digit-toggle] (integrated)."""
    circuit_id = 'rcnt1_cmpsh1_makematch'
    algorithm = 'rcnt1_cmpsh1_id'
    base_circuit = 'rcnt1_cmpsh1_ge1match'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(1, self.input_length, (batch,), generator=g)
        for k in range(64):
            bad = clean[rows, p] == clean[rows, p - 1]
            if not int(bad.any()): break
            p[bad] = torch.randint(1, self.input_length, (int(bad.sum()),), generator=g)
            if k >= 8:
                clean[bad] = torch.randint(0, 10, (int(bad.sum()), self.input_length), generator=g)
        assert not int((clean[rows, p] == clean[rows, p - 1]).any())
        corrupt = clean.clone()
        corrupt[rows, p] = clean[rows, p - 1]
        return clean, corrupt


class H5dGe1qualCircuit(SampleHold, DomainCircuit):
    """h5d_ge1qual on holdge5_diffsh1_id: isolated via Condition('ge1_bigdiff', lambda x: bool((torch.cat([x[:1], (x[1:] - x[:-1]) % 10]) >= 5).any()), lambda batch, seed: (lambda blk: blk[[bool((torch.cat([r[:1], (r[1:] - r[:-1]) % 10]) >= 5).any()) for r in blk]][:batch])(torch.randint(0, 10, (2 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'h5d_ge1qual'))))) (integrated)."""
    circuit_id = 'h5d_ge1qual'
    algorithm = 'holdge5_diffsh1_id'
    _cond = conditions.Condition('ge1_bigdiff', lambda x: bool((torch.cat([x[:1], (x[1:] - x[:-1]) % 10]) >= 5).any()), lambda batch, seed: (lambda blk: blk[[bool((torch.cat([r[:1], (r[1:] - r[:-1]) % 10]) >= 5).any()) for r in blk]][:batch])(torch.randint(0, 10, (2 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'h5d_ge1qual')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class H5dThresholdCrossCircuit(SampleHold, ContrastCircuit):
    """h5d_threshold_cross on holdge5_diffsh1_id: contrast vs h5d_ge1qual [threshold-straddle] (integrated)."""
    circuit_id = 'h5d_threshold_cross'
    algorithm = 'holdge5_diffsh1_id'
    base_circuit = 'h5d_ge1qual'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            p = torch.randint(1, L, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                dprev = (int(clean[i, p[k]]) - int(clean[i, p[k] - 1])) % 10
                if dprev >= 5:
                    continue
                nd = 5 + int(torch.randint(0, 5, (1,), generator=g))
                corrupt[i] = clean[i]
                corrupt[i, p[k]] = (int(clean[i, p[k] - 1]) + nd) % 10
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class H5dQualValueCircuit(SampleHold, ContrastCircuit):
    """h5d_qual_value on holdge5_diffsh1_id: contrast vs h5d_ge1qual [value-perturb-control] (integrated)."""
    circuit_id = 'h5d_qual_value'
    algorithm = 'holdge5_diffsh1_id'
    base_circuit = 'h5d_ge1qual'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            p = torch.randint(1, L, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                dprev = (int(clean[i, p[k]]) - int(clean[i, p[k] - 1])) % 10
                if dprev < 5:
                    continue
                nd = 5 + int(torch.randint(0, 5, (1,), generator=g))
                if nd == dprev:
                    continue
                corrupt[i] = clean[i]
                corrupt[i, p[k]] = (int(clean[i, p[k] - 1]) + nd) % 10
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class MulpairFullCircuit(Mulpair, DomainCircuit):
    """mulpair_full on mulpair_id: isolated via FULL (integrated)."""
    circuit_id = 'mulpair_full'
    algorithm = 'mulpair_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class MulpairPerturbCircuit(Mulpair, ContrastCircuit):
    """mulpair_perturb on mulpair_id: contrast vs mulpair_full [value-perturb-control] (integrated)."""
    circuit_id = 'mulpair_perturb'
    algorithm = 'mulpair_id'
    base_circuit = 'mulpair_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), self.input_length), generator=g)
            p = torch.randint(1, 9, (idx.numel(),), generator=g)
            new = torch.randint(0, 10, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                if int(new[k]) == int(clean[i, p[k]]): continue
                pp = int(p[k])
                c0 = (int(clean[i, pp - 1]) * int(clean[i, pp])) % 10 != (int(clean[i, pp - 1]) * int(new[k])) % 10
                c1 = (int(clean[i, pp]) * int(clean[i, pp + 1])) % 10 != (int(new[k]) * int(clean[i, pp + 1])) % 10
                if not (c0 or c1): continue
                corrupt[i] = clean[i]
                corrupt[i, pp] = new[k]
                need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool)
        m[p] = (int(clean_row[p - 1]) * int(clean_row[p])) % 10 != (int(clean_row[p - 1]) * int(corrupt_row[p])) % 10
        if p + 1 < 10:
            m[p + 1] = (int(clean_row[p]) * int(clean_row[p + 1])) % 10 != (int(corrupt_row[p]) * int(clean_row[p + 1])) % 10
        return m


class PrevsegmaxOnethreeCircuit(PrevSegmentMax, DomainCircuit):
    """prevsegmax_onethree on prevsegmax3_id: isolated via exactly_one_in([3]) (integrated)."""
    circuit_id = 'prevsegmax_onethree'
    algorithm = 'prevsegmax3_id'
    _cond = conditions.exactly_one_in([3])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PrevsegmaxMaxraiseCircuit(PrevSegmentMax, ContrastCircuit):
    """prevsegmax_maxraise on prevsegmax3_id: contrast vs prevsegmax_onethree [threshold-straddle] (integrated)."""
    circuit_id = 'prevsegmax_maxraise'
    algorithm = 'prevsegmax3_id'
    base_circuit = 'prevsegmax_onethree'
    local_predicate = False
    _cond = conditions.exactly_one_in([3])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
            for i in idx.tolist():
                b = int((clean[i] == 3).nonzero()[0])
                seg1 = [j for j in range(b) if int(clean[i, j]) != 3]
                if not seg1:
                    clean[i] = DomainCircuit.sample(self, 1, int(seed) * 811 + i + t + 2)[0]
                    continue
                mx = int(max(int(clean[i, j]) for j in seg1))
                q = [j for j in seg1 if int(clean[i, j]) == mx][0]
                corrupt[i] = clean[i]
                corrupt[i, q] = int(torch.randint(0, mx if mx > 0 else 1, (1,), generator=g))
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class PrevsegmaxBoundarydeleteCircuit(PrevSegmentMax, ContrastCircuit):
    """prevsegmax_boundarydelete on prevsegmax3_id: contrast vs prevsegmax_onethree [digit-toggle] (integrated)."""
    circuit_id = 'prevsegmax_boundarydelete'
    algorithm = 'prevsegmax3_id'
    base_circuit = 'prevsegmax_onethree'
    local_predicate = False
    _cond = conditions.Condition('ge2_threes', lambda x: int((x == 3).sum()) >= 2, lambda batch, seed: (lambda blk: blk[(blk == 3).sum(1) >= 2][:batch])(torch.randint(0, 10, (24 * batch + 256, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'prevsegmax_boundarydelete')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                b = int((clean[i] == 3).nonzero()[0])
                corrupt[i] = clean[i]
                corrupt[i, b] = 7
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Rot9FullCircuit(Rotate, DomainCircuit):
    """rot9_full on rot9_id: isolated via FULL (integrated)."""
    circuit_id = 'rot9_full'
    algorithm = 'rot9_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Rot9EditCircuit(Rotate, ContrastCircuit):
    """rot9_edit on rot9_id: contrast vs rot9_full [digit-toggle] (integrated)."""
    circuit_id = 'rot9_edit'
    algorithm = 'rot9_id'
    base_circuit = 'rot9_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool)
        m[(p + 1) % 10] = True
        return m


class Bmax2FullCircuit(WindowMax, DomainCircuit):
    """bmax2_full on bmax2_id: isolated via FULL (integrated)."""
    circuit_id = 'bmax2_full'
    algorithm = 'bmax2_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Bmax2AscendCircuit(WindowMax, DomainCircuit):
    """bmax2_ascend on bmax2_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'bmax2_ascend'
    algorithm = 'bmax2_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Bmax2DelayCircuit(WindowMax, DomainCircuit):
    """bmax2_delay on bmax2_id: isolated via monotone() (integrated)."""
    circuit_id = 'bmax2_delay'
    algorithm = 'bmax2_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Bmax2Raise9Circuit(WindowMax, ContrastCircuit):
    """bmax2_raise9 on bmax2_id: contrast vs bmax2_full [threshold-straddle] (integrated)."""
    circuit_id = 'bmax2_raise9'
    algorithm = 'bmax2_id'
    base_circuit = 'bmax2_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 9, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = 9
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        hi = min(p + 2, 9)
        m = torch.zeros(10, dtype=torch.bool)
        m[p:hi + 1] = True
        return m


class Bmax3FullCircuit(WindowMax, DomainCircuit):
    """bmax3_full on bmax3_id: isolated via FULL (integrated)."""
    circuit_id = 'bmax3_full'
    algorithm = 'bmax3_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Bmax3AscendCircuit(WindowMax, DomainCircuit):
    """bmax3_ascend on bmax3_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'bmax3_ascend'
    algorithm = 'bmax3_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Bmax3DelayCircuit(WindowMax, DomainCircuit):
    """bmax3_delay on bmax3_id: isolated via monotone() (integrated)."""
    circuit_id = 'bmax3_delay'
    algorithm = 'bmax3_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Bmax3Raise9Circuit(WindowMax, ContrastCircuit):
    """bmax3_raise9 on bmax3_id: contrast vs bmax3_full [threshold-straddle] (integrated)."""
    circuit_id = 'bmax3_raise9'
    algorithm = 'bmax3_id'
    base_circuit = 'bmax3_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 9, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = 9
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        hi = min(p + 3, 9)
        m = torch.zeros(10, dtype=torch.bool)
        m[p:hi + 1] = True
        return m


class OcccntWholeCircuit(Occcnt, DomainCircuit):
    """occcnt_whole on occcnt_id: isolated via FULL (integrated)."""
    circuit_id = 'occcnt_whole'
    algorithm = 'occcnt_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class OcccntBinary12Circuit(Occcnt, DomainCircuit):
    """occcnt_binary12 on occcnt_id: isolated via domain([1, 2]) (integrated)."""
    circuit_id = 'occcnt_binary12'
    algorithm = 'occcnt_id'
    _cond = conditions.domain([1, 2])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class OcccntValeditCircuit(Occcnt, ContrastCircuit):
    """occcnt_valedit on occcnt_id: contrast vs occcnt_whole [value-perturb-control] (integrated)."""
    circuit_id = 'occcnt_valedit'
    algorithm = 'occcnt_id'
    base_circuit = 'occcnt_whole'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        def occ(r):
            return [sum(1 for j in range(i) if r[j] == r[i]) for i in range(L)]
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            p = torch.randint(0, L, (idx.numel(),), generator=g)
            new = torch.randint(0, 10, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                if int(new[k]) == int(clean[i, p[k]]): continue
                c = clean[i].clone(); c[p[k]] = new[k]
                if occ(clean[i].tolist()) == occ(c.tolist()): continue
                corrupt[i] = c
                need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Segdist3OnethreeCircuit(SegmentDistinct, DomainCircuit):
    """segdist3_onethree on segdist3_id: isolated via exactly_one_in([3]) (integrated)."""
    circuit_id = 'segdist3_onethree'
    algorithm = 'segdist3_id'
    _cond = conditions.exactly_one_in([3])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Segdist3NoveltyCircuit(SegmentDistinct, ContrastCircuit):
    """segdist3_novelty on segdist3_id: contrast vs segdist3_onethree [digit-toggle] (integrated)."""
    circuit_id = 'segdist3_novelty'
    algorithm = 'segdist3_id'
    base_circuit = 'segdist3_onethree'
    local_predicate = False
    _cond = conditions.exactly_one_in([3])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
            for i in idx.tolist():
                b = int((clean[i] == 3).nonzero()[0])
                tail = list(range(b + 1, L))
                reps = [p for p in tail if int(clean[i, p]) in [int(clean[i, q]) for q in range(b + 1, p)]]
                if not reps:
                    clean[i] = DomainCircuit.sample(self, 1, int(seed) * 733 + i + t + 2)[0]
                    continue
                p = reps[int(torch.randint(0, len(reps), (1,), generator=g))]
                seg = set(int(clean[i, q]) for q in range(b + 1, L))
                fresh = [v for v in range(10) if v != 3 and v not in seg]
                if not fresh:
                    clean[i] = DomainCircuit.sample(self, 1, int(seed) * 739 + i + t + 3)[0]
                    continue
                corrupt[i] = clean[i]
                corrupt[i, p] = fresh[int(torch.randint(0, len(fresh), (1,), generator=g))]
                need[i] = False
        assert not int(need.any())
        return clean, corrupt


class DclampPinbandCircuit(Dclamp, DomainCircuit):
    """dclamp_pinband on dclamp_id: isolated via Condition('pin0_0_1_9', lambda x: int(x[0]) == 0 and int(x[1]) == 9, conditions.position_domain_sampler({0: [0], 1: [9]})) (integrated)."""
    circuit_id = 'dclamp_pinband'
    algorithm = 'dclamp_id'
    _cond = conditions.Condition('pin0_0_1_9', lambda x: int(x[0]) == 0 and int(x[1]) == 9, conditions.position_domain_sampler({0: [0], 1: [9]}))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DclampSideflipCircuit(Dclamp, ContrastCircuit):
    """dclamp_sideflip on dclamp_id: contrast vs dclamp_pinband [threshold-straddle] (integrated)."""
    circuit_id = 'dclamp_sideflip'
    algorithm = 'dclamp_id'
    base_circuit = 'dclamp_pinband'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            for i in idx.tolist():
                found = False
                for p in range(L - 1, 0, -1):
                    lo, hi = int(clean[i, :p].min()), int(clean[i, :p].max())
                    if hi - lo >= 2:
                        corrupt[i] = clean[i]
                        corrupt[i, p] = hi + 1 if hi < 9 else lo - 1
                        if hi + 1 > 9:
                            corrupt[i, p] = lo - 1
                        if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                            need[i] = False
                            found = True
                        break
                if not found and need[i]:
                    clean[i] = torch.randint(0, 10, (L,), generator=g)
        assert not int(need.any())
        return clean, corrupt


class FuAlldistinctCircuit(FirstUnique, DomainCircuit):
    """fu_alldistinct on firstuniq_id: isolated via Condition('all_distinct', lambda x: len(set(x.tolist())) == 10, lambda batch, seed: torch.argsort(torch.rand(batch, 10, generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'fu_alldistinct'))), dim=1)) (integrated)."""
    circuit_id = 'fu_alldistinct'
    algorithm = 'firstuniq_id'
    _cond = conditions.Condition('all_distinct', lambda x: len(set(x.tolist())) == 10, lambda batch, seed: torch.argsort(torch.rand(batch, 10, generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'fu_alldistinct'))), dim=1))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class FuKnockoutCircuit(FirstUnique, ContrastCircuit):
    """fu_knockout on firstuniq_id: contrast vs fu_alldistinct [digit-toggle] (integrated)."""
    circuit_id = 'fu_knockout'
    algorithm = 'firstuniq_id'
    base_circuit = 'fu_alldistinct'
    local_predicate = False
    _cond = conditions.Condition('all_distinct', lambda x: len(set(x.tolist())) == 10, lambda batch, seed: torch.argsort(torch.rand(batch, 10, generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'fu_knockout'))), dim=1))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        rows = torch.arange(batch)
        p = torch.randint(2, L, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = clean[rows, 0]
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        return torch.arange(10) >= p


class RunmedSortedCircuit(Runmed, DomainCircuit):
    """runmed_sorted on runmed_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'runmed_sorted'
    algorithm = 'runmed_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class RunmedCrossCircuit(Runmed, ContrastCircuit):
    """runmed_cross on runmed_id: contrast vs runmed_sorted [threshold-straddle] (integrated)."""
    circuit_id = 'runmed_cross'
    algorithm = 'runmed_id'
    base_circuit = 'runmed_sorted'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            p = torch.randint(0, L, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                if int(clean[i, p[k]]) > 2: continue
                corrupt[i] = clean[i]
                corrupt[i, p[k]] = 7 + int(torch.randint(0, 3, (1,), generator=g))
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Cntge3WholeCircuit(Cntge3, DomainCircuit):
    """cntge3_whole on cntge3_id: isolated via FULL (integrated)."""
    circuit_id = 'cntge3_whole'
    algorithm = 'cntge3_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Cntge3LowclassCircuit(Cntge3, DomainCircuit):
    """cntge3_lowclass on cntge3_id: isolated via domain([0, 1, 2, 3]) (integrated)."""
    circuit_id = 'cntge3_lowclass'
    algorithm = 'cntge3_id'
    _cond = conditions.domain([0, 1, 2, 3])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Cg3PromoteCircuit(Cntge3, ContrastCircuit):
    """cg3_promote on cntge3_id: contrast vs cntge3_whole [threshold-straddle] (integrated)."""
    circuit_id = 'cg3_promote'
    algorithm = 'cntge3_id'
    base_circuit = 'cntge3_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 3, (batch, L), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, L, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = 3 + torch.randint(0, 7, (batch,), generator=g)
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) >= int((clean_row != corrupt_row).nonzero()[0])


class TsWholeCircuit(Twosum10, DomainCircuit):
    """ts_whole on twosum10_id: isolated via FULL (integrated)."""
    circuit_id = 'ts_whole'
    algorithm = 'twosum10_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Ts05Circuit(Twosum10, DomainCircuit):
    """ts_05 on twosum10_id: isolated via domain([0, 5]) (integrated)."""
    circuit_id = 'ts_05'
    algorithm = 'twosum10_id'
    _cond = conditions.domain([0, 5])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class TsInjectCircuit(Twosum10, ContrastCircuit):
    """ts_inject on twosum10_id: contrast vs ts_whole [digit-toggle] (integrated)."""
    circuit_id = 'ts_inject'
    algorithm = 'twosum10_id'
    base_circuit = 'ts_whole'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        def ts(r):
            seen, y = set(), []
            for v in r:
                y.append(1 if ((10 - v) % 10) in seen else 0); seen.add(v)
            return y
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            p = torch.randint(1, L, (idx.numel(),), generator=g)
            new = torch.randint(0, 10, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                if int(new[k]) == int(clean[i, p[k]]): continue
                c = clean[i].clone(); c[p[k]] = new[k]
                if ts(clean[i].tolist()) == ts(c.tolist()): continue
                corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class HoHoldge5equivCircuit(Holdodd, DomainCircuit):
    """ho_holdge5equiv on holdodd_id: isolated via domain([0, 2, 4, 5, 7, 9]) (integrated)."""
    circuit_id = 'ho_holdge5equiv'
    algorithm = 'holdodd_id'
    _cond = conditions.domain([0, 2, 4, 5, 7, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class HoAlloddCircuit(Holdodd, DomainCircuit):
    """ho_allodd on holdodd_id: isolated via ALL_ODD (integrated)."""
    circuit_id = 'ho_allodd'
    algorithm = 'holdodd_id'
    _cond = conditions.ALL_ODD
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class HoPlantCircuit(Holdodd, ContrastCircuit):
    """ho_plant on holdodd_id: contrast vs ho_allodd [digit-toggle] (integrated)."""
    circuit_id = 'ho_plant'
    algorithm = 'holdodd_id'
    base_circuit = 'ho_allodd'
    local_predicate = False
    _cond = conditions.ALL_ODD
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            p = torch.randint(0, self.input_length, (idx.numel(),), generator=g)
            new = 2 * torch.randint(0, 5, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                c = clean[i].clone(); c[p[k]] = new[k]
                if torch.equal(self.eval(clean[i]), self.eval(c)): continue
                corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class AloAllevenCircuit(Addlastodd, DomainCircuit):
    """alo_alleven on addlastodd_id: isolated via ALL_EVEN (integrated)."""
    circuit_id = 'alo_alleven'
    algorithm = 'addlastodd_id'
    _cond = conditions.ALL_EVEN
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class AloAlloddCircuit(Addlastodd, DomainCircuit):
    """alo_allodd on addlastodd_id: isolated via ALL_ODD (integrated)."""
    circuit_id = 'alo_allodd'
    algorithm = 'addlastodd_id'
    _cond = conditions.ALL_ODD
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class AloOddswapCircuit(Addlastodd, ContrastCircuit):
    """alo_oddswap on addlastodd_id: contrast vs alo_allodd [value-perturb-control] (integrated)."""
    circuit_id = 'alo_oddswap'
    algorithm = 'addlastodd_id'
    base_circuit = 'alo_allodd'
    local_predicate = True
    _cond = conditions.ALL_ODD
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + 2) % 10
        return clean, corrupt


class BmAlldistinctCircuit(Bmcand, DomainCircuit):
    """bm_alldistinct on bmcand_id: isolated via Condition('all_distinct', lambda x: len(set(x.tolist())) == 10, lambda batch, seed: torch.argsort(torch.rand(batch, 10, generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'bm_alldistinct'))), dim=1)) (integrated)."""
    circuit_id = 'bm_alldistinct'
    algorithm = 'bmcand_id'
    _cond = conditions.Condition('all_distinct', lambda x: len(set(x.tolist())) == 10, lambda batch, seed: torch.argsort(torch.rand(batch, 10, generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'bm_alldistinct'))), dim=1))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class BmMatchbreakCircuit(Bmcand, ContrastCircuit):
    """bm_matchbreak on bmcand_id: contrast vs bm_alldistinct [digit-toggle] (integrated)."""
    circuit_id = 'bm_matchbreak'
    algorithm = 'bmcand_id'
    base_circuit = 'bm_alldistinct'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            p = torch.randint(1, L, (idx.numel(),), generator=g)
            new = torch.randint(0, 10, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                if int(new[k]) == int(clean[i, p[k]]): continue
                c = clean[i].clone(); c[p[k]] = new[k]
                if torch.equal(self.eval(clean[i]), self.eval(c)): continue
                corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class H2Ge2qualCircuit(Hold2ge5, DomainCircuit):
    """h2_ge2qual on hold2ge5_id: isolated via Condition('ge2_high', lambda x: int((x >= 5).sum()) >= 2, lambda batch, seed: (lambda blk: blk[(blk >= 5).sum(1) >= 2][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'h2_ge2qual'))))) (integrated)."""
    circuit_id = 'h2_ge2qual'
    algorithm = 'hold2ge5_id'
    _cond = conditions.Condition('ge2_high', lambda x: int((x >= 5).sum()) >= 2, lambda batch, seed: (lambda blk: blk[(blk >= 5).sum(1) >= 2][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'h2_ge2qual')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class H2RequalCircuit(Hold2ge5, ContrastCircuit):
    """h2_requal on hold2ge5_id: contrast vs h2_ge2qual [value-perturb-control] (integrated)."""
    circuit_id = 'h2_requal'
    algorithm = 'hold2ge5_id'
    base_circuit = 'h2_ge2qual'
    local_predicate = False
    _cond = conditions.Condition('ge2_high', lambda x: int((x >= 5).sum()) >= 2, lambda batch, seed: (lambda blk: blk[(blk >= 5).sum(1) >= 2][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'h2_requal')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                highs = (clean[i] >= 5).nonzero().flatten()
                q = int(highs[int(torch.randint(0, highs.numel(), (1,), generator=g))])
                corrupt[i] = clean[i]
                corrupt[i, q] = 5 + ((int(clean[i, q]) - 5 + 1 + int(torch.randint(0, 4, (1,), generator=g))) % 5)
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Sjf2JamCircuit(Sjf2, DomainCircuit):
    """sjf2_jam on sjf2_id: isolated via Condition('sjf2_jam', lambda x: int(x[0]) == 9 and int(x[1]) == 9, conditions.position_domain_sampler({0: [9], 1: [9]})) (integrated)."""
    circuit_id = 'sjf2_jam'
    algorithm = 'sjf2_id'
    _cond = conditions.Condition('sjf2_jam', lambda x: int(x[0]) == 9 and int(x[1]) == 9, conditions.position_domain_sampler({0: [9], 1: [9]}))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Sjf2Raisem2Circuit(Sjf2, ContrastCircuit):
    """sjf2_raiseM2 on sjf2_id: contrast vs sjf2_jam [value-perturb-control] (integrated)."""
    circuit_id = 'sjf2_raiseM2'
    algorithm = 'sjf2_id'
    base_circuit = 'sjf2_jam'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            p = torch.randint(0, L, (idx.numel(),), generator=g)
            d = torch.randint(1, 10, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                c = clean[i].clone(); c[p[k]] = (int(clean[i, p[k]]) + int(d[k])) % 10
                if torch.equal(self.eval(clean[i]), self.eval(c)): continue
                corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Gapge5Ge2highCircuit(Gapge5, DomainCircuit):
    """gapge5_ge2high on gapge5_id: isolated via Condition('ge2_high', lambda x: int((x >= 5).sum()) >= 2, lambda batch, seed: (lambda blk: blk[(blk >= 5).sum(1) >= 2][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'gapge5_ge2high'))))) (integrated)."""
    circuit_id = 'gapge5_ge2high'
    algorithm = 'gapge5_id'
    _cond = conditions.Condition('ge2_high', lambda x: int((x >= 5).sum()) >= 2, lambda batch, seed: (lambda blk: blk[(blk >= 5).sum(1) >= 2][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'gapge5_ge2high')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Gapge5MovehighCircuit(Gapge5, ContrastCircuit):
    """gapge5_movehigh on gapge5_id: contrast vs gapge5_ge2high [position-swap] (integrated)."""
    circuit_id = 'gapge5_movehigh'
    algorithm = 'gapge5_id'
    base_circuit = 'gapge5_ge2high'
    local_predicate = False
    _cond = conditions.Condition('ge2_high', lambda x: int((x >= 5).sum()) >= 2, lambda batch, seed: (lambda blk: blk[(blk >= 5).sum(1) >= 2][:batch])(torch.randint(0, 10, (4 * batch + 64, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'gapge5_movehigh')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                highs = (clean[i] >= 5).nonzero().flatten().tolist()
                lows = (clean[i] < 5).nonzero().flatten().tolist()
                if not lows:
                    clean[i] = DomainCircuit.sample(self, 1, int(seed) * 919 + i + t + 2)[0]
                    continue
                hp = highs[int(torch.randint(0, len(highs), (1,), generator=g))]
                lp = lows[int(torch.randint(0, len(lows), (1,), generator=g))]
                corrupt[i] = clean[i]
                corrupt[i, hp], corrupt[i, lp] = int(clean[i, lp]), int(clean[i, hp])
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class RiseeraseNoriseCircuit(Riseerase, DomainCircuit):
    """riseerase_norise on riseerase_id: isolated via monotone() (integrated)."""
    circuit_id = 'riseerase_norise'
    algorithm = 'riseerase_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class RiseeraseMakeriseCircuit(Riseerase, ContrastCircuit):
    """riseerase_makerise on riseerase_id: contrast vs riseerase_norise [threshold-straddle] (integrated)."""
    circuit_id = 'riseerase_makerise'
    algorithm = 'riseerase_id'
    base_circuit = 'riseerase_norise'
    local_predicate = True
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = 1 + int(torch.randint(0, self.input_length - 1, (1,), generator=g))
                if int(clean[i, p - 1]) >= 9: continue
                corrupt[i] = clean[i]
                corrupt[i, p] = int(clean[i, p - 1]) + 1
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) == int((clean_row != corrupt_row).nonzero()[0])


class Ebud3LowbudgetCircuit(Ebud3, DomainCircuit):
    """ebud3_lowbudget on ebud3_id: isolated via Condition('le3_high', lambda x: int((x >= 5).sum()) <= 3, lambda batch, seed: (lambda blk: blk[(blk >= 5).sum(1) <= 3][:batch])(torch.randint(0, 10, (16 * batch + 256, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'ebud3_lowbudget'))))) (integrated)."""
    circuit_id = 'ebud3_lowbudget'
    algorithm = 'ebud3_id'
    _cond = conditions.Condition('le3_high', lambda x: int((x >= 5).sum()) <= 3, lambda batch, seed: (lambda blk: blk[(blk >= 5).sum(1) <= 3][:batch])(torch.randint(0, 10, (16 * batch + 256, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'ebud3_lowbudget')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Ebud3SpendCircuit(Ebud3, ContrastCircuit):
    """ebud3_spend on ebud3_id: contrast vs ebud3_lowbudget [threshold-straddle] (integrated)."""
    circuit_id = 'ebud3_spend'
    algorithm = 'ebud3_id'
    base_circuit = 'ebud3_lowbudget'
    local_predicate = True
    _cond = conditions.Condition('le3_high', lambda x: int((x >= 5).sum()) <= 3, lambda batch, seed: (lambda blk: blk[(blk >= 5).sum(1) <= 3][:batch])(torch.randint(0, 10, (16 * batch + 256, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'ebud3_spend')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 5, (batch, L), generator=g)
        perm = torch.argsort(torch.rand(batch, L, generator=g), dim=1)
        rows = torch.arange(batch).unsqueeze(1)
        nhigh = torch.randint(0, 3, (batch,), generator=g)
        for b in range(batch):
            clean[b, perm[b, :int(nhigh[b])]] = 5 + torch.randint(0, 5, (int(nhigh[b]),), generator=g)
        corrupt = clean.clone()
        for b in range(batch):
            low = (clean[b] >= 1) & (clean[b] < 5)
            cand = low.nonzero().flatten()
            if not cand.numel():
                q = int((clean[b] < 5).nonzero().flatten()[0])
                clean[b, q] = 1 + int(torch.randint(0, 4, (1,), generator=g))
            else:
                q = int(cand[int(torch.randint(0, cand.numel(), (1,), generator=g))])
            corrupt[b] = clean[b]
            corrupt[b, q] = int(clean[b, q]) + 5
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) == int((clean_row != corrupt_row).nonzero()[0])


class TieswapNotieCircuit(Tieswap, DomainCircuit):
    """tieswap_notie on tieswap_id: isolated via Condition('no_extreme_tie', lambda x: all(int(x[i]) != int(x[:i].max()) and int(x[i]) != int(x[:i].min()) for i in range(1, 10)), lambda batch, seed: (lambda blk: blk[[all(int(r[i]) != int(r[:i].max()) and int(r[i]) != int(r[:i].min()) for i in range(1, 10)) for r in blk]][:batch])(torch.randint(0, 10, (16 * batch + 256, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'tieswap_notie'))))) (integrated)."""
    circuit_id = 'tieswap_notie'
    algorithm = 'tieswap_id'
    _cond = conditions.Condition('no_extreme_tie', lambda x: all(int(x[i]) != int(x[:i].max()) and int(x[i]) != int(x[:i].min()) for i in range(1, 10)), lambda batch, seed: (lambda blk: blk[[all(int(r[i]) != int(r[:i].max()) and int(r[i]) != int(r[:i].min()) for i in range(1, 10)) for r in blk]][:batch])(torch.randint(0, 10, (16 * batch + 256, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'tieswap_notie')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class TieswapPlantCircuit(Tieswap, ContrastCircuit):
    """tieswap_plant on tieswap_id: contrast vs tieswap_notie [threshold-straddle] (integrated)."""
    circuit_id = 'tieswap_plant'
    algorithm = 'tieswap_id'
    base_circuit = 'tieswap_notie'
    local_predicate = True
    _cond = conditions.Condition('no_extreme_tie', lambda x: all(int(x[i]) != int(x[:i].max()) and int(x[i]) != int(x[:i].min()) for i in range(1, 10)), lambda batch, seed: (lambda blk: blk[[all(int(r[i]) != int(r[:i].max()) and int(r[i]) != int(r[:i].min()) for i in range(1, 10)) for r in blk]][:batch])(torch.randint(0, 10, (16 * batch + 256, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'tieswap_plant')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        for i in range(batch):
            for k in range(64):
                opts = [p for p in range(1, 10)
                        if int(clean[i, :p].min()) < int(clean[i, p]) < int(clean[i, :p].max())]
                if opts: break
                clean[i] = DomainCircuit.sample(self, 1, int(seed) * 617 + i * 64 + k + 1)[0]
            assert opts
            p = opts[int(torch.randint(0, len(opts), (1,), generator=g))]
            corrupt[i] = clean[i]
            corrupt[i, p] = int(clean[i, :p].max())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) == int((clean_row != corrupt_row).nonzero()[0])


class RisemaxNoriseCircuit(Risemax, DomainCircuit):
    """risemax_norise on risemax_id: isolated via monotone() (integrated)."""
    circuit_id = 'risemax_norise'
    algorithm = 'risemax_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class RisemaxMakeriseCircuit(Risemax, ContrastCircuit):
    """risemax_makerise on risemax_id: contrast vs risemax_norise [threshold-straddle] (integrated)."""
    circuit_id = 'risemax_makerise'
    algorithm = 'risemax_id'
    base_circuit = 'risemax_norise'
    local_predicate = True
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = 1 + int(torch.randint(0, self.input_length - 1, (1,), generator=g))
                if int(clean[i, p - 1]) >= 9: continue
                corrupt[i] = clean[i]
                corrupt[i, p] = int(clean[i, p - 1]) + 1
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Delc2AlllowCircuit(Delc2, DomainCircuit):
    """delc2_alllow on delc2_id: isolated via domain([0, 1, 2, 3, 4]) (integrated)."""
    circuit_id = 'delc2_alllow'
    algorithm = 'delc2_id'
    _cond = conditions.domain([0, 1, 2, 3, 4])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Delc2TripflipCircuit(Delc2, ContrastCircuit):
    """delc2_tripflip on delc2_id: contrast vs delc2_alllow [threshold-straddle] (integrated)."""
    circuit_id = 'delc2_tripflip'
    algorithm = 'delc2_id'
    base_circuit = 'delc2_alllow'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            p = torch.randint(0, L - 2, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                pp = int(p[k])
                if int(clean[i, pp]) >= 5: continue
                if pp >= 2 and int(clean[i, pp - 2]) >= 5: continue
                if int(clean[i, pp + 2]) == 0: continue
                corrupt[i] = clean[i]
                corrupt[i, pp] = 5 + int(torch.randint(0, 5, (1,), generator=g))
                need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool)
        m[p] = True
        if p + 2 < 10:
            m[p + 2] = True
        return m


class RiskeepWholeCircuit(Riskeep, DomainCircuit):
    """riskeep_whole on riskeep_id: isolated via FULL (integrated)."""
    circuit_id = 'riskeep_whole'
    algorithm = 'riskeep_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class RiskeepBinary09Circuit(Riskeep, DomainCircuit):
    """riskeep_binary09 on riskeep_id: isolated via domain([0, 9]) (integrated)."""
    circuit_id = 'riskeep_binary09'
    algorithm = 'riskeep_id'
    _cond = conditions.domain([0, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class RiskeepMakeriserCircuit(Riskeep, ContrastCircuit):
    """riskeep_makeriser on riskeep_id: contrast vs riskeep_whole [threshold-straddle] (integrated)."""
    circuit_id = 'riskeep_makeriser'
    algorithm = 'riskeep_id'
    base_circuit = 'riskeep_whole'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            p = torch.randint(1, L, (idx.numel(),), generator=g)
            for k, i in enumerate(idx.tolist()):
                pp = int(p[k]); lo = int(clean[i, pp - 1])
                if lo >= 9 or int(clean[i, pp]) > lo: continue
                corrupt[i] = clean[i]
                corrupt[i, pp] = lo + 1 + int(torch.randint(0, 9 - lo, (1,), generator=g))
                if not torch.equal(self.eval(clean[i]), self.eval(corrupt[i])):
                    need[i] = False
        assert not int(need.any())
        return clean, corrupt


class MixtieNotieCircuit(Mixtie, DomainCircuit):
    """mixtie_notie on mixtie_id: isolated via Condition('no_extreme_tie', lambda x: all(int(x[i]) != int(x[:i].max()) and int(x[i]) != int(x[:i].min()) for i in range(1, 10)), lambda batch, seed: (lambda blk: blk[[all(int(r[i]) != int(r[:i].max()) and int(r[i]) != int(r[:i].min()) for i in range(1, 10)) for r in blk]][:batch])(torch.randint(0, 10, (16 * batch + 256, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'mixtie_notie'))))) (integrated)."""
    circuit_id = 'mixtie_notie'
    algorithm = 'mixtie_id'
    _cond = conditions.Condition('no_extreme_tie', lambda x: all(int(x[i]) != int(x[:i].max()) and int(x[i]) != int(x[:i].min()) for i in range(1, 10)), lambda batch, seed: (lambda blk: blk[[all(int(r[i]) != int(r[:i].max()) and int(r[i]) != int(r[:i].min()) for i in range(1, 10)) for r in blk]][:batch])(torch.randint(0, 10, (16 * batch + 256, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'mixtie_notie')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class MixtiePlantCircuit(Mixtie, ContrastCircuit):
    """mixtie_plant on mixtie_id: contrast vs mixtie_notie [threshold-straddle] (integrated)."""
    circuit_id = 'mixtie_plant'
    algorithm = 'mixtie_id'
    base_circuit = 'mixtie_notie'
    local_predicate = True
    _cond = conditions.Condition('no_extreme_tie', lambda x: all(int(x[i]) != int(x[:i].max()) and int(x[i]) != int(x[:i].min()) for i in range(1, 10)), lambda batch, seed: (lambda blk: blk[[all(int(r[i]) != int(r[:i].max()) and int(r[i]) != int(r[:i].min()) for i in range(1, 10)) for r in blk]][:batch])(torch.randint(0, 10, (16 * batch + 256, 10), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'mixtie_plant')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        for i in range(batch):
            for k in range(64):
                opts = [p for p in range(1, 10)
                        if int(clean[i, :p].min()) < int(clean[i, p]) < int(clean[i, :p].max())]
                if opts: break
                clean[i] = DomainCircuit.sample(self, 1, int(seed) * 641 + i * 64 + k + 1)[0]
            assert opts
            p = opts[int(torch.randint(0, len(opts), (1,), generator=g))]
            corrupt[i] = clean[i]
            corrupt[i, p] = int(clean[i, :p].max())
        return clean, corrupt


class Pmaxv89NovetoCircuit(Pmaxv89, DomainCircuit):
    """pmaxv89_noveto on pmaxv89_id: isolated via domain([0, 1, 2, 3, 4, 5, 6, 7]) (integrated)."""
    circuit_id = 'pmaxv89_noveto'
    algorithm = 'pmaxv89_id'
    _cond = conditions.domain([0, 1, 2, 3, 4, 5, 6, 7])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Pmaxv89DisqualifyCircuit(Pmaxv89, ContrastCircuit):
    """pmaxv89_disqualify on pmaxv89_id: contrast vs pmaxv89_noveto [threshold-straddle] (integrated)."""
    circuit_id = 'pmaxv89_disqualify'
    algorithm = 'pmaxv89_id'
    base_circuit = 'pmaxv89_noveto'
    local_predicate = False
    _cond = conditions.domain([0, 1, 2, 3, 4, 5, 6, 7])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 8, (batch, L), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, L, (batch,), generator=g)
        for _ in range(64):
            before = torch.arange(L).unsqueeze(0) < p.unsqueeze(1)
            clean = torch.where(before, clean.clamp(max=6), clean)
            clean[rows, p] = 7
            good = (torch.cummax(clean, 1).values[rows, p] == clean[rows, p])
            if bool(good.all()): break
            clean[~good] = torch.randint(0, 8, (int((~good).sum()), L), generator=g)
            p[~good] = torch.randint(0, L, (int((~good).sum()),), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = 8
        return clean, corrupt


class Pmaxv789NovetoCircuit(Pmaxv789, DomainCircuit):
    """pmaxv789_noveto on pmaxv789_id: isolated via domain([0, 1, 2, 3, 4, 5, 6]) (integrated)."""
    circuit_id = 'pmaxv789_noveto'
    algorithm = 'pmaxv789_id'
    _cond = conditions.domain([0, 1, 2, 3, 4, 5, 6])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Pmaxv789DisqualifyCircuit(Pmaxv789, ContrastCircuit):
    """pmaxv789_disqualify on pmaxv789_id: contrast vs pmaxv789_noveto [threshold-straddle] (integrated)."""
    circuit_id = 'pmaxv789_disqualify'
    algorithm = 'pmaxv789_id'
    base_circuit = 'pmaxv789_noveto'
    local_predicate = False
    _cond = conditions.domain([0, 1, 2, 3, 4, 5, 6])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 7, (batch, L), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, L, (batch,), generator=g)
        for _ in range(64):
            before = torch.arange(L).unsqueeze(0) < p.unsqueeze(1)
            clean = torch.where(before, clean.clamp(max=5), clean)
            clean[rows, p] = 6
            good = (torch.cummax(clean, 1).values[rows, p] == clean[rows, p])
            if bool(good.all()): break
            clean[~good] = torch.randint(0, 7, (int((~good).sum()), L), generator=g)
            p[~good] = torch.randint(0, L, (int((~good).sum()),), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = 7
        return clean, corrupt


class TgNozeroCircuit(Togsum0, DomainCircuit):
    """tg_nozero on togsum0_id: isolated via exclude_digit(0) (integrated)."""
    circuit_id = 'tg_nozero'
    algorithm = 'togsum0_id'
    _cond = conditions.exclude_digit(0)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class TgEveneditCircuit(Togsum0, ContrastCircuit):
    """tg_evenedit on togsum0_id: contrast vs tg_nozero [value-perturb-control] (integrated)."""
    circuit_id = 'tg_evenedit'
    algorithm = 'togsum0_id'
    base_circuit = 'tg_nozero'
    local_predicate = False
    _cond = conditions.exclude_digit(0)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                evens = ((clean[i] % 2 == 0)).nonzero().flatten()
                if not evens.numel():
                    clean[i] = DomainCircuit.sample(self, 1, int(seed) * 971 + i + t + 2)[0]
                    continue
                p = int(evens[int(torch.randint(0, evens.numel(), (1,), generator=g))])
                corrupt[i] = clean[i]
                corrupt[i, p] = (int(clean[i, p]) + 2) % 10
                if int(corrupt[i, p]) == 0: corrupt[i, p] = (int(clean[i, p]) + 4) % 10
                if int(corrupt[i, p]) == 0 or torch.equal(self.eval(clean[i]), self.eval(corrupt[i])): continue
                need[i] = False
        assert not int(need.any())
        return clean, corrupt


class SsNozeroCircuit(Signsum0, DomainCircuit):
    """ss_nozero on signsum0_id: isolated via exclude_digit(0) (integrated)."""
    circuit_id = 'ss_nozero'
    algorithm = 'signsum0_id'
    _cond = conditions.exclude_digit(0)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class SsValeditCircuit(Signsum0, ContrastCircuit):
    """ss_valedit on signsum0_id: contrast vs ss_nozero [value-perturb-control] (integrated)."""
    circuit_id = 'ss_valedit'
    algorithm = 'signsum0_id'
    base_circuit = 'ss_nozero'
    local_predicate = False
    _cond = conditions.exclude_digit(0)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] % 9) + 1
        for _ in range(32):
            bad = corrupt[rows, p] == clean[rows, p]
            if not int(bad.any()): break
            corrupt[rows[bad], p[bad]] = (corrupt[rows[bad], p[bad]] % 9) + 1
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) >= int((clean_row != corrupt_row).nonzero()[0])


class Dsh2FullCircuit(DiffShift, DomainCircuit):
    """dsh2_full on diffsh1_diffsh1_id: isolated via FULL (integrated)."""
    circuit_id = 'dsh2_full'
    algorithm = 'diffsh1_diffsh1_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Dsh2WindowhitCircuit(DiffShift, ContrastCircuit):
    """dsh2_windowhit on diffsh1_diffsh1_id: contrast vs dsh2_full [digit-toggle] (integrated)."""
    circuit_id = 'dsh2_windowhit'
    algorithm = 'diffsh1_diffsh1_id'
    base_circuit = 'dsh2_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.tensor([1, 3, 7, 9])[torch.randint(0, 4, (batch,), generator=g)]
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        hi = min(p + 2, 9)
        m = torch.zeros(10, dtype=torch.bool)
        m[p:hi + 1] = True
        return m


class Dsh3FullCircuit(DiffShift, DomainCircuit):
    """dsh3_full on diffsh1_diffsh1_diffsh1_id: isolated via FULL (integrated)."""
    circuit_id = 'dsh3_full'
    algorithm = 'diffsh1_diffsh1_diffsh1_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Dsh3WindowhitCircuit(DiffShift, ContrastCircuit):
    """dsh3_windowhit on diffsh1_diffsh1_diffsh1_id: contrast vs dsh3_full [digit-toggle] (integrated)."""
    circuit_id = 'dsh3_windowhit'
    algorithm = 'diffsh1_diffsh1_diffsh1_id'
    base_circuit = 'dsh3_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.tensor([1, 3, 7, 9])[torch.randint(0, 4, (batch,), generator=g)]
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        hi = min(p + 3, 9)
        m = torch.zeros(10, dtype=torch.bool)
        m[p:hi + 1] = True
        return m


class Med3FullCircuit(Median3, DomainCircuit):
    """med3_full on med3_id: isolated via FULL (integrated)."""
    circuit_id = 'med3_full'
    algorithm = 'med3_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Med3BinaryCircuit(Median3, DomainCircuit):
    """med3_binary on med3_id: isolated via BINARY (integrated)."""
    circuit_id = 'med3_binary'
    algorithm = 'med3_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Med3DistinctCircuit(Median3, DomainCircuit):
    """med3_distinct on med3_id: isolated via Condition('all_distinct', lambda x: len(set(x.tolist())) == 10, lambda batch, seed: torch.argsort(torch.rand(batch, 10, generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'med3_distinct'))), dim=1)) (integrated)."""
    circuit_id = 'med3_distinct'
    algorithm = 'med3_id'
    _cond = conditions.Condition('all_distinct', lambda x: len(set(x.tolist())) == 10, lambda batch, seed: torch.argsort(torch.rand(batch, 10, generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'med3_distinct'))), dim=1))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Med3SpikeCircuit(Median3, ContrastCircuit):
    """med3_spike on med3_id: contrast vs med3_full [digit-toggle] (integrated)."""
    circuit_id = 'med3_spike'
    algorithm = 'med3_id'
    base_circuit = 'med3_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = 1 + int(torch.randint(0, L - 2, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Maj3FullCircuit(Majority3, DomainCircuit):
    """maj3_full on maj3_id: isolated via FULL (integrated)."""
    circuit_id = 'maj3_full'
    algorithm = 'maj3_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Maj3DistinctCircuit(Majority3, DomainCircuit):
    """maj3_distinct on maj3_id: isolated via Condition('all_distinct', lambda x: len(set(x.tolist())) == 10, lambda batch, seed: torch.argsort(torch.rand(batch, 10, generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'maj3_distinct'))), dim=1)) (integrated)."""
    circuit_id = 'maj3_distinct'
    algorithm = 'maj3_id'
    _cond = conditions.Condition('all_distinct', lambda x: len(set(x.tolist())) == 10, lambda batch, seed: torch.argsort(torch.rand(batch, 10, generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'maj3_distinct'))), dim=1))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Maj3FlipCircuit(Majority3, ContrastCircuit):
    """maj3_flip on maj3_id: contrast vs maj3_full [digit-toggle] (integrated)."""
    circuit_id = 'maj3_flip'
    algorithm = 'maj3_id'
    base_circuit = 'maj3_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Spile4LowCircuit(Spile4, DomainCircuit):
    """spile4_low on spile4_id: isolated via domain([0, 1, 2, 3]) (integrated)."""
    circuit_id = 'spile4_low'
    algorithm = 'spile4_id'
    _cond = conditions.domain([0, 1, 2, 3])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Spile4HighCircuit(Spile4, DomainCircuit):
    """spile4_high on spile4_id: isolated via domain([4, 5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'spile4_high'
    algorithm = 'spile4_id'
    _cond = conditions.domain([4, 5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Spile4FullCircuit(Spile4, DomainCircuit):
    """spile4_full on spile4_id: isolated via FULL (integrated)."""
    circuit_id = 'spile4_full'
    algorithm = 'spile4_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Spile4CrossCircuit(Spile4, ContrastCircuit):
    """spile4_cross on spile4_id: contrast vs spile4_full [digit-toggle] (integrated)."""
    circuit_id = 'spile4_cross'
    algorithm = 'spile4_id'
    base_circuit = 'spile4_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                hi = int(clean[i, p]) >= 4
                c = clean[i].clone(); c[p] = int(torch.randint(0, 4, (1,), generator=g)) if hi else 4 + int(torch.randint(0, 6, (1,), generator=g))
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Spile5LowCircuit(Spile5, DomainCircuit):
    """spile5_low on spile5_id: isolated via domain([0, 1, 2, 3, 4]) (integrated)."""
    circuit_id = 'spile5_low'
    algorithm = 'spile5_id'
    _cond = conditions.domain([0, 1, 2, 3, 4])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Spile5HighCircuit(Spile5, DomainCircuit):
    """spile5_high on spile5_id: isolated via domain([5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'spile5_high'
    algorithm = 'spile5_id'
    _cond = conditions.domain([5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Spile5FullCircuit(Spile5, DomainCircuit):
    """spile5_full on spile5_id: isolated via FULL (integrated)."""
    circuit_id = 'spile5_full'
    algorithm = 'spile5_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Spile5CrossCircuit(Spile5, ContrastCircuit):
    """spile5_cross on spile5_id: contrast vs spile5_full [digit-toggle] (integrated)."""
    circuit_id = 'spile5_cross'
    algorithm = 'spile5_id'
    base_circuit = 'spile5_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                hi = int(clean[i, p]) >= 5
                c = clean[i].clone(); c[p] = int(torch.randint(0, 5, (1,), generator=g)) if hi else 5 + int(torch.randint(0, 5, (1,), generator=g))
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Spile6LowCircuit(Spile6, DomainCircuit):
    """spile6_low on spile6_id: isolated via domain([0, 1, 2, 3, 4, 5]) (integrated)."""
    circuit_id = 'spile6_low'
    algorithm = 'spile6_id'
    _cond = conditions.domain([0, 1, 2, 3, 4, 5])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Spile6HighCircuit(Spile6, DomainCircuit):
    """spile6_high on spile6_id: isolated via domain([6, 7, 8, 9]) (integrated)."""
    circuit_id = 'spile6_high'
    algorithm = 'spile6_id'
    _cond = conditions.domain([6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Spile6FullCircuit(Spile6, DomainCircuit):
    """spile6_full on spile6_id: isolated via FULL (integrated)."""
    circuit_id = 'spile6_full'
    algorithm = 'spile6_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Spile6CrossCircuit(Spile6, ContrastCircuit):
    """spile6_cross on spile6_id: contrast vs spile6_full [digit-toggle] (integrated)."""
    circuit_id = 'spile6_cross'
    algorithm = 'spile6_id'
    base_circuit = 'spile6_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                hi = int(clean[i, p]) >= 6
                c = clean[i].clone(); c[p] = int(torch.randint(0, 6, (1,), generator=g)) if hi else 6 + int(torch.randint(0, 4, (1,), generator=g))
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class CycaEvensCircuit(Cyca, DomainCircuit):
    """cyca_evens on cyca_id: isolated via ALL_EVEN (integrated)."""
    circuit_id = 'cyca_evens'
    algorithm = 'cyca_id'
    _cond = conditions.ALL_EVEN
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CycaOddsCircuit(Cyca, DomainCircuit):
    """cyca_odds on cyca_id: isolated via ALL_ODD (integrated)."""
    circuit_id = 'cyca_odds'
    algorithm = 'cyca_id'
    _cond = conditions.ALL_ODD
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CycaBinaryCircuit(Cyca, DomainCircuit):
    """cyca_binary on cyca_id: isolated via BINARY (integrated)."""
    circuit_id = 'cyca_binary'
    algorithm = 'cyca_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CycaEditCircuit(Cyca, ContrastCircuit):
    """cyca_edit on cyca_id: contrast vs cyca_binary [digit-toggle] (integrated)."""
    circuit_id = 'cyca_edit'
    algorithm = 'cyca_id'
    base_circuit = 'cyca_binary'
    local_predicate = True
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = 1 - int(clean[i, p])
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class R184t4AllhighCircuit(Rule184Thr4, DomainCircuit):
    """r184t4_allhigh on r184t4_id: isolated via domain([4, 5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'r184t4_allhigh'
    algorithm = 'r184t4_id'
    _cond = conditions.domain([4, 5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class R184t4AlllowCircuit(Rule184Thr4, DomainCircuit):
    """r184t4_alllow on r184t4_id: isolated via domain([0, 1, 2, 3]) (integrated)."""
    circuit_id = 'r184t4_alllow'
    algorithm = 'r184t4_id'
    _cond = conditions.domain([0, 1, 2, 3])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class R184t4FullCircuit(Rule184Thr4, DomainCircuit):
    """r184t4_full on r184t4_id: isolated via FULL (integrated)."""
    circuit_id = 'r184t4_full'
    algorithm = 'r184t4_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class R184t4CrossCircuit(Rule184Thr4, ContrastCircuit):
    """r184t4_cross on r184t4_id: contrast vs r184t4_full [digit-toggle] (integrated)."""
    circuit_id = 'r184t4_cross'
    algorithm = 'r184t4_id'
    base_circuit = 'r184t4_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                hi = int(clean[i, p]) >= 4
                c = clean[i].clone(); c[p] = int(torch.randint(0, 4, (1,), generator=g)) if hi else 4 + int(torch.randint(0, 6, (1,), generator=g))
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class R184t5AllhighCircuit(Rule184Thr5, DomainCircuit):
    """r184t5_allhigh on r184t5_id: isolated via domain([5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'r184t5_allhigh'
    algorithm = 'r184t5_id'
    _cond = conditions.domain([5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class R184t5AlllowCircuit(Rule184Thr5, DomainCircuit):
    """r184t5_alllow on r184t5_id: isolated via domain([0, 1, 2, 3, 4]) (integrated)."""
    circuit_id = 'r184t5_alllow'
    algorithm = 'r184t5_id'
    _cond = conditions.domain([0, 1, 2, 3, 4])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class R184t5FullCircuit(Rule184Thr5, DomainCircuit):
    """r184t5_full on r184t5_id: isolated via FULL (integrated)."""
    circuit_id = 'r184t5_full'
    algorithm = 'r184t5_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class R184t5CrossCircuit(Rule184Thr5, ContrastCircuit):
    """r184t5_cross on r184t5_id: contrast vs r184t5_full [digit-toggle] (integrated)."""
    circuit_id = 'r184t5_cross'
    algorithm = 'r184t5_id'
    base_circuit = 'r184t5_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                hi = int(clean[i, p]) >= 5
                c = clean[i].clone(); c[p] = int(torch.randint(0, 5, (1,), generator=g)) if hi else 5 + int(torch.randint(0, 5, (1,), generator=g))
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class R184t6AllhighCircuit(Rule184Thr6, DomainCircuit):
    """r184t6_allhigh on r184t6_id: isolated via domain([6, 7, 8, 9]) (integrated)."""
    circuit_id = 'r184t6_allhigh'
    algorithm = 'r184t6_id'
    _cond = conditions.domain([6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class R184t6AlllowCircuit(Rule184Thr6, DomainCircuit):
    """r184t6_alllow on r184t6_id: isolated via domain([0, 1, 2, 3, 4, 5]) (integrated)."""
    circuit_id = 'r184t6_alllow'
    algorithm = 'r184t6_id'
    _cond = conditions.domain([0, 1, 2, 3, 4, 5])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class R184t6FullCircuit(Rule184Thr6, DomainCircuit):
    """r184t6_full on r184t6_id: isolated via FULL (integrated)."""
    circuit_id = 'r184t6_full'
    algorithm = 'r184t6_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class R184t6CrossCircuit(Rule184Thr6, ContrastCircuit):
    """r184t6_cross on r184t6_id: contrast vs r184t6_full [digit-toggle] (integrated)."""
    circuit_id = 'r184t6_cross'
    algorithm = 'r184t6_id'
    base_circuit = 'r184t6_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                hi = int(clean[i, p]) >= 6
                c = clean[i].clone(); c[p] = int(torch.randint(0, 6, (1,), generator=g)) if hi else 6 + int(torch.randint(0, 4, (1,), generator=g))
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class LeakyBin01Circuit(Ewma2, DomainCircuit):
    """leaky_bin01 on ewma2_id: isolated via BINARY (integrated)."""
    circuit_id = 'leaky_bin01'
    algorithm = 'ewma2_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class LeakyWholeCircuit(Ewma2, DomainCircuit):
    """leaky_whole on ewma2_id: isolated via FULL (integrated)."""
    circuit_id = 'leaky_whole'
    algorithm = 'ewma2_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class LeakyKickCircuit(Ewma2, ContrastCircuit):
    """leaky_kick on ewma2_id: contrast vs leaky_whole [digit-toggle] (integrated)."""
    circuit_id = 'leaky_kick'
    algorithm = 'ewma2_id'
    base_circuit = 'leaky_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = (int(clean[i, p]) + int(torch.randint(1, 10, (1,), generator=g))) % 10
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class ZcWholeCircuit(Zcross5, DomainCircuit):
    """zc_whole on zcross5_id: isolated via FULL (integrated)."""
    circuit_id = 'zc_whole'
    algorithm = 'zcross5_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class ZcLowsideCircuit(Zcross5, DomainCircuit):
    """zc_lowside on zcross5_id: isolated via domain([0, 1, 2, 3, 4]) (integrated)."""
    circuit_id = 'zc_lowside'
    algorithm = 'zcross5_id'
    _cond = conditions.domain([0, 1, 2, 3, 4])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class ZcFlipCircuit(Zcross5, ContrastCircuit):
    """zc_flip on zcross5_id: contrast vs zc_whole [digit-toggle] (integrated)."""
    circuit_id = 'zc_flip'
    algorithm = 'zcross5_id'
    base_circuit = 'zc_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                hi = int(clean[i, p]) >= 5
                c = clean[i].clone(); c[p] = int(torch.randint(0, 5, (1,), generator=g)) if hi else 5 + int(torch.randint(0, 5, (1,), generator=g))
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class HeapmaxDescCircuit(Heapmax, DomainCircuit):
    """heapmax_desc on heapmax_id: isolated via monotone() (integrated)."""
    circuit_id = 'heapmax_desc'
    algorithm = 'heapmax_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class HeapmaxAscCircuit(Heapmax, DomainCircuit):
    """heapmax_asc on heapmax_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'heapmax_asc'
    algorithm = 'heapmax_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class HeapmaxFullCircuit(Heapmax, DomainCircuit):
    """heapmax_full on heapmax_id: isolated via FULL (integrated)."""
    circuit_id = 'heapmax_full'
    algorithm = 'heapmax_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class HeapmaxEditCircuit(Heapmax, ContrastCircuit):
    """heapmax_edit on heapmax_id: contrast vs heapmax_full [digit-toggle] (integrated)."""
    circuit_id = 'heapmax_edit'
    algorithm = 'heapmax_id'
    base_circuit = 'heapmax_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Vig2FullCircuit(Vig2, DomainCircuit):
    """vig2_full on vig2_id: isolated via FULL (integrated)."""
    circuit_id = 'vig2_full'
    algorithm = 'vig2_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Vig2SingleCircuit(Vig2, ContrastCircuit):
    """vig2_single on vig2_id: contrast vs vig2_full [digit-toggle] (integrated)."""
    circuit_id = 'vig2_single'
    algorithm = 'vig2_id'
    base_circuit = 'vig2_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = (int(clean[i, p]) + int(torch.randint(1, 10, (1,), generator=g))) % 10
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool); m[p] = True
        return m


class Vig2CongswapCircuit(Vig2, ContrastCircuit):
    """vig2_congswap on vig2_id: contrast vs vig2_full [position-swap] (integrated)."""
    circuit_id = 'vig2_congswap'
    algorithm = 'vig2_id'
    base_circuit = 'vig2_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            for i in idx.tolist():
                opts = [(a, b) for a in range(L) for b in range(a + 2, L, 2) if int(clean[i, a]) != int(clean[i, b])]
                if not opts:
                    clean[i] = torch.randint(0, 10, (L,), generator=g); continue
                a, b = opts[int(torch.randint(0, len(opts), (1,), generator=g))]
                corrupt[i] = clean[i]
                corrupt[i, a], corrupt[i, b] = int(clean[i, b]), int(clean[i, a])
                need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        d = (clean_row != corrupt_row).nonzero().flatten()
        m = torch.zeros(10, dtype=torch.bool); m[d] = True
        return m


class Vig3FullCircuit(Vig3, DomainCircuit):
    """vig3_full on vig3_id: isolated via FULL (integrated)."""
    circuit_id = 'vig3_full'
    algorithm = 'vig3_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Vig3SingleCircuit(Vig3, ContrastCircuit):
    """vig3_single on vig3_id: contrast vs vig3_full [digit-toggle] (integrated)."""
    circuit_id = 'vig3_single'
    algorithm = 'vig3_id'
    base_circuit = 'vig3_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = (int(clean[i, p]) + int(torch.randint(1, 10, (1,), generator=g))) % 10
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool); m[p] = True
        return m


class Vig3CongswapCircuit(Vig3, ContrastCircuit):
    """vig3_congswap on vig3_id: contrast vs vig3_full [position-swap] (integrated)."""
    circuit_id = 'vig3_congswap'
    algorithm = 'vig3_id'
    base_circuit = 'vig3_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            for i in idx.tolist():
                opts = [(a, b) for a in range(L) for b in range(a + 3, L, 3) if int(clean[i, a]) != int(clean[i, b])]
                if not opts:
                    clean[i] = torch.randint(0, 10, (L,), generator=g); continue
                a, b = opts[int(torch.randint(0, len(opts), (1,), generator=g))]
                corrupt[i] = clean[i]
                corrupt[i, a], corrupt[i, b] = int(clean[i, b]), int(clean[i, a])
                need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        d = (clean_row != corrupt_row).nonzero().flatten()
        m = torch.zeros(10, dtype=torch.bool); m[d] = True
        return m


class Vig5FullCircuit(Vig5, DomainCircuit):
    """vig5_full on vig5_id: isolated via FULL (integrated)."""
    circuit_id = 'vig5_full'
    algorithm = 'vig5_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Vig5SingleCircuit(Vig5, ContrastCircuit):
    """vig5_single on vig5_id: contrast vs vig5_full [digit-toggle] (integrated)."""
    circuit_id = 'vig5_single'
    algorithm = 'vig5_id'
    base_circuit = 'vig5_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = (int(clean[i, p]) + int(torch.randint(1, 10, (1,), generator=g))) % 10
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool); m[p] = True
        return m


class Vig5CongswapCircuit(Vig5, ContrastCircuit):
    """vig5_congswap on vig5_id: contrast vs vig5_full [position-swap] (integrated)."""
    circuit_id = 'vig5_congswap'
    algorithm = 'vig5_id'
    base_circuit = 'vig5_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(96):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 12 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            for i in idx.tolist():
                opts = [(a, b) for a in range(L) for b in range(a + 5, L, 5) if int(clean[i, a]) != int(clean[i, b])]
                if not opts:
                    clean[i] = torch.randint(0, 10, (L,), generator=g); continue
                a, b = opts[int(torch.randint(0, len(opts), (1,), generator=g))]
                corrupt[i] = clean[i]
                corrupt[i, a], corrupt[i, b] = int(clean[i, b]), int(clean[i, a])
                need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        d = (clean_row != corrupt_row).nonzero().flatten()
        m = torch.zeros(10, dtype=torch.bool); m[d] = True
        return m


class Hamk3Alph123Circuit(Hamk3, DomainCircuit):
    """hamk3_alph123 on hamk3_id: isolated via domain([1, 2, 3]) (integrated)."""
    circuit_id = 'hamk3_alph123'
    algorithm = 'hamk3_id'
    _cond = conditions.domain([1, 2, 3])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Hamk3FullCircuit(Hamk3, DomainCircuit):
    """hamk3_full on hamk3_id: isolated via FULL (integrated)."""
    circuit_id = 'hamk3_full'
    algorithm = 'hamk3_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Hamk3EditCircuit(Hamk3, ContrastCircuit):
    """hamk3_edit on hamk3_id: contrast vs hamk3_full [digit-toggle] (integrated)."""
    circuit_id = 'hamk3_edit'
    algorithm = 'hamk3_id'
    base_circuit = 'hamk3_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        pat = [1, 2, 3]
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool)
        for j in (p, p + 1, p + 2):
            if j >= 10: continue
            k = p - j + 2
            if (int(clean_row[p]) == pat[k]) != (int(corrupt_row[p]) == pat[k]):
                m[j] = True
        return m


class AltrunWholeCircuit(Altrun, DomainCircuit):
    """altrun_whole on altrun_id: isolated via FULL (integrated)."""
    circuit_id = 'altrun_whole'
    algorithm = 'altrun_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class AltrunBreakCircuit(Altrun, ContrastCircuit):
    """altrun_break on altrun_id: contrast vs altrun_whole [digit-toggle] (integrated)."""
    circuit_id = 'altrun_break'
    algorithm = 'altrun_id'
    base_circuit = 'altrun_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = 1 + int(torch.randint(0, L - 1, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(clean[i, p - 1])
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class BlkparEvenCircuit(BlockSwapParity, DomainCircuit):
    """blkpar_even on blkpar_id: isolated via ALL_EVEN (integrated)."""
    circuit_id = 'blkpar_even'
    algorithm = 'blkpar_id'
    _cond = conditions.ALL_EVEN
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class BlkparOddCircuit(BlockSwapParity, DomainCircuit):
    """blkpar_odd on blkpar_id: isolated via ALL_ODD (integrated)."""
    circuit_id = 'blkpar_odd'
    algorithm = 'blkpar_id'
    _cond = conditions.ALL_ODD
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class BlkparToggleCircuit(BlockSwapParity, ContrastCircuit):
    """blkpar_toggle on blkpar_id: contrast vs blkpar_even [digit-toggle] (integrated)."""
    circuit_id = 'blkpar_toggle'
    algorithm = 'blkpar_id'
    base_circuit = 'blkpar_even'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Blkoe2AscCircuit(BrickSort2, DomainCircuit):
    """blkoe2_asc on blkoe2_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'blkoe2_asc'
    algorithm = 'blkoe2_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Blkoe2DescCircuit(BrickSort2, DomainCircuit):
    """blkoe2_desc on blkoe2_id: isolated via monotone() (integrated)."""
    circuit_id = 'blkoe2_desc'
    algorithm = 'blkoe2_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Blkoe2EditCircuit(BrickSort2, ContrastCircuit):
    """blkoe2_edit on blkoe2_id: contrast vs blkoe2_asc [digit-toggle] (integrated)."""
    circuit_id = 'blkoe2_edit'
    algorithm = 'blkoe2_id'
    base_circuit = 'blkoe2_asc'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class BlkaddCtrl0Circuit(FeistelAdd, DomainCircuit):
    """blkadd_ctrl0 on blkadd_id: isolated via Condition('blkadd_ctrl0', lambda x: all(int(x[k]) == 0 for k in range(0, 10, 2)), conditions.position_domain_sampler({0: [0], 2: [0], 4: [0], 6: [0], 8: [0]})) (integrated)."""
    circuit_id = 'blkadd_ctrl0'
    algorithm = 'blkadd_id'
    _cond = conditions.Condition('blkadd_ctrl0', lambda x: all(int(x[k]) == 0 for k in range(0, 10, 2)), conditions.position_domain_sampler({0: [0], 2: [0], 4: [0], 6: [0], 8: [0]}))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class BlkaddTargetCircuit(FeistelAdd, ContrastCircuit):
    """blkadd_target on blkadd_id: contrast vs blkadd_ctrl0 [digit-toggle] (integrated)."""
    circuit_id = 'blkadd_target'
    algorithm = 'blkadd_id'
    base_circuit = 'blkadd_ctrl0'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = 2 * torch.randint(0, 5, (batch,), generator=g) + 1
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool); m[p] = True
        return m


class BlkaddCtrlCircuit(FeistelAdd, ContrastCircuit):
    """blkadd_ctrl on blkadd_id: contrast vs blkadd_ctrl0 [digit-toggle] (integrated)."""
    circuit_id = 'blkadd_ctrl'
    algorithm = 'blkadd_id'
    base_circuit = 'blkadd_ctrl0'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = 2 * int(torch.randint(0, 5, (1,), generator=g))
                c = clean[i].clone(); c[p] = (int(clean[i, p]) + int(torch.randint(1, 10, (1,), generator=g))) % 10
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class BlkmaxAscCircuit(BlockMax2, DomainCircuit):
    """blkmax_asc on blkmax_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'blkmax_asc'
    algorithm = 'blkmax_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class BlkmaxDescCircuit(BlockMax2, DomainCircuit):
    """blkmax_desc on blkmax_id: isolated via monotone() (integrated)."""
    circuit_id = 'blkmax_desc'
    algorithm = 'blkmax_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class BlkmaxFullCircuit(BlockMax2, DomainCircuit):
    """blkmax_full on blkmax_id: isolated via FULL (integrated)."""
    circuit_id = 'blkmax_full'
    algorithm = 'blkmax_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class BlkmaxRaiseCircuit(BlockMax2, ContrastCircuit):
    """blkmax_raise on blkmax_id: contrast vs blkmax_full [digit-toggle] (integrated)."""
    circuit_id = 'blkmax_raise'
    algorithm = 'blkmax_id'
    base_circuit = 'blkmax_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Errdiff2GridCircuit(ErrorDiffuse2, DomainCircuit):
    """errdiff2_grid on errdiff2_id: isolated via domain([0, 2, 4, 6, 8]) (integrated)."""
    circuit_id = 'errdiff2_grid'
    algorithm = 'errdiff2_id'
    _cond = conditions.domain([0, 2, 4, 6, 8])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Errdiff2FullCircuit(ErrorDiffuse2, DomainCircuit):
    """errdiff2_full on errdiff2_id: isolated via FULL (integrated)."""
    circuit_id = 'errdiff2_full'
    algorithm = 'errdiff2_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Errdiff2KickCircuit(ErrorDiffuse2, ContrastCircuit):
    """errdiff2_kick on errdiff2_id: contrast vs errdiff2_full [digit-toggle] (integrated)."""
    circuit_id = 'errdiff2_kick'
    algorithm = 'errdiff2_id'
    base_circuit = 'errdiff2_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Errdiff3GridCircuit(ErrorDiffuse3, DomainCircuit):
    """errdiff3_grid on errdiff3_id: isolated via domain([0, 3, 6, 9]) (integrated)."""
    circuit_id = 'errdiff3_grid'
    algorithm = 'errdiff3_id'
    _cond = conditions.domain([0, 3, 6, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Errdiff3FullCircuit(ErrorDiffuse3, DomainCircuit):
    """errdiff3_full on errdiff3_id: isolated via FULL (integrated)."""
    circuit_id = 'errdiff3_full'
    algorithm = 'errdiff3_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Errdiff3KickCircuit(ErrorDiffuse3, ContrastCircuit):
    """errdiff3_kick on errdiff3_id: contrast vs errdiff3_full [digit-toggle] (integrated)."""
    circuit_id = 'errdiff3_kick'
    algorithm = 'errdiff3_id'
    base_circuit = 'errdiff3_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Errdiff5GridCircuit(ErrorDiffuse5, DomainCircuit):
    """errdiff5_grid on errdiff5_id: isolated via domain([0, 5]) (integrated)."""
    circuit_id = 'errdiff5_grid'
    algorithm = 'errdiff5_id'
    _cond = conditions.domain([0, 5])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Errdiff5FullCircuit(ErrorDiffuse5, DomainCircuit):
    """errdiff5_full on errdiff5_id: isolated via FULL (integrated)."""
    circuit_id = 'errdiff5_full'
    algorithm = 'errdiff5_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Errdiff5KickCircuit(ErrorDiffuse5, ContrastCircuit):
    """errdiff5_kick on errdiff5_id: contrast vs errdiff5_full [digit-toggle] (integrated)."""
    circuit_id = 'errdiff5_kick'
    algorithm = 'errdiff5_id'
    base_circuit = 'errdiff5_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Boxsum2FullCircuit(BoxSum2, DomainCircuit):
    """boxsum2_full on boxsum2_id: isolated via FULL (integrated)."""
    circuit_id = 'boxsum2_full'
    algorithm = 'boxsum2_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Boxsum2WinCircuit(BoxSum2, ContrastCircuit):
    """boxsum2_win on boxsum2_id: contrast vs boxsum2_full [digit-toggle] (integrated)."""
    circuit_id = 'boxsum2_win'
    algorithm = 'boxsum2_id'
    base_circuit = 'boxsum2_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        hi = min(p + 2 - 1, 9)
        m = torch.zeros(10, dtype=torch.bool); m[p:hi + 1] = True
        return m


class Boxsum3FullCircuit(BoxSum3, DomainCircuit):
    """boxsum3_full on boxsum3_id: isolated via FULL (integrated)."""
    circuit_id = 'boxsum3_full'
    algorithm = 'boxsum3_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Boxsum3WinCircuit(BoxSum3, ContrastCircuit):
    """boxsum3_win on boxsum3_id: contrast vs boxsum3_full [digit-toggle] (integrated)."""
    circuit_id = 'boxsum3_win'
    algorithm = 'boxsum3_id'
    base_circuit = 'boxsum3_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        hi = min(p + 3 - 1, 9)
        m = torch.zeros(10, dtype=torch.bool); m[p:hi + 1] = True
        return m


class Boxsum4FullCircuit(BoxSum4, DomainCircuit):
    """boxsum4_full on boxsum4_id: isolated via FULL (integrated)."""
    circuit_id = 'boxsum4_full'
    algorithm = 'boxsum4_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Boxsum4WinCircuit(BoxSum4, ContrastCircuit):
    """boxsum4_win on boxsum4_id: contrast vs boxsum4_full [digit-toggle] (integrated)."""
    circuit_id = 'boxsum4_win'
    algorithm = 'boxsum4_id'
    base_circuit = 'boxsum4_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        hi = min(p + 4 - 1, 9)
        m = torch.zeros(10, dtype=torch.bool); m[p:hi + 1] = True
        return m


class Boxsum5FullCircuit(BoxSum5, DomainCircuit):
    """boxsum5_full on boxsum5_id: isolated via FULL (integrated)."""
    circuit_id = 'boxsum5_full'
    algorithm = 'boxsum5_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Boxsum5WinCircuit(BoxSum5, ContrastCircuit):
    """boxsum5_win on boxsum5_id: contrast vs boxsum5_full [digit-toggle] (integrated)."""
    circuit_id = 'boxsum5_win'
    algorithm = 'boxsum5_id'
    base_circuit = 'boxsum5_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        hi = min(p + 5 - 1, 9)
        m = torch.zeros(10, dtype=torch.bool); m[p:hi + 1] = True
        return m


class SchmittNobandCircuit(Schmitt, DomainCircuit):
    """schmitt_noband on schmitt_id: isolated via domain([0, 1, 2, 3, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'schmitt_noband'
    algorithm = 'schmitt_id'
    _cond = conditions.domain([0, 1, 2, 3, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class SchmittSetCircuit(Schmitt, ContrastCircuit):
    """schmitt_set on schmitt_id: contrast vs schmitt_noband [digit-toggle] (integrated)."""
    circuit_id = 'schmitt_set'
    algorithm = 'schmitt_id'
    base_circuit = 'schmitt_noband'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Railfence2FullCircuit(RailFence2, DomainCircuit):
    """railfence2_full on railfence2_id: isolated via FULL (integrated)."""
    circuit_id = 'railfence2_full'
    algorithm = 'railfence2_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Railfence2EditCircuit(RailFence2, ContrastCircuit):
    """railfence2_edit on railfence2_id: contrast vs railfence2_full [digit-toggle] (integrated)."""
    circuit_id = 'railfence2_edit'
    algorithm = 'railfence2_id'
    base_circuit = 'railfence2_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = (int(clean[i, p]) + int(torch.randint(1, 10, (1,), generator=g))) % 10
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        rails = [[] for _ in range(2)]
        r, d = 0, 1
        for i in range(10):
            rails[r].append(i)
            if r == 0: d = 1
            elif r == 2 - 1: d = -1
            r += d
        order = [i for rail in rails for i in rail]
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool)
        m[order.index(p)] = True
        return m


class Railfence3FullCircuit(RailFence3, DomainCircuit):
    """railfence3_full on railfence3_id: isolated via FULL (integrated)."""
    circuit_id = 'railfence3_full'
    algorithm = 'railfence3_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Railfence3EditCircuit(RailFence3, ContrastCircuit):
    """railfence3_edit on railfence3_id: contrast vs railfence3_full [digit-toggle] (integrated)."""
    circuit_id = 'railfence3_edit'
    algorithm = 'railfence3_id'
    base_circuit = 'railfence3_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = (int(clean[i, p]) + int(torch.randint(1, 10, (1,), generator=g))) % 10
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        rails = [[] for _ in range(3)]
        r, d = 0, 1
        for i in range(10):
            rails[r].append(i)
            if r == 0: d = 1
            elif r == 3 - 1: d = -1
            r += d
        order = [i for rail in rails for i in rail]
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool)
        m[order.index(p)] = True
        return m


class Railfence4FullCircuit(RailFence4, DomainCircuit):
    """railfence4_full on railfence4_id: isolated via FULL (integrated)."""
    circuit_id = 'railfence4_full'
    algorithm = 'railfence4_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Railfence4EditCircuit(RailFence4, ContrastCircuit):
    """railfence4_edit on railfence4_id: contrast vs railfence4_full [digit-toggle] (integrated)."""
    circuit_id = 'railfence4_edit'
    algorithm = 'railfence4_id'
    base_circuit = 'railfence4_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = (int(clean[i, p]) + int(torch.randint(1, 10, (1,), generator=g))) % 10
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        rails = [[] for _ in range(4)]
        r, d = 0, 1
        for i in range(10):
            rails[r].append(i)
            if r == 0: d = 1
            elif r == 4 - 1: d = -1
            r += d
        order = [i for rail in rails for i in rail]
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(10, dtype=torch.bool)
        m[order.index(p)] = True
        return m


class Qlen3HighCircuit(QueueWalk3, DomainCircuit):
    """qlen3_high on qlen3_id: isolated via domain([3, 4, 5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'qlen3_high'
    algorithm = 'qlen3_id'
    _cond = conditions.domain([3, 4, 5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Qlen3LowCircuit(QueueWalk3, DomainCircuit):
    """qlen3_low on qlen3_id: isolated via domain([0, 1, 2]) (integrated)."""
    circuit_id = 'qlen3_low'
    algorithm = 'qlen3_id'
    _cond = conditions.domain([0, 1, 2])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Qlen3FlipCircuit(QueueWalk3, ContrastCircuit):
    """qlen3_flip on qlen3_id: contrast vs qlen3_high [digit-toggle] (integrated)."""
    circuit_id = 'qlen3_flip'
    algorithm = 'qlen3_id'
    base_circuit = 'qlen3_high'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                hi = int(clean[i, p]) >= 3
                c = clean[i].clone(); c[p] = int(torch.randint(0, 3, (1,), generator=g)) if hi else 3 + int(torch.randint(0, 7, (1,), generator=g))
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Qlen4HighCircuit(QueueWalk4, DomainCircuit):
    """qlen4_high on qlen4_id: isolated via domain([4, 5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'qlen4_high'
    algorithm = 'qlen4_id'
    _cond = conditions.domain([4, 5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Qlen4LowCircuit(QueueWalk4, DomainCircuit):
    """qlen4_low on qlen4_id: isolated via domain([0, 1, 2, 3]) (integrated)."""
    circuit_id = 'qlen4_low'
    algorithm = 'qlen4_id'
    _cond = conditions.domain([0, 1, 2, 3])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Qlen4FlipCircuit(QueueWalk4, ContrastCircuit):
    """qlen4_flip on qlen4_id: contrast vs qlen4_high [digit-toggle] (integrated)."""
    circuit_id = 'qlen4_flip'
    algorithm = 'qlen4_id'
    base_circuit = 'qlen4_high'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                hi = int(clean[i, p]) >= 4
                c = clean[i].clone(); c[p] = int(torch.randint(0, 4, (1,), generator=g)) if hi else 4 + int(torch.randint(0, 6, (1,), generator=g))
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Qlen5HighCircuit(QueueWalk5, DomainCircuit):
    """qlen5_high on qlen5_id: isolated via domain([5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'qlen5_high'
    algorithm = 'qlen5_id'
    _cond = conditions.domain([5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Qlen5LowCircuit(QueueWalk5, DomainCircuit):
    """qlen5_low on qlen5_id: isolated via domain([0, 1, 2, 3, 4]) (integrated)."""
    circuit_id = 'qlen5_low'
    algorithm = 'qlen5_id'
    _cond = conditions.domain([0, 1, 2, 3, 4])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Qlen5FlipCircuit(QueueWalk5, ContrastCircuit):
    """qlen5_flip on qlen5_id: contrast vs qlen5_high [digit-toggle] (integrated)."""
    circuit_id = 'qlen5_flip'
    algorithm = 'qlen5_id'
    base_circuit = 'qlen5_high'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                hi = int(clean[i, p]) >= 5
                c = clean[i].clone(); c[p] = int(torch.randint(0, 5, (1,), generator=g)) if hi else 5 + int(torch.randint(0, 5, (1,), generator=g))
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Qlen6HighCircuit(QueueWalk6, DomainCircuit):
    """qlen6_high on qlen6_id: isolated via domain([6, 7, 8, 9]) (integrated)."""
    circuit_id = 'qlen6_high'
    algorithm = 'qlen6_id'
    _cond = conditions.domain([6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Qlen6LowCircuit(QueueWalk6, DomainCircuit):
    """qlen6_low on qlen6_id: isolated via domain([0, 1, 2, 3, 4, 5]) (integrated)."""
    circuit_id = 'qlen6_low'
    algorithm = 'qlen6_id'
    _cond = conditions.domain([0, 1, 2, 3, 4, 5])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Qlen6FlipCircuit(QueueWalk6, ContrastCircuit):
    """qlen6_flip on qlen6_id: contrast vs qlen6_high [digit-toggle] (integrated)."""
    circuit_id = 'qlen6_flip'
    algorithm = 'qlen6_id'
    base_circuit = 'qlen6_high'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                hi = int(clean[i, p]) >= 6
                c = clean[i].clone(); c[p] = int(torch.randint(0, 6, (1,), generator=g)) if hi else 6 + int(torch.randint(0, 4, (1,), generator=g))
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class BpathAscCircuit(MinBottleneck, DomainCircuit):
    """bpath_asc on bpath_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'bpath_asc'
    algorithm = 'bpath_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class BpathDescCircuit(MinBottleneck, DomainCircuit):
    """bpath_desc on bpath_id: isolated via monotone() (integrated)."""
    circuit_id = 'bpath_desc'
    algorithm = 'bpath_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class BpathFullCircuit(MinBottleneck, DomainCircuit):
    """bpath_full on bpath_id: isolated via FULL (integrated)."""
    circuit_id = 'bpath_full'
    algorithm = 'bpath_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class BpathEditCircuit(MinBottleneck, ContrastCircuit):
    """bpath_edit on bpath_id: contrast vs bpath_full [digit-toggle] (integrated)."""
    circuit_id = 'bpath_edit'
    algorithm = 'bpath_id'
    base_circuit = 'bpath_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class WpathDescCircuit(MaxCapacity, DomainCircuit):
    """wpath_desc on wpath_id: isolated via monotone() (integrated)."""
    circuit_id = 'wpath_desc'
    algorithm = 'wpath_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class WpathAscCircuit(MaxCapacity, DomainCircuit):
    """wpath_asc on wpath_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'wpath_asc'
    algorithm = 'wpath_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class WpathFullCircuit(MaxCapacity, DomainCircuit):
    """wpath_full on wpath_id: isolated via FULL (integrated)."""
    circuit_id = 'wpath_full'
    algorithm = 'wpath_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class WpathEditCircuit(MaxCapacity, ContrastCircuit):
    """wpath_edit on wpath_id: contrast vs wpath_full [digit-toggle] (integrated)."""
    circuit_id = 'wpath_edit'
    algorithm = 'wpath_id'
    base_circuit = 'wpath_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class HsubDescCircuit(HeapSubtreeMax, DomainCircuit):
    """hsub_desc on hsub_id: isolated via monotone() (integrated)."""
    circuit_id = 'hsub_desc'
    algorithm = 'hsub_id'
    _cond = conditions.monotone()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class HsubAscCircuit(HeapSubtreeMax, DomainCircuit):
    """hsub_asc on hsub_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'hsub_asc'
    algorithm = 'hsub_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class HsubFullCircuit(HeapSubtreeMax, DomainCircuit):
    """hsub_full on hsub_id: isolated via FULL (integrated)."""
    circuit_id = 'hsub_full'
    algorithm = 'hsub_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class HsubEditCircuit(HeapSubtreeMax, ContrastCircuit):
    """hsub_edit on hsub_id: contrast vs hsub_full [digit-toggle] (integrated)."""
    circuit_id = 'hsub_edit'
    algorithm = 'hsub_id'
    base_circuit = 'hsub_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class IncrunBin01Circuit(IncRunMax, DomainCircuit):
    """incrun_bin01 on incrun_id: isolated via BINARY (integrated)."""
    circuit_id = 'incrun_bin01'
    algorithm = 'incrun_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class IncrunWholeCircuit(IncRunMax, DomainCircuit):
    """incrun_whole on incrun_id: isolated via FULL (integrated)."""
    circuit_id = 'incrun_whole'
    algorithm = 'incrun_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class IncrunBreakCircuit(IncRunMax, ContrastCircuit):
    """incrun_break on incrun_id: contrast vs incrun_whole [digit-toggle] (integrated)."""
    circuit_id = 'incrun_break'
    algorithm = 'incrun_id'
    base_circuit = 'incrun_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = 1 + int(torch.randint(0, L - 1, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(clean[i, p - 1])
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class HaarZerodetailCircuit(HaarStep, DomainCircuit):
    """haar_zerodetail on haar_id: isolated via Condition('haar_zerodetail', lambda x: all(int(x[k]) == 0 for k in range(1, 10, 2)), conditions.position_domain_sampler({1: [0], 3: [0], 5: [0], 7: [0], 9: [0]})) (integrated)."""
    circuit_id = 'haar_zerodetail'
    algorithm = 'haar_id'
    _cond = conditions.Condition('haar_zerodetail', lambda x: all(int(x[k]) == 0 for k in range(1, 10, 2)), conditions.position_domain_sampler({1: [0], 3: [0], 5: [0], 7: [0], 9: [0]}))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class HaarSingleCircuit(HaarStep, ContrastCircuit):
    """haar_single on haar_id: contrast vs haar_zerodetail [digit-toggle] (integrated)."""
    circuit_id = 'haar_single'
    algorithm = 'haar_id'
    base_circuit = 'haar_zerodetail'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        d = torch.randint(1, 10, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = (clean[rows, p] + d) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        k = p - (p % 2)
        m = torch.zeros(10, dtype=torch.bool); m[k] = True; m[k + 1] = True
        return m


class RunmadWholeCircuit(RunMAD, DomainCircuit):
    """runmad_whole on runmad_id: isolated via FULL (integrated)."""
    circuit_id = 'runmad_whole'
    algorithm = 'runmad_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class MadLocalCircuit(RunMAD, ContrastCircuit):
    """mad_local on runmad_id: contrast vs runmad_whole [digit-toggle] (integrated)."""
    circuit_id = 'mad_local'
    algorithm = 'runmad_id'
    base_circuit = 'runmad_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class ContfracWholeCircuit(ContFrac, DomainCircuit):
    """contfrac_whole on contfrac_id: isolated via FULL (integrated)."""
    circuit_id = 'contfrac_whole'
    algorithm = 'contfrac_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CfEditCircuit(ContFrac, ContrastCircuit):
    """cf_edit on contfrac_id: contrast vs contfrac_whole [digit-toggle] (integrated)."""
    circuit_id = 'cf_edit'
    algorithm = 'contfrac_id'
    base_circuit = 'contfrac_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(160):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class S3prodFullCircuit(GroupProductS3, DomainCircuit):
    """s3prod_full on s3prod_id: isolated via FULL (integrated)."""
    circuit_id = 's3prod_full'
    algorithm = 's3prod_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class S3prodEditCircuit(GroupProductS3, ContrastCircuit):
    """s3prod_edit on s3prod_id: contrast vs s3prod_full [digit-toggle] (integrated)."""
    circuit_id = 's3prod_edit'
    algorithm = 's3prod_id'
    base_circuit = 's3prod_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) >= int((clean_row != corrupt_row).nonzero()[0])


class D4prodFullCircuit(GroupProductD4, DomainCircuit):
    """d4prod_full on d4prod_id: isolated via FULL (integrated)."""
    circuit_id = 'd4prod_full'
    algorithm = 'd4prod_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class D4prodEditCircuit(GroupProductD4, ContrastCircuit):
    """d4prod_edit on d4prod_id: contrast vs d4prod_full [digit-toggle] (integrated)."""
    circuit_id = 'd4prod_edit'
    algorithm = 'd4prod_id'
    base_circuit = 'd4prod_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) >= int((clean_row != corrupt_row).nonzero()[0])


class Q8prodFullCircuit(GroupProductQ8, DomainCircuit):
    """q8prod_full on q8prod_id: isolated via FULL (integrated)."""
    circuit_id = 'q8prod_full'
    algorithm = 'q8prod_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Q8prodEditCircuit(GroupProductQ8, ContrastCircuit):
    """q8prod_edit on q8prod_id: contrast vs q8prod_full [digit-toggle] (integrated)."""
    circuit_id = 'q8prod_edit'
    algorithm = 'q8prod_id'
    base_circuit = 'q8prod_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) >= int((clean_row != corrupt_row).nonzero()[0])


class D5prodFullCircuit(GroupProductD5, DomainCircuit):
    """d5prod_full on d5prod_id: isolated via FULL (integrated)."""
    circuit_id = 'd5prod_full'
    algorithm = 'd5prod_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class D5prodEditCircuit(GroupProductD5, ContrastCircuit):
    """d5prod_edit on d5prod_id: contrast vs d5prod_full [digit-toggle] (integrated)."""
    circuit_id = 'd5prod_edit'
    algorithm = 'd5prod_id'
    base_circuit = 'd5prod_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) >= int((clean_row != corrupt_row).nonzero()[0])


class D4charFullCircuit(D4Character, DomainCircuit):
    """d4char_full on d4char_id: isolated via FULL (integrated)."""
    circuit_id = 'd4char_full'
    algorithm = 'd4char_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class D4charEditCircuit(D4Character, ContrastCircuit):
    """d4char_edit on d4char_id: contrast vs d4char_full [digit-toggle] (integrated)."""
    circuit_id = 'd4char_edit'
    algorithm = 'd4char_id'
    base_circuit = 'd4char_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class D5actFullCircuit(GroupActionD5, DomainCircuit):
    """d5act_full on d5act_id: isolated via FULL (integrated)."""
    circuit_id = 'd5act_full'
    algorithm = 'd5act_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class D5actEditCircuit(GroupActionD5, ContrastCircuit):
    """d5act_edit on d5act_id: contrast vs d5act_full [digit-toggle] (integrated)."""
    circuit_id = 'd5act_edit'
    algorithm = 'd5act_id'
    base_circuit = 'd5act_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class RangeaddWholeCircuit(RunningRangeAdd, DomainCircuit):
    """rangeadd_whole on rangeadd_id: isolated via FULL (integrated)."""
    circuit_id = 'rangeadd_whole'
    algorithm = 'rangeadd_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class RangeaddMoveCircuit(RunningRangeAdd, ContrastCircuit):
    """rangeadd_move on rangeadd_id: contrast vs rangeadd_whole [digit-toggle] (integrated)."""
    circuit_id = 'rangeadd_move'
    algorithm = 'rangeadd_id'
    base_circuit = 'rangeadd_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class MedaddWholeCircuit(RunningMedianAdd, DomainCircuit):
    """medadd_whole on medadd_id: isolated via FULL (integrated)."""
    circuit_id = 'medadd_whole'
    algorithm = 'medadd_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class MedaddMoveCircuit(RunningMedianAdd, ContrastCircuit):
    """medadd_move on medadd_id: contrast vs medadd_whole [digit-toggle] (integrated)."""
    circuit_id = 'medadd_move'
    algorithm = 'medadd_id'
    base_circuit = 'medadd_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class TribscanWholeCircuit(ThirdOrderScan, DomainCircuit):
    """tribscan_whole on tribscan_id: isolated via FULL (integrated)."""
    circuit_id = 'tribscan_whole'
    algorithm = 'tribscan_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class TribscanMoveCircuit(ThirdOrderScan, ContrastCircuit):
    """tribscan_move on tribscan_id: contrast vs tribscan_whole [digit-toggle] (integrated)."""
    circuit_id = 'tribscan_move'
    algorithm = 'tribscan_id'
    base_circuit = 'tribscan_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class Rotor3WholeCircuit(LinearRotor3, DomainCircuit):
    """rotor3_whole on rotor3_id: isolated via FULL (integrated)."""
    circuit_id = 'rotor3_whole'
    algorithm = 'rotor3_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Rotor3EditCircuit(LinearRotor3, ContrastCircuit):
    """rotor3_edit on rotor3_id: contrast vs rotor3_whole [digit-toggle] (integrated)."""
    circuit_id = 'rotor3_edit'
    algorithm = 'rotor3_id'
    base_circuit = 'rotor3_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class ModecntaddWholeCircuit(RunningModeCountAdd, DomainCircuit):
    """modecntadd_whole on modecntadd_id: isolated via FULL (integrated)."""
    circuit_id = 'modecntadd_whole'
    algorithm = 'modecntadd_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class ModecntaddEditCircuit(RunningModeCountAdd, ContrastCircuit):
    """modecntadd_edit on modecntadd_id: contrast vs modecntadd_whole [digit-toggle] (integrated)."""
    circuit_id = 'modecntadd_edit'
    algorithm = 'modecntadd_id'
    base_circuit = 'modecntadd_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class OscposWholeCircuit(Oscpos, DomainCircuit):
    """oscpos_whole on oscpos_id: isolated via FULL (integrated)."""
    circuit_id = 'oscpos_whole'
    algorithm = 'oscpos_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class OscposEditCircuit(Oscpos, ContrastCircuit):
    """oscpos_edit on oscpos_id: contrast vs oscpos_whole [digit-toggle] (integrated)."""
    circuit_id = 'oscpos_edit'
    algorithm = 'oscpos_id'
    base_circuit = 'oscpos_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class OscmomWholeCircuit(Oscmom, DomainCircuit):
    """oscmom_whole on oscmom_id: isolated via FULL (integrated)."""
    circuit_id = 'oscmom_whole'
    algorithm = 'oscmom_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class OscmomEditCircuit(Oscmom, ContrastCircuit):
    """oscmom_edit on oscmom_id: contrast vs oscmom_whole [digit-toggle] (integrated)."""
    circuit_id = 'oscmom_edit'
    algorithm = 'oscmom_id'
    base_circuit = 'oscmom_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class RefracAlllowCircuit(RefractoryVeto, DomainCircuit):
    """refrac_alllow on refrac_id: isolated via domain([0, 1, 2, 3, 4, 5, 6]) (integrated)."""
    circuit_id = 'refrac_alllow'
    algorithm = 'refrac_id'
    _cond = conditions.domain([0, 1, 2, 3, 4, 5, 6])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class RefracEditCircuit(RefractoryVeto, ContrastCircuit):
    """refrac_edit on refrac_id: contrast vs refrac_alllow [digit-toggle] (integrated)."""
    circuit_id = 'refrac_edit'
    algorithm = 'refrac_id'
    base_circuit = 'refrac_alllow'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class MexhatMonotoneCircuit(MexicanHatVeto, DomainCircuit):
    """mexhat_monotone on mexhat_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'mexhat_monotone'
    algorithm = 'mexhat_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class MexhatFullCircuit(MexicanHatVeto, DomainCircuit):
    """mexhat_full on mexhat_id: isolated via FULL (integrated)."""
    circuit_id = 'mexhat_full'
    algorithm = 'mexhat_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class MexhatFillCircuit(MexicanHatVeto, ContrastCircuit):
    """mexhat_fill on mexhat_id: contrast vs mexhat_full [digit-toggle] (integrated)."""
    circuit_id = 'mexhat_fill'
    algorithm = 'mexhat_id'
    base_circuit = 'mexhat_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class MaxfreqFullCircuit(MaxFrequency, DomainCircuit):
    """maxfreq_full on maxfreq_id: isolated via FULL (integrated)."""
    circuit_id = 'maxfreq_full'
    algorithm = 'maxfreq_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class MaxfreqAddmodeCircuit(MaxFrequency, ContrastCircuit):
    """maxfreq_addmode on maxfreq_id: contrast vs maxfreq_full [digit-toggle] (integrated)."""
    circuit_id = 'maxfreq_addmode'
    algorithm = 'maxfreq_id'
    base_circuit = 'maxfreq_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class DucciBinaryCircuit(DucciStep, DomainCircuit):
    """ducci_binary on ducci_id: isolated via BINARY (integrated)."""
    circuit_id = 'ducci_binary'
    algorithm = 'ducci_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DucciFullCircuit(DucciStep, DomainCircuit):
    """ducci_full on ducci_id: isolated via FULL (integrated)."""
    circuit_id = 'ducci_full'
    algorithm = 'ducci_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DucciEditCircuit(DucciStep, ContrastCircuit):
    """ducci_edit on ducci_id: contrast vs ducci_full [digit-toggle] (integrated)."""
    circuit_id = 'ducci_edit'
    algorithm = 'ducci_id'
    base_circuit = 'ducci_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        def du(r):
            return [abs(r[i] - r[(i + 1) % 10]) for i in range(10)]
        a = du(clean_row.tolist()); b = du(corrupt_row.tolist())
        return torch.tensor([a[i] != b[i] for i in range(10)])


class PgcdFullCircuit(PairGcd, DomainCircuit):
    """pgcd_full on pgcd_id: isolated via FULL (integrated)."""
    circuit_id = 'pgcd_full'
    algorithm = 'pgcd_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PgcdEditCircuit(PairGcd, ContrastCircuit):
    """pgcd_edit on pgcd_id: contrast vs pgcd_full [digit-toggle] (integrated)."""
    circuit_id = 'pgcd_edit'
    algorithm = 'pgcd_id'
    base_circuit = 'pgcd_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class FibscanFullCircuit(SecondOrderScan, DomainCircuit):
    """fibscan_full on fibscan_id: isolated via FULL (integrated)."""
    circuit_id = 'fibscan_full'
    algorithm = 'fibscan_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class FibscanMoveCircuit(SecondOrderScan, ContrastCircuit):
    """fibscan_move on fibscan_id: contrast vs fibscan_full [digit-toggle] (integrated)."""
    circuit_id = 'fibscan_move'
    algorithm = 'fibscan_id'
    base_circuit = 'fibscan_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class MeanintFullCircuit(NivenByCount, DomainCircuit):
    """meanint_full on meanint_id: isolated via FULL (integrated)."""
    circuit_id = 'meanint_full'
    algorithm = 'meanint_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class MeanintEditCircuit(NivenByCount, ContrastCircuit):
    """meanint_edit on meanint_id: contrast vs meanint_full [digit-toggle] (integrated)."""
    circuit_id = 'meanint_edit'
    algorithm = 'meanint_id'
    base_circuit = 'meanint_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class EuclstepFullCircuit(EuclidStep, DomainCircuit):
    """euclstep_full on euclstep_id: isolated via FULL (integrated)."""
    circuit_id = 'euclstep_full'
    algorithm = 'euclstep_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class EuclstepEditCircuit(EuclidStep, ContrastCircuit):
    """euclstep_edit on euclstep_id: contrast vs euclstep_full [digit-toggle] (integrated)."""
    circuit_id = 'euclstep_edit'
    algorithm = 'euclstep_id'
    base_circuit = 'euclstep_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        def eu(r):
            return [r[0]] + [r[i - 1] % r[i] if r[i] != 0 else r[i - 1] for i in range(1, 10)]
        a = eu(clean_row.tolist()); b = eu(corrupt_row.tolist())
        return torch.tensor([a[i] != b[i] for i in range(10)])


class PaircarryFullCircuit(PairCarry, DomainCircuit):
    """paircarry_full on paircarry_id: isolated via FULL (integrated)."""
    circuit_id = 'paircarry_full'
    algorithm = 'paircarry_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PaircarryCrossCircuit(PairCarry, ContrastCircuit):
    """paircarry_cross on paircarry_id: contrast vs paircarry_full [digit-toggle] (integrated)."""
    circuit_id = 'paircarry_cross'
    algorithm = 'paircarry_id'
    base_circuit = 'paircarry_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        def pc(r):
            return [0] + [1 if r[i] + r[i - 1] >= 10 else 0 for i in range(1, 10)]
        a = pc(clean_row.tolist()); b = pc(corrupt_row.tolist())
        return torch.tensor([a[i] != b[i] for i in range(10)])


class InvcountFullCircuit(InversionCount, DomainCircuit):
    """invcount_full on invcount_id: isolated via FULL (integrated)."""
    circuit_id = 'invcount_full'
    algorithm = 'invcount_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class InvcountEditCircuit(InversionCount, ContrastCircuit):
    """invcount_edit on invcount_id: contrast vs invcount_full [digit-toggle] (integrated)."""
    circuit_id = 'invcount_edit'
    algorithm = 'invcount_id'
    base_circuit = 'invcount_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class MajindexFullCircuit(MajorIndex, DomainCircuit):
    """majindex_full on majindex_id: isolated via FULL (integrated)."""
    circuit_id = 'majindex_full'
    algorithm = 'majindex_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class MajindexEditCircuit(MajorIndex, ContrastCircuit):
    """majindex_edit on majindex_id: contrast vs majindex_full [digit-toggle] (integrated)."""
    circuit_id = 'majindex_edit'
    algorithm = 'majindex_id'
    base_circuit = 'majindex_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class NimhistFullCircuit(NimHistogram, DomainCircuit):
    """nimhist_full on nimhist_id: isolated via FULL (integrated)."""
    circuit_id = 'nimhist_full'
    algorithm = 'nimhist_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class NimhistEditCircuit(NimHistogram, ContrastCircuit):
    """nimhist_edit on nimhist_id: contrast vs nimhist_full [digit-toggle] (integrated)."""
    circuit_id = 'nimhist_edit'
    algorithm = 'nimhist_id'
    base_circuit = 'nimhist_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class GraviclumpBinaryCircuit(MeanRelax, DomainCircuit):
    """graviclump_binary on graviclump_id: isolated via BINARY (integrated)."""
    circuit_id = 'graviclump_binary'
    algorithm = 'graviclump_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class GraviclumpFullCircuit(MeanRelax, DomainCircuit):
    """graviclump_full on graviclump_id: isolated via FULL (integrated)."""
    circuit_id = 'graviclump_full'
    algorithm = 'graviclump_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class GraviclumpEditCircuit(MeanRelax, ContrastCircuit):
    """graviclump_edit on graviclump_id: contrast vs graviclump_full [digit-toggle] (integrated)."""
    circuit_id = 'graviclump_edit'
    algorithm = 'graviclump_id'
    base_circuit = 'graviclump_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class EntrainPin3Circuit(AnchorNudge, DomainCircuit):
    """entrain_pin3 on entrain_id: isolated via pin(0, [3]) (integrated)."""
    circuit_id = 'entrain_pin3'
    algorithm = 'entrain_id'
    _cond = conditions.pin(0, [3])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class EntrainFullCircuit(AnchorNudge, DomainCircuit):
    """entrain_full on entrain_id: isolated via FULL (integrated)."""
    circuit_id = 'entrain_full'
    algorithm = 'entrain_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class EntrainFollowerCircuit(AnchorNudge, ContrastCircuit):
    """entrain_follower on entrain_id: contrast vs entrain_full [digit-toggle] (integrated)."""
    circuit_id = 'entrain_follower'
    algorithm = 'entrain_id'
    base_circuit = 'entrain_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = 1 + int(torch.randint(0, L - 1, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return torch.arange(10) == int((clean_row != corrupt_row).nonzero()[0])


class PhaselockFullCircuit(PhaseNudge, DomainCircuit):
    """phaselock_full on phaselock_id: isolated via FULL (integrated)."""
    circuit_id = 'phaselock_full'
    algorithm = 'phaselock_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PhaselockEditCircuit(PhaseNudge, ContrastCircuit):
    """phaselock_edit on phaselock_id: contrast vs phaselock_full [digit-toggle] (integrated)."""
    circuit_id = 'phaselock_edit'
    algorithm = 'phaselock_id'
    base_circuit = 'phaselock_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class HeatxFullCircuit(HeatExchange, DomainCircuit):
    """heatx_full on heatx_id: isolated via FULL (integrated)."""
    circuit_id = 'heatx_full'
    algorithm = 'heatx_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class HeatxEditCircuit(HeatExchange, ContrastCircuit):
    """heatx_edit on heatx_id: contrast vs heatx_full [digit-toggle] (integrated)."""
    circuit_id = 'heatx_edit'
    algorithm = 'heatx_id'
    base_circuit = 'heatx_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class E2symFullCircuit(ElemSym2, DomainCircuit):
    """e2sym_full on e2sym_id: isolated via FULL (integrated)."""
    circuit_id = 'e2sym_full'
    algorithm = 'e2sym_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class E2symEditCircuit(ElemSym2, ContrastCircuit):
    """e2sym_edit on e2sym_id: contrast vs e2sym_full [digit-toggle] (integrated)."""
    circuit_id = 'e2sym_edit'
    algorithm = 'e2sym_id'
    base_circuit = 'e2sym_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class RunmexBinaryCircuit(RunningMex, DomainCircuit):
    """runmex_binary on runmex_id: isolated via BINARY (integrated)."""
    circuit_id = 'runmex_binary'
    algorithm = 'runmex_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class RunmexFullCircuit(RunningMex, DomainCircuit):
    """runmex_full on runmex_id: isolated via FULL (integrated)."""
    circuit_id = 'runmex_full'
    algorithm = 'runmex_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class RunmexPlant0Circuit(RunningMex, ContrastCircuit):
    """runmex_plant0 on runmex_id: contrast vs runmex_full [digit-toggle] (integrated)."""
    circuit_id = 'runmex_plant0'
    algorithm = 'runmex_id'
    base_circuit = 'runmex_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt


class PaladdWholeCircuit(MirrorAdd, DomainCircuit):
    """paladd_whole on paladd_id: isolated via FULL (integrated)."""
    circuit_id = 'paladd_whole'
    algorithm = 'paladd_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PaladdPalindromeCircuit(MirrorAdd, DomainCircuit):
    """paladd_palindrome on paladd_id: isolated via Condition('palindrome', lambda x: bool((x == x.flip(0)).all()), lambda batch, seed: (lambda h: torch.cat([h, h.flip(1)], dim=1))(torch.randint(0, 10, (batch, 5), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'paladd_palindrome'))))) (integrated)."""
    circuit_id = 'paladd_palindrome'
    algorithm = 'paladd_id'
    _cond = conditions.Condition('palindrome', lambda x: bool((x == x.flip(0)).all()), lambda batch, seed: (lambda h: torch.cat([h, h.flip(1)], dim=1))(torch.randint(0, 10, (batch, 5), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'paladd_palindrome')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PaladdUniformCircuit(MirrorAdd, DomainCircuit):
    """paladd_uniform on paladd_id: isolated via Condition('all_equal', lambda x: bool((x == x[0]).all()), lambda batch, seed: torch.randint(0, 10, (batch, 1), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'paladd_uniform'))).expand(batch, 10).contiguous()) (integrated)."""
    circuit_id = 'paladd_uniform'
    algorithm = 'paladd_id'
    _cond = conditions.Condition('all_equal', lambda x: bool((x == x[0]).all()), lambda batch, seed: torch.randint(0, 10, (batch, 1), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'paladd_uniform'))).expand(batch, 10).contiguous())
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PaladdEditCircuit(MirrorAdd, ContrastCircuit):
    """paladd_edit on paladd_id: contrast vs paladd_whole [value-perturb] (integrated)."""
    circuit_id = 'paladd_edit'
    algorithm = 'paladd_id'
    base_circuit = 'paladd_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        clean = DomainCircuit.sample(self, batch, seed)
        g = torch.Generator().manual_seed(int(seed) + 1)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        rows = torch.arange(batch)
        corrupt = clean.clone()
        delta = torch.randint(1, 10, (batch,), generator=g)
        corrupt[rows, p] = (clean[rows, p] + delta) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class PalmaxWholeCircuit(MirrorMax, DomainCircuit):
    """palmax_whole on palmax_id: isolated via FULL (integrated)."""
    circuit_id = 'palmax_whole'
    algorithm = 'palmax_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PalmaxPalindromeCircuit(MirrorMax, DomainCircuit):
    """palmax_palindrome on palmax_id: isolated via Condition('palindrome', lambda x: bool((x == x.flip(0)).all()), lambda batch, seed: (lambda h: torch.cat([h, h.flip(1)], dim=1))(torch.randint(0, 10, (batch, 5), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'palmax_palindrome'))))) (integrated)."""
    circuit_id = 'palmax_palindrome'
    algorithm = 'palmax_id'
    _cond = conditions.Condition('palindrome', lambda x: bool((x == x.flip(0)).all()), lambda batch, seed: (lambda h: torch.cat([h, h.flip(1)], dim=1))(torch.randint(0, 10, (batch, 5), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'palmax_palindrome')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PalmaxEditCircuit(MirrorMax, ContrastCircuit):
    """palmax_edit on palmax_id: contrast vs palmax_whole [value-perturb] (integrated)."""
    circuit_id = 'palmax_edit'
    algorithm = 'palmax_id'
    base_circuit = 'palmax_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for _t in range(256):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class PalminWholeCircuit(MirrorMin, DomainCircuit):
    """palmin_whole on palmin_id: isolated via FULL (integrated)."""
    circuit_id = 'palmin_whole'
    algorithm = 'palmin_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PalminPalindromeCircuit(MirrorMin, DomainCircuit):
    """palmin_palindrome on palmin_id: isolated via Condition('palindrome', lambda x: bool((x == x.flip(0)).all()), lambda batch, seed: (lambda h: torch.cat([h, h.flip(1)], dim=1))(torch.randint(0, 10, (batch, 5), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'palmin_palindrome'))))) (integrated)."""
    circuit_id = 'palmin_palindrome'
    algorithm = 'palmin_id'
    _cond = conditions.Condition('palindrome', lambda x: bool((x == x.flip(0)).all()), lambda batch, seed: (lambda h: torch.cat([h, h.flip(1)], dim=1))(torch.randint(0, 10, (batch, 5), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'palmin_palindrome')))))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class RankaddWholeCircuit(OnlineRankAdd, DomainCircuit):
    """rankadd_whole on rankadd_id: isolated via FULL (integrated)."""
    circuit_id = 'rankadd_whole'
    algorithm = 'rankadd_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class RankaddSelfCircuit(OnlineRankAdd, ContrastCircuit):
    """rankadd_self on rankadd_id: contrast vs rankadd_whole [value-perturb-control] (integrated)."""
    circuit_id = 'rankadd_self'
    algorithm = 'rankadd_id'
    base_circuit = 'rankadd_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L - 1, (1,), generator=g))
                a = int(clean[i, p])
                rank_a = int((clean[i, :p] < a).sum())
                b = int(torch.randint(0, 10, (1,), generator=g))
                if b == a: continue
                if int((clean[i, :p] < b).sum()) != rank_a: continue
                c = clean[i].clone(); c[p] = b
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class RankaddRaiseCircuit(OnlineRankAdd, ContrastCircuit):
    """rankadd_raise on rankadd_id: contrast vs rankadd_whole [value-perturb] (integrated)."""
    circuit_id = 'rankadd_raise'
    algorithm = 'rankadd_id'
    base_circuit = 'rankadd_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(1, L - 1, (1,), generator=g))
                lo = max(int(clean[i, :p].max()), int(clean[i, p])) + 1
                if lo > 9: continue
                b = int(torch.randint(lo, 10, (1,), generator=g))
                c = clean[i].clone(); c[p] = b
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class SigndiffWholeCircuit(SignDiffuse, DomainCircuit):
    """signdiff_whole on signdiff_id: isolated via FULL (integrated)."""
    circuit_id = 'signdiff_whole'
    algorithm = 'signdiff_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class SigndiffFlatCircuit(SignDiffuse, DomainCircuit):
    """signdiff_flat on signdiff_id: isolated via Condition('all_equal', lambda x: bool((x == x[0]).all()), lambda batch, seed: torch.randint(0, 10, (batch, 1), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'signdiff_flat'))).expand(batch, 10).contiguous()) (integrated)."""
    circuit_id = 'signdiff_flat'
    algorithm = 'signdiff_id'
    _cond = conditions.Condition('all_equal', lambda x: bool((x == x[0]).all()), lambda batch, seed: torch.randint(0, 10, (batch, 1), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'signdiff_flat'))).expand(batch, 10).contiguous())
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Dblge5WholeCircuit(DoubleGE, DomainCircuit):
    """dblge5_whole on dblge5_id: isolated via FULL (integrated)."""
    circuit_id = 'dblge5_whole'
    algorithm = 'dblge5_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Dblge5OdddetectCircuit(DoubleGE, DomainCircuit):
    """dblge5_odddetect on dblge5_id: isolated via domain([0, 2, 4, 5, 7, 9]) (integrated)."""
    circuit_id = 'dblge5_odddetect'
    algorithm = 'dblge5_id'
    _cond = conditions.domain([0, 2, 4, 5, 7, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Dblge5LowcopyCircuit(DoubleGE, DomainCircuit):
    """dblge5_lowcopy on dblge5_id: isolated via domain([0, 1, 2, 3, 4]) (integrated)."""
    circuit_id = 'dblge5_lowcopy'
    algorithm = 'dblge5_id'
    _cond = conditions.domain([0, 1, 2, 3, 4])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Dblge5HighdblCircuit(DoubleGE, DomainCircuit):
    """dblge5_highdbl on dblge5_id: isolated via domain([5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'dblge5_highdbl'
    algorithm = 'dblge5_id'
    _cond = conditions.domain([5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Dblge5PromoteCircuit(DoubleGE, ContrastCircuit):
    """dblge5_promote on dblge5_id: contrast vs dblge5_whole [threshold-straddle] (integrated)."""
    circuit_id = 'dblge5_promote'
    algorithm = 'dblge5_id'
    base_circuit = 'dblge5_whole'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length; rows = torch.arange(batch)
        clean = torch.randint(0, 10, (batch, L), generator=g)
        p = torch.randint(0, L, (batch,), generator=g)
        low = torch.randint(0, 5, (batch,), generator=g)
        high = torch.randint(5, 10, (batch,), generator=g)
        collide = ((2 * high) % 10 == low)
        while bool(collide.any()):
            high = torch.where(collide, torch.randint(5, 10, (batch,), generator=g), high)
            collide = ((2 * high) % 10 == low)
        clean[rows, p] = low
        corrupt = clean.clone(); corrupt[rows, p] = high
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class MidaddWholeCircuit(MidrangeAdd, DomainCircuit):
    """midadd_whole on midadd_id: isolated via FULL (integrated)."""
    circuit_id = 'midadd_whole'
    algorithm = 'midadd_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class MidaddSelfFrozenCircuit(MidrangeAdd, ContrastCircuit):
    """midadd_self_frozen on midadd_id: contrast vs midadd_whole [value-perturb-control] (integrated)."""
    circuit_id = 'midadd_self_frozen'
    algorithm = 'midadd_id'
    base_circuit = 'midadd_whole'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(256):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t > 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
            for i in idx.tolist():
                r = clean[i]
                hi = torch.cummax(r, 0).values; lo = torch.cummin(r, 0).values
                for p in (torch.randperm(L - 1, generator=g) + 1).tolist():
                    Mb = int(hi[p - 1]); mb = int(lo[p - 1])
                    if Mb > mb and mb <= int(r[p]) <= Mb:
                        ch = [v for v in range(mb, Mb + 1) if v != int(r[p])]
                        c = r.clone(); c[p] = ch[int(torch.randint(0, len(ch), (1,), generator=g))]
                        corrupt[i] = c; need[i] = False; break
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class MidaddNewExtremeCircuit(MidrangeAdd, ContrastCircuit):
    """midadd_new_extreme on midadd_id: contrast vs midadd_whole [value-perturb] (integrated)."""
    circuit_id = 'midadd_new_extreme'
    algorithm = 'midadd_id'
    base_circuit = 'midadd_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 8, (batch, L), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(0, L - 1, (batch,), generator=g)
        corrupt = clean.clone()
        corrupt[rows, p] = 9
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class SeedsumWholeCircuit(SpacedSeedSum, DomainCircuit):
    """seedsum_whole on seedsum_id: isolated via FULL (integrated)."""
    circuit_id = 'seedsum_whole'
    algorithm = 'seedsum_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class SeedsumSameCircuit(SpacedSeedSum, DomainCircuit):
    """seedsum_same on seedsum_id: isolated via Condition('all_equal', lambda x: bool((x == x[0]).all()), lambda batch, seed: torch.randint(0, 10, (batch, 1), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'seedsum_same'))).expand(batch, 10).contiguous()) (integrated)."""
    circuit_id = 'seedsum_same'
    algorithm = 'seedsum_id'
    _cond = conditions.Condition('all_equal', lambda x: bool((x == x[0]).all()), lambda batch, seed: torch.randint(0, 10, (batch, 1), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'seedsum_same'))).expand(batch, 10).contiguous())
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class SeedsumTapeditCircuit(SpacedSeedSum, ContrastCircuit):
    """seedsum_tapedit on seedsum_id: contrast vs seedsum_whole [value-perturb] (integrated)."""
    circuit_id = 'seedsum_tapedit'
    algorithm = 'seedsum_id'
    base_circuit = 'seedsum_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        rows = torch.arange(batch)
        p = torch.randint(3, L, (batch,), generator=g)
        new = torch.randint(0, 10, (batch,), generator=g)
        for _ in range(32):
            same = new == clean[rows, p]
            if not int(same.any()): break
            new[same] = torch.randint(0, 10, (int(same.sum()),), generator=g)
        assert not int((new == clean[rows, p]).any())
        corrupt = clean.clone(); corrupt[rows, p] = new
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class ArgposaddWholeCircuit(ArgPosAdd, DomainCircuit):
    """argposadd_whole on argposadd_id: isolated via FULL (integrated)."""
    circuit_id = 'argposadd_whole'
    algorithm = 'argposadd_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class ArgposaddNewMaxCircuit(ArgPosAdd, ContrastCircuit):
    """argposadd_new_max on argposadd_id: contrast vs argposadd_whole [value-perturb] (integrated)."""
    circuit_id = 'argposadd_new_max'
    algorithm = 'argposadd_id'
    base_circuit = 'argposadd_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(1, L - 1, (1,), generator=g))
                pre_max = int(clean[i, :p].max())
                if int(clean[i, p]) > pre_max: continue
                g_max = int(clean[i].max())
                if g_max + 1 > 9: continue
                b = int(torch.randint(g_max + 1, 10, (1,), generator=g))
                c = clean[i].clone(); c[p] = b
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class ArgposaddInteriorCircuit(ArgPosAdd, ContrastCircuit):
    """argposadd_interior on argposadd_id: contrast vs argposadd_whole [value-perturb-control] (integrated)."""
    circuit_id = 'argposadd_interior'
    algorithm = 'argposadd_id'
    base_circuit = 'argposadd_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = DomainCircuit.sample(self, idx.numel(), int(seed) + t + 1)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(1, L, (1,), generator=g))
                pre_max = int(clean[i, :p].max())
                a = int(clean[i, p])
                if pre_max < 2 or a >= pre_max: continue
                b = int(torch.randint(0, pre_max, (1,), generator=g))
                if b == a: continue
                c = clean[i].clone(); c[p] = b
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class LanemaxWholeCircuit(LaneRunningMax, DomainCircuit):
    """lanemax_whole on lanemax_id: isolated via FULL (integrated)."""
    circuit_id = 'lanemax_whole'
    algorithm = 'lanemax_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class LanemaxLanesortedCircuit(LaneRunningMax, DomainCircuit):
    """lanemax_lanesorted on lanemax_id: isolated via Condition('lane_sorted', lambda x: bool((x[0::2][1:] >= x[0::2][:-1]).all()) and bool((x[1::2][1:] >= x[1::2][:-1]).all()), lambda batch, seed, _a=conditions.monotone_sampler(True, 5): torch.stack([_a(batch, 2 * int(seed)), _a(batch, 2 * int(seed) + 1)], dim=2).reshape(batch, 10)) (integrated)."""
    circuit_id = 'lanemax_lanesorted'
    algorithm = 'lanemax_id'
    _cond = conditions.Condition('lane_sorted', lambda x: bool((x[0::2][1:] >= x[0::2][:-1]).all()) and bool((x[1::2][1:] >= x[1::2][:-1]).all()), lambda batch, seed, _a=conditions.monotone_sampler(True, 5): torch.stack([_a(batch, 2 * int(seed)), _a(batch, 2 * int(seed) + 1)], dim=2).reshape(batch, 10))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class LanemaxRaiseCircuit(LaneRunningMax, ContrastCircuit):
    """lanemax_raise on lanemax_id: contrast vs lanemax_whole [value-perturb] (integrated)."""
    circuit_id = 'lanemax_raise'
    algorithm = 'lanemax_id'
    base_circuit = 'lanemax_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 9, (batch, L), generator=g)
        p = torch.randint(0, L, (batch,), generator=g)
        rows = torch.arange(batch)
        corrupt = clean.clone()
        corrupt[rows, p] = 9
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class PancakeWholeCircuit(PancakeFlip, DomainCircuit):
    """pancake_whole on pancake_id: isolated via FULL (integrated)."""
    circuit_id = 'pancake_whole'
    algorithm = 'pancake_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PancakeMaxfrontCircuit(PancakeFlip, DomainCircuit):
    """pancake_maxfront on pancake_id: isolated via pin(0, [9]) (integrated)."""
    circuit_id = 'pancake_maxfront'
    algorithm = 'pancake_id'
    _cond = conditions.pin(0, [9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PancakeAscsortedCircuit(PancakeFlip, DomainCircuit):
    """pancake_ascsorted on pancake_id: isolated via Condition('pancake_asc_strict', lambda x: bool((x[1:] >= x[:-1]).all()) and int(x[-1]) > int(x[-2]), lambda batch, seed, _b=conditions.bounded_sum_sampler(8, 10, 0, 9): torch.cumsum(_b(batch, seed) + torch.tensor([0] * 9 + [1]), dim=1)) (integrated)."""
    circuit_id = 'pancake_ascsorted'
    algorithm = 'pancake_id'
    _cond = conditions.Condition('pancake_asc_strict', lambda x: bool((x[1:] >= x[:-1]).all()) and int(x[-1]) > int(x[-2]), lambda batch, seed, _b=conditions.bounded_sum_sampler(8, 10, 0, 9): torch.cumsum(_b(batch, seed) + torch.tensor([0] * 9 + [1]), dim=1))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class PancakeMoveargmaxCircuit(PancakeFlip, ContrastCircuit):
    """pancake_moveargmax on pancake_id: contrast vs pancake_whole [value-perturb (relocate argmax to a later strict max -> grows the reversed prefix)] (integrated)."""
    circuit_id = 'pancake_moveargmax'
    algorithm = 'pancake_id'
    base_circuit = 'pancake_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        cleans, corrupts, tries = [], [], 0
        while len(cleans) < batch:
            block = torch.randint(0, 10, (max(4 * (batch - len(cleans)), 512), L), generator=g)
            for row in block:
                m = int(row.max()); k = int((row == m).nonzero()[0])
                if k >= L - 1 or m >= 9:
                    continue
                q = int(torch.randint(k + 1, L, (1,), generator=g))
                corrupt = row.clone(); corrupt[q] = int(torch.randint(m + 1, 10, (1,), generator=g))
                if bool((self.eval(row) != self.eval(corrupt)).any()):
                    cleans.append(row.clone()); corrupts.append(corrupt)
                    if len(cleans) == batch:
                        break
            tries += 1
            assert tries < 500, 'pancake_moveargmax: argmax rarely relocated'
        return torch.stack(cleans), torch.stack(corrupts)
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class PancakeSuffixeditCircuit(PancakeFlip, ContrastCircuit):
    """pancake_suffixedit on pancake_id: contrast vs pancake_whole [value-perturb-control (edit a position after the argmax; flip window fixed -> singleton)] (integrated)."""
    circuit_id = 'pancake_suffixedit'
    algorithm = 'pancake_id'
    base_circuit = 'pancake_whole'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        cleans, corrupts, tries = [], [], 0
        while len(cleans) < batch:
            block = torch.randint(0, 10, (max(4 * (batch - len(cleans)), 512), L), generator=g)
            for row in block:
                m = int(row.max()); k = int((row == m).nonzero()[0])
                if k >= L - 1:
                    continue
                q = int(torch.randint(k + 1, L, (1,), generator=g))
                choices = [d for d in range(m) if d != int(row[q])]
                if not choices:
                    continue
                corrupt = row.clone()
                corrupt[q] = choices[int(torch.randint(0, len(choices), (1,), generator=g))]
                cleans.append(row.clone()); corrupts.append(corrupt)
                if len(cleans) == batch:
                    break
            tries += 1
            assert tries < 500, 'pancake_suffixedit: no valid suffix edit'
        return torch.stack(cleans), torch.stack(corrupts)
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class BitmajWholeCircuit(BitPlaneMajority, DomainCircuit):
    """bitmaj_whole on bitmaj_id: isolated via FULL (integrated)."""
    circuit_id = 'bitmaj_whole'
    algorithm = 'bitmaj_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class BitmajBin01Circuit(BitPlaneMajority, DomainCircuit):
    """bitmaj_bin01 on bitmaj_id: isolated via domain([0,1]) (integrated)."""
    circuit_id = 'bitmaj_bin01'
    algorithm = 'bitmaj_id'
    _cond = conditions.domain([0,1])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class BitmajPlane8Circuit(BitPlaneMajority, DomainCircuit):
    """bitmaj_plane8 on bitmaj_id: isolated via domain([0,8]) (integrated)."""
    circuit_id = 'bitmaj_plane8'
    algorithm = 'bitmaj_id'
    _cond = conditions.domain([0,8])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class BitmajPlaneflipCircuit(BitPlaneMajority, ContrastCircuit):
    """bitmaj_planeflip on bitmaj_id: contrast vs bitmaj_whole [value-perturb] (integrated)."""
    circuit_id = 'bitmaj_planeflip'
    algorithm = 'bitmaj_id'
    base_circuit = 'bitmaj_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            if t >= 16 and t % 4 == 0:
                clean[idx] = torch.randint(0, 10, (idx.numel(), L), generator=g)
                corrupt[idx] = clean[idx]
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                b = int(torch.randint(0, 10, (1,), generator=g))
                if b == int(clean[i, p]): continue
                c = clean[i].clone(); c[p] = b
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class Skipn5WholeCircuit(PredicatedSkip, DomainCircuit):
    """skipn5_whole on skipn5_id: isolated via FULL (integrated)."""
    circuit_id = 'skipn5_whole'
    algorithm = 'skipn5_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Skipn5AlllowCircuit(PredicatedSkip, DomainCircuit):
    """skipn5_alllow on skipn5_id: isolated via domain([0, 1, 2, 3, 4]) (integrated)."""
    circuit_id = 'skipn5_alllow'
    algorithm = 'skipn5_id'
    _cond = conditions.domain([0, 1, 2, 3, 4])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Skipn5AllhighCircuit(PredicatedSkip, DomainCircuit):
    """skipn5_allhigh on skipn5_id: isolated via domain([5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'skipn5_allhigh'
    algorithm = 'skipn5_id'
    _cond = conditions.domain([5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Skipn5ArmflipCircuit(PredicatedSkip, ContrastCircuit):
    """skipn5_armflip on skipn5_id: contrast vs skipn5_whole [threshold-straddle] (integrated)."""
    circuit_id = 'skipn5_armflip'
    algorithm = 'skipn5_id'
    base_circuit = 'skipn5_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        LOW = torch.tensor([0, 1, 2, 3, 4]); HIGH = torch.tensor([5, 6, 7, 8, 9])
        cleans, corrupts = [], []
        while len(cleans) < batch:
            block = torch.randint(0, 10, (max(4 * (batch - len(cleans)), 256), L), generator=g)
            for row in block:
                p = int(torch.randint(0, L, (1,), generator=g))
                lo = int(LOW[torch.randint(0, 5, (1,), generator=g)])
                hi = int(HIGH[torch.randint(0, 5, (1,), generator=g)])
                c = row.clone(); c[p] = lo
                k = row.clone(); k[p] = hi
                if bool((self.eval(c) != self.eval(k)).any()):
                    cleans.append(c); corrupts.append(k)
                    if len(cleans) == batch: break
        return torch.stack(cleans), torch.stack(corrupts)
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class Skipn5WithinCircuit(PredicatedSkip, ContrastCircuit):
    """skipn5_within on skipn5_id: contrast vs skipn5_whole [value-perturb-control] (integrated)."""
    circuit_id = 'skipn5_within'
    algorithm = 'skipn5_id'
    base_circuit = 'skipn5_whole'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        LOW = torch.tensor([0, 1, 2, 3, 4])
        cleans, corrupts = [], []
        while len(cleans) < batch:
            block = torch.randint(0, 10, (max(4 * (batch - len(cleans)), 256), L), generator=g)
            for row in block:
                p = int(torch.randint(0, L, (1,), generator=g))
                a = int(LOW[torch.randint(0, 5, (1,), generator=g)])
                b = int(LOW[torch.randint(0, 5, (1,), generator=g)])
                if a == b: continue
                c = row.clone(); c[p] = a
                k = row.clone(); k[p] = b
                if bool((self.eval(c) != self.eval(k)).any()):
                    cleans.append(c); corrupts.append(k)
                    if len(cleans) == batch: break
        return torch.stack(cleans), torch.stack(corrupts)
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(self.input_length, dtype=torch.bool); m[p] = True
        return m


class BittvWholeCircuit(BitTransitionCount, DomainCircuit):
    """bittv_whole on bittv_id: isolated via FULL (integrated)."""
    circuit_id = 'bittv_whole'
    algorithm = 'bittv_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class BittvFlipeditCircuit(BitTransitionCount, ContrastCircuit):
    """bittv_flipedit on bittv_id: contrast vs bittv_whole [value-perturb] (integrated)."""
    circuit_id = 'bittv_flipedit'
    algorithm = 'bittv_id'
    base_circuit = 'bittv_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for t in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class Vm1WholeCircuit(OpcodeMachine, DomainCircuit):
    """vm1_whole on vm1_id: isolated via FULL (integrated)."""
    circuit_id = 'vm1_whole'
    algorithm = 'vm1_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Vm1AllloadCircuit(OpcodeMachine, DomainCircuit):
    """vm1_allload on vm1_id: isolated via domain([0, 1, 2, 3, 4]) (integrated)."""
    circuit_id = 'vm1_allload'
    algorithm = 'vm1_id'
    _cond = conditions.domain([0, 1, 2, 3, 4])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Vm1NoloadCircuit(OpcodeMachine, DomainCircuit):
    """vm1_noload on vm1_id: isolated via domain([5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'vm1_noload'
    algorithm = 'vm1_id'
    _cond = conditions.domain([5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Vm1OpflipCircuit(OpcodeMachine, ContrastCircuit):
    """vm1_opflip on vm1_id: contrast vs vm1_whole [value-perturb] (integrated)."""
    circuit_id = 'vm1_opflip'
    algorithm = 'vm1_id'
    base_circuit = 'vm1_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for _ in range(512):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            for i in idx.tolist():
                row = clean[i]
                ops = [p for p in range(L - 1) if int(row[p]) >= 5 and int(row[p + 1]) >= 5]
                if not ops:
                    clean[i] = torch.randint(0, 10, (L,), generator=g); continue
                p = ops[int(torch.randint(0, len(ops), (1,), generator=g))]
                new = int(torch.randint(5, 10, (1,), generator=g))
                if new == int(row[p]): continue
                c = row.clone(); c[p] = new
                if not torch.equal(self.eval(row), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class Vm1ClobberedCircuit(OpcodeMachine, ContrastCircuit):
    """vm1_clobbered on vm1_id: contrast vs vm1_whole [value-perturb-control] (integrated)."""
    circuit_id = 'vm1_clobbered'
    algorithm = 'vm1_id'
    base_circuit = 'vm1_whole'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for _ in range(512):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            for i in idx.tolist():
                row = clean[i]
                cand = [p for p in range(L - 1) if int(row[p]) >= 5 and int(row[p + 1]) <= 4]
                if not cand:
                    clean[i] = torch.randint(0, 10, (L,), generator=g); continue
                p = cand[int(torch.randint(0, len(cand), (1,), generator=g))]
                new = int(torch.randint(5, 10, (1,), generator=g))
                if new == int(row[p]): continue
                c = row.clone(); c[p] = new
                if not torch.equal(self.eval(row), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class TourneyWholeCircuit(KnockoutDepth, DomainCircuit):
    """tourney_whole on tourney_id: isolated via FULL (integrated)."""
    circuit_id = 'tourney_whole'
    algorithm = 'tourney_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class TourneyFlipmatchCircuit(KnockoutDepth, ContrastCircuit):
    """tourney_flipmatch on tourney_id: contrast vs tourney_whole [value-perturb] (integrated)."""
    circuit_id = 'tourney_flipmatch'
    algorithm = 'tourney_id'
    base_circuit = 'tourney_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for _ in range(256):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            for i in idx.tolist():
                row = clean[i]; y = self.eval(row)
                for k in range(0, L, 2):
                    a, b = int(row[k]), int(row[k + 1])
                    if a == b: continue
                    lo = k if a < b else k + 1
                    c = row.clone(); c[lo] = 9
                    if not torch.equal(y, self.eval(c)):
                        corrupt[i] = c; need[i] = False; break
                    hi = k + 1 if a < b else k
                    c = row.clone(); c[hi] = 0
                    if not torch.equal(y, self.eval(c)):
                        corrupt[i] = c; need[i] = False; break
                if bool(need[i]):
                    clean[i] = torch.randint(0, 10, (L,), generator=g)
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class EnvmaxWholeCircuit(EnvelopeMax, DomainCircuit):
    """envmax_whole on envmax_id: isolated via FULL (integrated)."""
    circuit_id = 'envmax_whole'
    algorithm = 'envmax_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class EnvmaxMonotoneCircuit(EnvelopeMax, DomainCircuit):
    """envmax_monotone on envmax_id: isolated via monotone(True) (integrated)."""
    circuit_id = 'envmax_monotone'
    algorithm = 'envmax_id'
    _cond = conditions.monotone(True)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class EnvmaxRaiseCircuit(EnvelopeMax, ContrastCircuit):
    """envmax_raise on envmax_id: contrast vs envmax_whole [value-perturb] (integrated)."""
    circuit_id = 'envmax_raise'
    algorithm = 'envmax_id'
    base_circuit = 'envmax_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        cleans, corrupts, stalls = [], [], 0
        while len(cleans) < batch:
            before = len(cleans)
            block = torch.randint(0, 10, (max(4 * (batch - len(cleans)), 512), L), generator=g)
            for row in block:
                p = int(torch.randint(0, L, (1,), generator=g))
                m = int(self.eval(row)[p])
                if m >= 9:
                    continue
                corrupt = row.clone()
                corrupt[p] = int(torch.randint(m + 1, 10, (1,), generator=g))
                if bool((self.eval(row) != self.eval(corrupt)).any()):
                    cleans.append(row.clone()); corrupts.append(corrupt)
                    if len(cleans) == batch:
                        break
            stalls = stalls + 1 if len(cleans) == before else 0
            if stalls >= 20:
                raise CircuitError(f"{self.circuit_id}: raise rarely moves the output — degenerate contrast")
        return torch.stack(cleans), torch.stack(corrupts)
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class CminWholeCircuit(CountMinSketch, DomainCircuit):
    """cmin_whole on cmin_id: isolated via FULL (integrated)."""
    circuit_id = 'cmin_whole'
    algorithm = 'cmin_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CminNocollideCircuit(CountMinSketch, DomainCircuit):
    """cmin_nocollide on cmin_id: isolated via domain([3, 6]) (integrated)."""
    circuit_id = 'cmin_nocollide'
    algorithm = 'cmin_id'
    _cond = conditions.domain([3, 6])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CminBumpeditCircuit(CountMinSketch, ContrastCircuit):
    """cmin_bumpedit on cmin_id: contrast vs cmin_whole [value-perturb] (integrated)."""
    circuit_id = 'cmin_bumpedit'
    algorithm = 'cmin_id'
    base_circuit = 'cmin_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for _ in range(128):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class BittransWholeCircuit(BitTransposeGather, DomainCircuit):
    """bittrans_whole on bittrans_id: isolated via FULL (integrated)."""
    circuit_id = 'bittrans_whole'
    algorithm = 'bittrans_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class BittransBin01Circuit(BitTransposeGather, DomainCircuit):
    """bittrans_bin01 on bittrans_id: isolated via domain([0,1]) (integrated)."""
    circuit_id = 'bittrans_bin01'
    algorithm = 'bittrans_id'
    _cond = conditions.domain([0,1])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class BittransDigiteditCircuit(BitTransposeGather, ContrastCircuit):
    """bittrans_digitedit on bittrans_id: contrast vs bittrans_whole [value-perturb] (integrated)."""
    circuit_id = 'bittrans_digitedit'
    algorithm = 'bittrans_id'
    base_circuit = 'bittrans_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for _ in range(256):
            idx = need.nonzero().flatten()
            if not idx.numel(): break
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                c = clean[i].clone(); c[p] = int(torch.randint(0, 10, (1,), generator=g))
                if int(c[p]) == int(clean[i, p]): continue
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class BittransBiteditCircuit(BitTransposeGather, ContrastCircuit):
    """bittrans_bitedit on bittrans_id: contrast vs bittrans_whole [value-perturb-control] (integrated)."""
    circuit_id = 'bittrans_bitedit'
    algorithm = 'bittrans_id'
    base_circuit = 'bittrans_whole'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = DomainCircuit.sample(self, batch, seed)
        corrupt = clean.clone()
        for i in range(batch):
            while True:
                p = int(torch.randint(0, L, (1,), generator=g))
                b = int(torch.randint(0, 4, (1,), generator=g))
                v = int(clean[i, p]) ^ (1 << b)
                if 0 <= v <= 9:
                    corrupt[i, p] = v; break
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class InternWholeCircuit(InternRank, DomainCircuit):
    """intern_whole on intern_id: isolated via FULL (integrated)."""
    circuit_id = 'intern_whole'
    algorithm = 'intern_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class InternRepeatEditCircuit(InternRank, ContrastCircuit):
    """intern_repeat_edit on intern_id: contrast vs intern_whole [value-perturb-control] (integrated)."""
    circuit_id = 'intern_repeat_edit'
    algorithm = 'intern_id'
    base_circuit = 'intern_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.empty(batch, L, dtype=torch.long)
        corrupt = torch.empty(batch, L, dtype=torch.long)
        for i in range(batch):
            done = False
            for _ in range(256):
                row = torch.randint(0, 10, (L,), generator=g)
                vals = row.tolist()
                cands = []
                for p in range(L):
                    seen = set(vals[:p])
                    if vals[p] in seen:
                        alts = [w for w in seen if w != vals[p]]
                        if alts:
                            cands.append((p, alts))
                if not cands:
                    continue
                p, alts = cands[int(torch.randint(0, len(cands), (1,), generator=g))]
                w = alts[int(torch.randint(0, len(alts), (1,), generator=g))]
                c = row.clone(); c[p] = w
                clean[i] = row; corrupt[i] = c
                done = True
                break
            assert done, 'intern_repeat_edit: no repeat-with-alternative row found'
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class InternFirsteditCircuit(InternRank, ContrastCircuit):
    """intern_firstedit on intern_id: contrast vs intern_whole [value-perturb] (integrated)."""
    circuit_id = 'intern_firstedit'
    algorithm = 'intern_id'
    base_circuit = 'intern_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.empty(batch, L, dtype=torch.long)
        corrupt = torch.empty(batch, L, dtype=torch.long)
        for i in range(batch):
            done = False
            for _ in range(256):
                row = torch.randint(0, 10, (L,), generator=g)
                vals = row.tolist()
                firsts = [p for p in range(1, L) if vals[p] not in vals[:p]]
                if not firsts:
                    continue
                p = firsts[int(torch.randint(0, len(firsts), (1,), generator=g))]
                seen = list(set(vals[:p]))
                w = seen[int(torch.randint(0, len(seen), (1,), generator=g))]
                c = row.clone(); c[p] = w
                clean[i] = row; corrupt[i] = c
                done = True
                break
            assert done, 'intern_firstedit: no first-occurrence position >=1 found'
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class DupxorWholeCircuit(MultiplicityParityXor, DomainCircuit):
    """dupxor_whole on dupxor_id: isolated via FULL (integrated)."""
    circuit_id = 'dupxor_whole'
    algorithm = 'dupxor_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DupxorAlldistinctCircuit(MultiplicityParityXor, DomainCircuit):
    """dupxor_alldistinct on dupxor_id: isolated via Condition('all_distinct', lambda x: len(set(x.tolist())) == 10, lambda batch, seed: torch.argsort(torch.rand(batch, 10, generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'dupxor_alldistinct'))), dim=1)) (integrated)."""
    circuit_id = 'dupxor_alldistinct'
    algorithm = 'dupxor_id'
    _cond = conditions.Condition('all_distinct', lambda x: len(set(x.tolist())) == 10, lambda batch, seed: torch.argsort(torch.rand(batch, 10, generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'dupxor_alldistinct'))), dim=1))
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DupxorValeditCircuit(MultiplicityParityXor, ContrastCircuit):
    """dupxor_valedit on dupxor_id: contrast vs dupxor_whole [value-perturb] (integrated)."""
    circuit_id = 'dupxor_valedit'
    algorithm = 'dupxor_id'
    base_circuit = 'dupxor_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        clean = DomainCircuit.sample(self, batch, seed)
        g = torch.Generator().manual_seed(int(seed) + 1)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        rows = torch.arange(batch)
        corrupt = clean.clone()
        delta = torch.randint(1, 10, (batch,), generator=g)
        corrupt[rows, p] = (clean[rows, p] + delta) % 10
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class NbrdegWholeCircuit(ProximityDegree, DomainCircuit):
    """nbrdeg_whole on nbrdeg_id: isolated via FULL (integrated)."""
    circuit_id = 'nbrdeg_whole'
    algorithm = 'nbrdeg_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Nbrdeg2clusterCircuit(ProximityDegree, DomainCircuit):
    """nbrdeg_2cluster on nbrdeg_id: isolated via domain([0, 1, 8, 9]) (integrated)."""
    circuit_id = 'nbrdeg_2cluster'
    algorithm = 'nbrdeg_id'
    _cond = conditions.domain([0, 1, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class NbrdegValeditCircuit(ProximityDegree, ContrastCircuit):
    """nbrdeg_valedit on nbrdeg_id: contrast vs nbrdeg_whole [value-perturb] (integrated)."""
    circuit_id = 'nbrdeg_valedit'
    algorithm = 'nbrdeg_id'
    base_circuit = 'nbrdeg_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        need = torch.ones(batch, dtype=torch.bool)
        for _ in range(256):
            idx = need.nonzero().flatten()
            if not idx.numel():
                break
            for i in idx.tolist():
                p = int(torch.randint(0, L, (1,), generator=g))
                w = int(torch.randint(0, 10, (1,), generator=g))
                if w == int(clean[i, p]):
                    continue
                c = clean[i].clone(); c[p] = w
                if not torch.equal(self.eval(clean[i]), self.eval(c)):
                    corrupt[i] = c; need[i] = False
        assert not int(need.any())
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class CfreqWholeCircuit(CumFreqRank, DomainCircuit):
    """cfreq_whole on cfreq_id: isolated via FULL (integrated)."""
    circuit_id = 'cfreq_whole'
    algorithm = 'cfreq_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CfreqTwovaluesCircuit(CumFreqRank, DomainCircuit):
    """cfreq_twovalues on cfreq_id: isolated via TWO_VALUES (integrated)."""
    circuit_id = 'cfreq_twovalues'
    algorithm = 'cfreq_id'
    _cond = conditions.TWO_VALUES
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CfreqCounts4321Circuit(CumFreqRank, DomainCircuit):
    """cfreq_counts4321 on cfreq_id: isolated via count_profile([4, 3, 2, 1]) (integrated)."""
    circuit_id = 'cfreq_counts4321'
    algorithm = 'cfreq_id'
    _cond = conditions.count_profile([4, 3, 2, 1])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CfreqOnerepeatCircuit(CumFreqRank, DomainCircuit):
    """cfreq_onerepeat on cfreq_id: isolated via ONE_REPEAT (integrated)."""
    circuit_id = 'cfreq_onerepeat'
    algorithm = 'cfreq_id'
    _cond = conditions.ONE_REPEAT
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CfreqTipmodeCircuit(CumFreqRank, ContrastCircuit):
    """cfreq_tipmode on cfreq_id: contrast vs cfreq_whole [count-class promotion (singleton -> mode block)] (integrated)."""
    circuit_id = 'cfreq_tipmode'
    algorithm = 'cfreq_id'
    base_circuit = 'cfreq_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        cleans, corrupts = [], []
        for _ in range(64):
            block = torch.randint(0, 10, (2 * (batch - len(cleans)) + 512, L), generator=g)
            for row in block:
                vals = row.tolist()
                cnt = {v: vals.count(v) for v in set(vals)}
                ranked = sorted(cnt.values(), reverse=True)
                if len(cnt) < 2 or ranked[0] < 3 or ranked[0] == ranked[1]:
                    continue
                singles = [i for i in range(L) if cnt[vals[i]] == 1]
                if not singles:
                    continue
                m = max(cnt, key=lambda v: cnt[v])
                p = singles[int(torch.randint(0, len(singles), (1,), generator=g))]
                corrupt = row.clone(); corrupt[p] = m
                cleans.append(row); corrupts.append(corrupt)
                if len(cleans) == batch:
                    return torch.stack(cleans), torch.stack(corrupts)
        raise CircuitError("cfreq_tipmode: too few rows with a strict mode (count >= 3) and a singleton")
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class DsubFullCircuit(DistinctSubseqCount, DomainCircuit):
    """dsub_full on dsub_id: isolated via FULL (integrated)."""
    circuit_id = 'dsub_full'
    algorithm = 'dsub_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DsubOnerepeatCircuit(DistinctSubseqCount, DomainCircuit):
    """dsub_onerepeat on dsub_id: isolated via ONE_REPEAT (integrated)."""
    circuit_id = 'dsub_onerepeat'
    algorithm = 'dsub_id'
    _cond = conditions.ONE_REPEAT
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DsubAdjrepeatsCircuit(DistinctSubseqCount, DomainCircuit):
    """dsub_adjrepeats on dsub_id: isolated via REPEATS_ADJACENT (integrated)."""
    circuit_id = 'dsub_adjrepeats'
    algorithm = 'dsub_id'
    _cond = conditions.REPEATS_ADJACENT
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DsubNorepeatadjCircuit(DistinctSubseqCount, DomainCircuit):
    """dsub_norepeatadj on dsub_id: isolated via NO_ADJACENT_REPEAT (integrated)."""
    circuit_id = 'dsub_norepeatadj'
    algorithm = 'dsub_id'
    _cond = conditions.NO_ADJACENT_REPEAT
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DsubRepeatbreakCircuit(DistinctSubseqCount, ContrastCircuit):
    """dsub_repeatbreak on dsub_id: contrast vs dsub_full [value-perturb (fresh value <-> repeat of an earlier value)] (integrated)."""
    circuit_id = 'dsub_repeatbreak'
    algorithm = 'dsub_id'
    base_circuit = 'dsub_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        cleans, corrupts = [], []
        for _ in range(64):
            block = torch.randint(0, 10, (2 * (batch - len(cleans)) + 512, L), generator=g)
            for row in block:
                vals = row.tolist()
                p = 1 + int(torch.randint(0, L - 1, (1,), generator=g))
                earlier = sorted(set(vals[:p]))
                fresh = [d for d in range(10) if d not in vals[:p] and d not in vals[p + 1:]]
                clean = row.clone(); clean[p] = fresh[int(torch.randint(0, len(fresh), (1,), generator=g))]
                corrupt = row.clone(); corrupt[p] = earlier[int(torch.randint(0, len(earlier), (1,), generator=g))]
                if torch.equal(self.eval(clean), self.eval(corrupt)):
                    continue
                cleans.append(clean); corrupts.append(corrupt)
                if len(cleans) == batch:
                    return torch.stack(cleans), torch.stack(corrupts)
        raise CircuitError("dsub_repeatbreak: the repeat/fresh swap never moved the output")
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class DsubDbloddWholeCircuit(DoubleOdd, DomainCircuit):
    """dsub_dblodd_whole on dblodd_dsub_id: isolated via FULL (integrated)."""
    circuit_id = 'dsub_dblodd_whole'
    algorithm = 'dblodd_dsub_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CodonskipInframeCircuit(CodonSkip, DomainCircuit):
    """codonskip_inframe on codonskip_id: isolated via GAPS_IN_FRAME (integrated)."""
    circuit_id = 'codonskip_inframe'
    algorithm = 'codonskip_id'
    _cond = conditions.GAPS_IN_FRAME
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CodonskipOnegapCircuit(CodonSkip, DomainCircuit):
    """codonskip_onegap on codonskip_id: isolated via at_most_k_of(0, 1) (integrated)."""
    circuit_id = 'codonskip_onegap'
    algorithm = 'codonskip_id'
    _cond = conditions.at_most_k_of(0, 1)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CodonskipBinaryCircuit(CodonSkip, DomainCircuit):
    """codonskip_binary on codonskip_id: isolated via BINARY (integrated)."""
    circuit_id = 'codonskip_binary'
    algorithm = 'codonskip_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class CodonskipIndelCircuit(CodonSkip, ContrastCircuit):
    """codonskip_indel on codonskip_id: contrast vs codonskip_inframe [digit-toggle] (integrated)."""
    circuit_id = 'codonskip_indel'
    algorithm = 'codonskip_id'
    base_circuit = 'codonskip_inframe'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        interior = torch.arange(1, L - 1)
        p = torch.zeros(batch, dtype=torch.long)
        for i in range(batch):
            elig = interior[clean[i, 1:L - 1] != 0]
            while not elig.numel():                      # an all-gap interior has no base to delete
                clean[i] = torch.randint(0, 10, (L,), generator=g)
                elig = interior[clean[i, 1:L - 1] != 0]
            p[i] = elig[torch.randint(0, elig.numel(), (1,), generator=g)]
        corrupt = clean.clone()
        corrupt[torch.arange(batch), p] = 0
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        m = clean_row != 0                               # gaps print 0 on both sides
        m[:int((clean_row != corrupt_row).nonzero()[0])] = False
        return m


class CodonskipInframe3Circuit(CodonSkip, ContrastCircuit):
    """codonskip_inframe3 on codonskip_id: contrast vs codonskip_inframe [compensated-three-site] (integrated)."""
    circuit_id = 'codonskip_inframe3'
    algorithm = 'codonskip_id'
    base_circuit = 'codonskip_inframe'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        interior = torch.arange(1, L - 1)
        corrupt = clean.clone()
        for i in range(batch):
            elig = interior[clean[i, 1:L - 1] != 0]
            while elig.numel() < 3:                      # need three bases to delete in frame
                clean[i] = torch.randint(0, 10, (L,), generator=g)
                elig = interior[clean[i, 1:L - 1] != 0]
            pick = elig[torch.randperm(elig.numel(), generator=g)[:3]]
            corrupt[i] = clean[i]
            corrupt[i, pick] = 0
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        cut = (clean_row != corrupt_row).nonzero().flatten()
        m = clean_row != 0                               # the frame returns after the third deletion,
        m[:int(cut[0])] = False                          # so nothing outside [p, r] can move
        m[int(cut[-1]) + 1:] = False
        return m


class FraglenOnecutCircuit(FragLen, DomainCircuit):
    """fraglen_onecut on fraglen_id: isolated via ONE_CUT (integrated)."""
    circuit_id = 'fraglen_onecut'
    algorithm = 'fraglen_id'
    _cond = conditions.ONE_CUT
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class FraglenTwocutsCircuit(FragLen, DomainCircuit):
    """fraglen_twocuts on fraglen_id: isolated via TWO_CUTS (integrated)."""
    circuit_id = 'fraglen_twocuts'
    algorithm = 'fraglen_id'
    _cond = conditions.TWO_CUTS
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class FraglenCutgainCircuit(FragLen, ContrastCircuit):
    """fraglen_cutgain on fraglen_id: contrast vs fraglen_onecut [threshold-straddle] (integrated)."""
    circuit_id = 'fraglen_cutgain'
    algorithm = 'fraglen_id'
    base_circuit = 'fraglen_onecut'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        sites = torch.arange(1, L)
        corrupt = clean.clone()
        for i in range(batch):
            elig = sites[clean[i, 1:] < 5]
            while not elig.numel():                      # every interior digit is already a cut site
                clean[i] = torch.randint(0, 10, (L,), generator=g)
                elig = sites[clean[i, 1:] < 5]
            p = int(elig[torch.randint(0, elig.numel(), (1,), generator=g)])
            corrupt[i] = clean[i]
            corrupt[i, p] = int(torch.randint(5, 10, (1,), generator=g))
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        L = self.input_length
        p = int((clean_row != corrupt_row).nonzero()[0])   # the clean piece [a, b) splits at p
        a = max([j for j in range(1, p) if int(clean_row[j]) >= 5] + [0])
        b = next((j for j in range(p + 1, L) if int(clean_row[j]) >= 5), L)
        m = torch.zeros(L, dtype=torch.bool)
        m[a:p], m[p:b] = min(b - a, 9) != min(p - a, 9), min(b - a, 9) != min(b - p, 9)
        return m


class FraglenCutlossCircuit(FragLen, ContrastCircuit):
    """fraglen_cutloss on fraglen_id: contrast vs fraglen_onecut [threshold-straddle] (integrated)."""
    circuit_id = 'fraglen_cutloss'
    algorithm = 'fraglen_id'
    base_circuit = 'fraglen_onecut'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        clean = torch.randint(0, 10, (batch, L), generator=g)
        sites = torch.arange(1, L)
        corrupt = clean.clone()
        for i in range(batch):
            elig = sites[clean[i, 1:] >= 5]
            while not elig.numel():                      # no interior cut site to destroy
                clean[i] = torch.randint(0, 10, (L,), generator=g)
                elig = sites[clean[i, 1:] >= 5]
            p = int(elig[torch.randint(0, elig.numel(), (1,), generator=g)])
            corrupt[i] = clean[i]
            corrupt[i, p] = int(torch.randint(0, 5, (1,), generator=g))
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        L = self.input_length
        p = int((clean_row != corrupt_row).nonzero()[0])   # [a, p) and [p, b) merge into [a, b)
        a = max([j for j in range(1, p) if int(clean_row[j]) >= 5] + [0])
        b = next((j for j in range(p + 1, L) if int(clean_row[j]) >= 5), L)
        m = torch.zeros(L, dtype=torch.bool)
        m[a:p], m[p:b] = min(p - a, 9) != min(b - a, 9), min(b - p, 9) != min(b - a, 9)
        return m


class DivzetaWholeCircuit(DivisorZeta, DomainCircuit):
    """divzeta_whole on divzeta_id: isolated via FULL (integrated)."""
    circuit_id = 'divzeta_whole'
    algorithm = 'divzeta_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DivzetaChainCircuit(DivisorZeta, DomainCircuit):
    """divzeta_chain on divzeta_id: isolated via domain([1, 2, 4, 8], name="div_chain") (integrated)."""
    circuit_id = 'divzeta_chain'
    algorithm = 'divzeta_id'
    _cond = conditions.domain([1, 2, 4, 8], name="div_chain")
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DivzetaAntichainCircuit(DivisorZeta, DomainCircuit):
    """divzeta_antichain on divzeta_id: isolated via domain([2, 3, 5, 7], name="div_antichain") (integrated)."""
    circuit_id = 'divzeta_antichain'
    algorithm = 'divzeta_id'
    _cond = conditions.domain([2, 3, 5, 7], name="div_antichain")
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DivzetaB2Circuit(DivisorZeta, DomainCircuit):
    """divzeta_b2 on divzeta_id: isolated via domain([1, 2, 3, 6], name="div_b2") (integrated)."""
    circuit_id = 'divzeta_b2'
    algorithm = 'divzeta_id'
    _cond = conditions.domain([1, 2, 3, 6], name="div_b2")
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DivzetaTop09Circuit(DivisorZeta, ContrastCircuit):
    """divzeta_top09 on divzeta_id: contrast vs divzeta_whole [digit-toggle] (integrated)."""
    circuit_id = 'divzeta_top09'
    algorithm = 'divzeta_id'
    base_circuit = 'divzeta_whole'
    local_predicate = True
    _cond = conditions.pin(5, [0])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        keep, n = [], 0
        dead = torch.tensor([1, 3, 9])
        while n < batch:
            block = torch.randint(0, 10, (max(4 * (batch - n), 512), self.input_length), generator=g)
            block[:, 5] = 0
            good = block[(~torch.isin(block[:, :5], dead)).any(1)]
            keep.append(good); n += int(good.shape[0])
        clean = torch.cat(keep)[:batch]
        corrupt = clean.clone(); corrupt[:, 5] = 9
        return clean, corrupt


class DivzetaBot17Circuit(DivisorZeta, ContrastCircuit):
    """divzeta_bot17 on divzeta_id: contrast vs divzeta_whole [digit-toggle] (integrated)."""
    circuit_id = 'divzeta_bot17'
    algorithm = 'divzeta_id'
    base_circuit = 'divzeta_whole'
    local_predicate = True
    _cond = conditions.pin(2, [1])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        keep, n = [], 0
        dead = torch.tensor([0, 7])
        while n < batch:
            block = torch.randint(0, 10, (max(4 * (batch - n), 512), self.input_length), generator=g)
            block[:, 2] = 1
            good = block[(~torch.isin(block[:, 3:], dead)).any(1)]
            keep.append(good); n += int(good.shape[0])
        clean = torch.cat(keep)[:batch]
        corrupt = clean.clone(); corrupt[:, 2] = 7
        return clean, corrupt


class MobinvWholeCircuit(MobiusInvert, DomainCircuit):
    """mobinv_whole on mobinv_id: isolated via FULL (integrated)."""
    circuit_id = 'mobinv_whole'
    algorithm = 'mobinv_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class MobinvFlatCircuit(MobiusInvert, DomainCircuit):
    """mobinv_flat on mobinv_id: isolated via ALL_EQUAL (integrated)."""
    circuit_id = 'mobinv_flat'
    algorithm = 'mobinv_id'
    _cond = conditions.ALL_EQUAL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class MobinvGf2Circuit(MobiusInvert, DomainCircuit):
    """mobinv_gf2 on mobinv_id: isolated via domain([0, 5], name="gf2_flat05") (integrated)."""
    circuit_id = 'mobinv_gf2'
    algorithm = 'mobinv_id'
    _cond = conditions.domain([0, 5], name="gf2_flat05")
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class MobinvZimgCircuit(MobiusInvert, DomainCircuit):
    """mobinv_zimg on mobinv_id: isolated via ZETA_IMAGE (integrated)."""
    circuit_id = 'mobinv_zimg'
    algorithm = 'mobinv_id'
    _cond = conditions.ZETA_IMAGE
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class MobinvP7Circuit(MobiusInvert, ContrastCircuit):
    """mobinv_p7 on mobinv_id: contrast vs mobinv_whole [digit-toggle] (integrated)."""
    circuit_id = 'mobinv_p7'
    algorithm = 'mobinv_id'
    base_circuit = 'mobinv_whole'
    local_predicate = True
    _cond = conditions.pin(7, [0])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        clean[:, 7] = 0
        corrupt = clean.clone(); corrupt[:, 7] = 9
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        m = torch.zeros(10, dtype=torch.bool); m[[7]] = True
        return m


class MobinvP1Circuit(MobiusInvert, ContrastCircuit):
    """mobinv_p1 on mobinv_id: contrast vs mobinv_whole [digit-toggle] (integrated)."""
    circuit_id = 'mobinv_p1'
    algorithm = 'mobinv_id'
    base_circuit = 'mobinv_whole'
    local_predicate = True
    _cond = conditions.pin(1, [0])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        clean[:, 1] = 0
        corrupt = clean.clone(); corrupt[:, 1] = 9
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        m = torch.zeros(10, dtype=torch.bool); m[[1, 2, 3, 5, 6, 7]] = True
        return m


class FactnormFullCircuit(FactoradicNormalize, DomainCircuit):
    """factnorm_full on factnorm_id: isolated via FULL (integrated)."""
    circuit_id = 'factnorm_full'
    algorithm = 'factnorm_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class FactnormCanonicalCircuit(FactoradicNormalize, DomainCircuit):
    """factnorm_canonical on factnorm_id: isolated via FACTORADIC_CANONICAL (integrated)."""
    circuit_id = 'factnorm_canonical'
    algorithm = 'factnorm_id'
    _cond = conditions.FACTORADIC_CANONICAL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return Id(self.input_length).eval(input_seq)


class FactnormCarryheadCircuit(FactoradicNormalize, DomainCircuit):
    """factnorm_carryhead on factnorm_id: isolated via CARRY_SOURCE_ONLY_FIRST (integrated)."""
    circuit_id = 'factnorm_carryhead'
    algorithm = 'factnorm_id'
    _cond = conditions.CARRY_SOURCE_ONLY_FIRST
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class FactnormLastflipCircuit(FactoradicNormalize, ContrastCircuit):
    """factnorm_lastflip on factnorm_id: contrast vs factnorm_full [digit-toggle] (integrated)."""
    circuit_id = 'factnorm_lastflip'
    algorithm = 'factnorm_id'
    base_circuit = 'factnorm_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        clean = DomainCircuit.sample(self, batch, seed)
        clean[:, -1] = 0
        corrupt = clean.clone()
        corrupt[:, -1] = 9
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        m = torch.zeros(self.input_length, dtype=torch.bool)
        m[-1] = True
        return m


class FactnormCarryflipCircuit(FactoradicNormalize, ContrastCircuit):
    """factnorm_carryflip on factnorm_id: contrast vs factnorm_full [digit-toggle] (integrated)."""
    circuit_id = 'factnorm_carryflip'
    algorithm = 'factnorm_id'
    base_circuit = 'factnorm_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        clean = DomainCircuit.sample(self, batch, seed)
        clean[:, 0] = 0
        corrupt = clean.clone()
        corrupt[:, 0] = 9
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class Nonadj4FullCircuit(NonAdjacentMax4, DomainCircuit):
    """nonadj4_full on nonadj4_id: isolated via FULL (integrated)."""
    circuit_id = 'nonadj4_full'
    algorithm = 'nonadj4_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Nonadj4LanesCircuit(NonAdjacentMax4, DomainCircuit):
    """nonadj4_lanes on nonadj4_id: isolated via LO_EVEN_HI_ODD (integrated)."""
    circuit_id = 'nonadj4_lanes'
    algorithm = 'nonadj4_id'
    _cond = conditions.LO_EVEN_HI_ODD
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Nonadj4Sumle9Circuit(NonAdjacentMax4, DomainCircuit):
    """nonadj4_sumle9 on nonadj4_id: isolated via sum_at_most(9) (integrated)."""
    circuit_id = 'nonadj4_sumle9'
    algorithm = 'nonadj4_id'
    _cond = conditions.sum_at_most(9)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Nonadj4SwingCircuit(NonAdjacentMax4, ContrastCircuit):
    """nonadj4_swing on nonadj4_id: contrast vs nonadj4_full [value-perturb] (integrated)."""
    circuit_id = 'nonadj4_swing'
    algorithm = 'nonadj4_id'
    base_circuit = 'nonadj4_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        def edit(row, g):
            p = int(torch.randint(0, self.input_length, (1,), generator=g))
            corrupt = row.clone()
            corrupt[p] = 9 if int(row[p]) <= 4 else 0
            return corrupt
        return _rejection_pairs(self, batch, seed, edit)
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class Nonadj4TailCircuit(NonAdjacentMax4, ContrastCircuit):
    """nonadj4_tail on nonadj4_id: contrast vs nonadj4_full [value-perturb] (integrated)."""
    circuit_id = 'nonadj4_tail'
    algorithm = 'nonadj4_id'
    base_circuit = 'nonadj4_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        def edit(row, g):
            corrupt = row.clone()
            corrupt[-1] = 9 if int(row[-1]) <= 4 else 0
            return corrupt
        return _rejection_pairs(self, batch, seed, edit)
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        m = torch.zeros(self.input_length, dtype=torch.bool)
        m[-1] = True
        return m


class HkconfRangeLe2Circuit(BoundedConfidenceMean, DomainCircuit):
    """hkconf_range_le2 on hkconf_id: isolated via range_at_most(2) (integrated)."""
    circuit_id = 'hkconf_range_le2'
    algorithm = 'hkconf_id'
    _cond = conditions.range_at_most(2)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return torch.full_like(input_seq, int((2 * input_seq.sum() + self.input_length) // (2 * self.input_length)))


class HkconfSep3ValuesCircuit(BoundedConfidenceMean, DomainCircuit):
    """hkconf_sep3_values on hkconf_id: isolated via separated_values(3) (integrated)."""
    circuit_id = 'hkconf_sep3_values'
    algorithm = 'hkconf_id'
    _cond = conditions.separated_values(3)
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return Id(self.input_length).eval(input_seq)


class HkconfBimodalGapCircuit(BoundedConfidenceMean, DomainCircuit):
    """hkconf_bimodal_gap on hkconf_id: isolated via BIMODAL_GAP (integrated)."""
    circuit_id = 'hkconf_bimodal_gap'
    algorithm = 'hkconf_id'
    _cond = conditions.BIMODAL_GAP
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return (lambda lo: torch.where(lo, (2 * input_seq[lo].sum() + int(lo.sum())) // (2 * int(lo.sum())), (2 * input_seq[~lo].sum() + int((~lo).sum())) // (2 * int((~lo).sum()))))(input_seq - input_seq.min() <= 2)


class HkconfFlip09Circuit(BoundedConfidenceMean, ContrastCircuit):
    """hkconf_flip09 on hkconf_id: contrast vs hkconf_bimodal_gap [digit-toggle] (integrated)."""
    circuit_id = 'hkconf_flip09'
    algorithm = 'hkconf_id'
    base_circuit = 'hkconf_bimodal_gap'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        p, rows = torch.randint(0, self.input_length, (batch,), generator=g), torch.arange(batch)
        clean[rows, p] = 0                       # p's own mean is over {0,1,2} clean, {7,8,9} corrupt
        corrupt = clean.clone()
        corrupt[rows, p] = 9
        return clean, corrupt


class HkconfLinkCircuit(BoundedConfidenceMean, ContrastCircuit):
    """hkconf_link on hkconf_id: contrast vs hkconf_bimodal_gap [threshold-straddle] (integrated)."""
    circuit_id = 'hkconf_link'
    algorithm = 'hkconf_id'
    base_circuit = 'hkconf_bimodal_gap'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = torch.randint(0, 2, (batch, self.input_length), generator=g)
        p, rows = torch.randint(0, self.input_length, (batch,), generator=g), torch.arange(batch)
        clean[rows, p] = 3                       # 3 is linked to the {0,1} background, 4 is not
        corrupt = clean.clone()
        corrupt[rows, p] = 4
        return clean, corrupt


class BaksnepNoExtremeTieCircuit(ExtremalReplace, DomainCircuit):
    """baksnep_no_extreme_tie on baksnep_id: isolated via NO_EXTREME_TIE (integrated)."""
    circuit_id = 'baksnep_no_extreme_tie'
    algorithm = 'baksnep_id'
    _cond = conditions.NO_EXTREME_TIE
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return torch.where((torch.arange(self.input_length, device=input_seq.device) - int(input_seq.argmin())).abs() <= 1, input_seq.max(), input_seq)


class BaksnepMinFirstCircuit(ExtremalReplace, DomainCircuit):
    """baksnep_min_first on baksnep_id: isolated via MIN_FIRST (integrated)."""
    circuit_id = 'baksnep_min_first'
    algorithm = 'baksnep_id'
    _cond = conditions.MIN_FIRST
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return torch.cat([input_seq.max().repeat(2), input_seq[2:]])


class BaksnepMaxFirstCircuit(ExtremalReplace, DomainCircuit):
    """baksnep_max_first on baksnep_id: isolated via MAX_FIRST (integrated)."""
    circuit_id = 'baksnep_max_first'
    algorithm = 'baksnep_id'
    _cond = conditions.MAX_FIRST
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return torch.where((torch.arange(self.input_length, device=input_seq.device) - int((input_seq == input_seq.min()).nonzero()[0])).abs() <= 1, input_seq[0], input_seq)


class BaksnepMoveminCircuit(ExtremalReplace, ContrastCircuit):
    """baksnep_movemin on baksnep_id: contrast vs baksnep_no_extreme_tie [position-swap] (integrated)."""
    circuit_id = 'baksnep_movemin'
    algorithm = 'baksnep_id'
    base_circuit = 'baksnep_no_extreme_tie'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L, cleans, corrupts, tries = self.input_length, [], [], 0
        cap = batch * REJECTION_BUDGET_PER_ROW
        while len(cleans) < batch:
            tries += 1
            p, q = torch.randperm(L, generator=g)[:2].tolist()
            if abs(p - q) < 3:                                            # disjoint write windows
                continue
            m = int(torch.randint(0, 9, (1,), generator=g))
            clean = m + 1 + (torch.rand((L,), generator=g) * (9 - m)).long()
            clean[p] = m                                                  # the unique minimum
            corrupt = clean.clone()
            corrupt[p], corrupt[q] = clean[q], clean[p]                   # relocate it to q
            if bool((self.eval(clean) != self.eval(corrupt)).any()):
                cleans.append(clean); corrupts.append(corrupt)
            if tries > cap:
                raise CircuitError(f"{self.circuit_id}: relocation accept-rate too low "
                                   f"({len(cleans)}/{batch} after {tries} tries)")
        return torch.stack(cleans), torch.stack(corrupts)


class BaksnepBumpmaxCircuit(ExtremalReplace, ContrastCircuit):
    """baksnep_bumpmax on baksnep_id: contrast vs baksnep_no_extreme_tie [off-window value edit] (integrated)."""
    circuit_id = 'baksnep_bumpmax'
    algorithm = 'baksnep_id'
    base_circuit = 'baksnep_no_extreme_tie'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L, cleans, corrupts = self.input_length, [], []
        while len(cleans) < batch:
            i, a = torch.randperm(L, generator=g)[:2].tolist()
            if abs(i - a) < 3:                                            # argmax off the write window
                continue
            m = int(torch.randint(0, 6, (1,), generator=g))
            clean = m + 1 + torch.randint(0, 6 - m, (L,), generator=g)    # interior digits in [m+1, 6]
            clean[i], clean[a] = m, 7                                     # unique min, unique max
            corrupt = clean.clone()
            corrupt[a] = 9                                                # the written value 7 -> 9
            cleans.append(clean); corrupts.append(corrupt)
        return torch.stack(cleans), torch.stack(corrupts)
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        i = int((clean_row == clean_row.min()).nonzero()[0])
        a = int((clean_row != corrupt_row).nonzero()[0])
        return _mask(self.input_length, [j for j in range(self.input_length) if abs(j - i) <= 1 or j == a])


class HsifteFullCircuit(Hsifte, DomainCircuit):
    """hsifte_full on hsifte_id: isolated via FULL (integrated)."""
    circuit_id = 'hsifte_full'
    algorithm = 'hsifte_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class HsifteHeapordCircuit(Hsifte, DomainCircuit):
    """hsifte_heapord on hsifte_id: isolated via HEAP_ORDERED (integrated)."""
    circuit_id = 'hsifte_heapord'
    algorithm = 'hsifte_id'
    _cond = conditions.HEAP_ORDERED
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return Id(self.input_length).eval(input_seq)


class HsifteAntiheapCircuit(Hsifte, DomainCircuit):
    """hsifte_antiheap on hsifte_id: isolated via ANTIHEAP_LEFTMAX (integrated)."""
    circuit_id = 'hsifte_antiheap'
    algorithm = 'hsifte_id'
    _cond = conditions.ANTIHEAP_LEFTMAX
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return Id(self.input_length).eval(input_seq)[torch.tensor([1, 0, 2, 7, 9, 5, 6, 3, 8, 4])]


class HsifteLiftlCircuit(Hsifte, ContrastCircuit):
    """hsifte_liftl on hsifte_id: contrast vs hsifte_full [threshold-straddle] (integrated)."""
    circuit_id = 'hsifte_liftl'
    algorithm = 'hsifte_id'
    base_circuit = 'hsifte_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        quiet = torch.tensor([(a, b, c) for a in range(9) for b in range(a + 1) for c in range(a + 1)])
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        clean[:, :3] = quiet[torch.randint(0, quiet.shape[0], (batch,), generator=g)]
        u = torch.rand((batch,), generator=g)
        corrupt = clean.clone()
        corrupt[:, 1] = clean[:, 0] + 1 + (u * (9 - clean[:, 0])).long()
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        m = torch.zeros(self.output_length(), dtype=torch.bool)
        m[0] = True
        m[1] = bool(clean_row[0] != clean_row[1])
        return m


class HsifteChildduelCircuit(Hsifte, ContrastCircuit):
    """hsifte_childduel on hsifte_id: contrast vs hsifte_full [threshold-straddle] (integrated)."""
    circuit_id = 'hsifte_childduel'
    algorithm = 'hsifte_id'
    base_circuit = 'hsifte_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        duel = torch.tensor([(a, b, c) for b in range(1, 9) for c in range(1, b + 1) for a in range(c)])
        clean = torch.randint(0, 10, (batch, self.input_length), generator=g)
        clean[:, :3] = duel[torch.randint(0, duel.shape[0], (batch,), generator=g)]
        u = torch.rand((batch,), generator=g)
        corrupt = clean.clone()
        corrupt[:, 2] = clean[:, 1] + 1 + (u * (9 - clean[:, 1])).long()
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        m = torch.zeros(self.output_length(), dtype=torch.bool)
        m[:3] = True
        return m


class LprobeWholeCircuit(LinearProbe, DomainCircuit):
    """lprobe_whole on lprobe_id: isolated via FULL (integrated)."""
    circuit_id = 'lprobe_whole'
    algorithm = 'lprobe_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class LprobeAlleqCircuit(LinearProbe, DomainCircuit):
    """lprobe_alleq on lprobe_id: isolated via Condition('all_equal', lambda x: bool((x == x[0]).all()), lambda batch, seed: torch.randint(0, 10, (batch, 1), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'lprobe_alleq'))).expand(batch, 10).contiguous()) (integrated)."""
    circuit_id = 'lprobe_alleq'
    algorithm = 'lprobe_id'
    _cond = conditions.Condition('all_equal', lambda x: bool((x == x[0]).all()), lambda batch, seed: torch.randint(0, 10, (batch, 1), generator=torch.Generator().manual_seed(conditions.stream_key(seed, 'lprobe_alleq'))).expand(batch, 10).contiguous())
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class LprobeSpacedCircuit(LinearProbe, DomainCircuit):
    """lprobe_spaced on lprobe_id: isolated via SPACED_PAIRS (integrated)."""
    circuit_id = 'lprobe_spaced'
    algorithm = 'lprobe_id'
    _cond = conditions.SPACED_PAIRS
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class LprobeRetargetCircuit(LinearProbe, ContrastCircuit):
    """lprobe_retarget on lprobe_id: contrast vs lprobe_whole [retarget-free-slot (aim one key at either of two FREE slots)] (integrated)."""
    circuit_id = 'lprobe_retarget'
    algorithm = 'lprobe_id'
    base_circuit = 'lprobe_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        base = torch.randint(0, 10, (batch, L), generator=g)
        clean, corrupt = base.clone(), base.clone()
        for i in range(batch):
            p = int(torch.randint(0, L - 1, (1,), generator=g))     # p <= L-2 leaves >= 2 slots free
            occ = [False] * 10
            for v in base[i, :p].tolist():
                s = v
                while occ[s]:
                    s = (s + 1) % 10
                occ[s] = True
            free = [s for s in range(10) if not occ[s]]
            a, b = torch.randperm(len(free), generator=g)[:2].tolist()
            clean[i, p], corrupt[i, p] = free[a], free[b]
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class LprobePen2Circuit(LinearProbe, ContrastCircuit):
    """lprobe_pen2 on lprobe_id: contrast vs lprobe_whole [retarget-free-slot (penultimate key: mask provably {8,9})] (integrated)."""
    circuit_id = 'lprobe_pen2'
    algorithm = 'lprobe_id'
    base_circuit = 'lprobe_whole'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        base = torch.randint(0, 10, (batch, L), generator=g)
        clean, corrupt = base.clone(), base.clone()
        for i in range(batch):
            occ = [False] * 10
            for v in base[i, :L - 2].tolist():
                s = v
                while occ[s]:
                    s = (s + 1) % 10
                occ[s] = True
            f0, f1 = [s for s in range(10) if not occ[s]]
            clean[i, L - 2], corrupt[i, L - 2] = f0, f1
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        m = torch.zeros(self.output_length(), dtype=torch.bool)
        m[-2:] = True
        return m


class DutchWholeCircuit(FlagPartition, DomainCircuit):
    """dutch_whole on dutch_id: isolated via FULL (integrated)."""
    circuit_id = 'dutch_whole'
    algorithm = 'dutch_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DutchNomidCircuit(FlagPartition, DomainCircuit):
    """dutch_nomid on dutch_id: isolated via NO_MIDS (integrated)."""
    circuit_id = 'dutch_nomid'
    algorithm = 'dutch_id'
    _cond = conditions.NO_MIDS
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class DutchOrderedCircuit(FlagPartition, DomainCircuit):
    """dutch_ordered on dutch_id: isolated via FLAG_ORDERED (integrated)."""
    circuit_id = 'dutch_ordered'
    algorithm = 'dutch_id'
    _cond = conditions.FLAG_ORDERED
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return Id(self.input_length).eval(input_seq)


class DutchPromoteCircuit(FlagPartition, ContrastCircuit):
    """dutch_promote on dutch_id: contrast vs dutch_whole [class-change (re-draw one digit until its flag class differs)] (integrated)."""
    circuit_id = 'dutch_promote'
    algorithm = 'dutch_id'
    base_circuit = 'dutch_whole'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        cls = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        clean = torch.randint(0, 10, (batch, L), generator=g)
        corrupt = clean.clone()
        for i in range(batch):
            done = False
            for _ in range(256):                     # reject only the (position, value) draw, so clean stays uniform
                p = int(torch.randint(0, L, (1,), generator=g))
                v = int(torch.randint(0, 10, (1,), generator=g))
                if int(cls[v]) != int(cls[clean[i, p]]):
                    corrupt[i, p] = v
                    done = True
                    break
            assert done, 'dutch_promote: no cross-class value drawn in 256 tries'
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return self.eval(clean_row) != self.eval(corrupt_row)


class DutchSameswapCircuit(FlagPartition, ContrastCircuit):
    """dutch_sameswap on dutch_id: contrast vs dutch_whole [position-swap (adjacent SAME-class pair)] (integrated)."""
    circuit_id = 'dutch_sameswap'
    algorithm = 'dutch_id'
    base_circuit = 'dutch_whole'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        L = self.input_length
        cls = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        clean = torch.empty(batch, L, dtype=torch.long)
        corrupt = torch.empty(batch, L, dtype=torch.long)
        for i in range(batch):
            done = False
            for _ in range(256):                     # the edit is undefined on rows with no such adjacent pair
                row = torch.randint(0, 10, (L,), generator=g)
                cand = [j for j in range(L - 1)
                        if int(cls[row[j]]) == int(cls[row[j + 1]]) and int(row[j]) != int(row[j + 1])]
                if not cand:
                    continue
                j = cand[int(torch.randint(0, len(cand), (1,), generator=g))]
                c = row.clone()
                c[j], c[j + 1] = row[j + 1], row[j]
                clean[i], corrupt[i] = row, c
                done = True
                break
            assert done, 'dutch_sameswap: no adjacent same-class differing pair in 256 rows'
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        cls = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        j = int((clean_row != corrupt_row).nonzero()[0])
        k = int(cls[clean_row[j]])
        d = int((cls[clean_row] < k).sum()) + int((cls[clean_row[:j]] == k).sum())
        m = torch.zeros(self.output_length(), dtype=torch.bool)
        m[d], m[d + 1] = True, True
        return m


class Slew2FullCircuit(Slew2, DomainCircuit):
    """slew2_full on slew2_id: isolated via FULL (integrated)."""
    circuit_id = 'slew2_full'
    algorithm = 'slew2_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Slew2BinaryCircuit(Slew2, DomainCircuit):
    """slew2_binary on slew2_id: isolated via BINARY (integrated)."""
    circuit_id = 'slew2_binary'
    algorithm = 'slew2_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Slew2CalmCircuit(Slew2, DomainCircuit):
    """slew2_calm on slew2_id: isolated via STEP_LE2 (integrated)."""
    circuit_id = 'slew2_calm'
    algorithm = 'slew2_id'
    _cond = conditions.STEP_LE2
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Slew2EvenCircuit(Slew2, DomainCircuit):
    """slew2_even on slew2_id: isolated via ALL_EVEN (integrated)."""
    circuit_id = 'slew2_even'
    algorithm = 'slew2_id'
    _cond = conditions.ALL_EVEN
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Slew2SaturatedCircuit(Slew2, DomainCircuit):
    """slew2_saturated on slew2_id: isolated via SLEW_SAT (integrated)."""
    circuit_id = 'slew2_saturated'
    algorithm = 'slew2_id'
    _cond = conditions.SLEW_SAT
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Slew2NudgeCircuit(Slew2, ContrastCircuit):
    """slew2_nudge on slew2_id: contrast vs slew2_calm [value-perturb-control] (integrated)."""
    circuit_id = 'slew2_nudge'
    algorithm = 'slew2_id'
    base_circuit = 'slew2_calm'
    local_predicate = True
    _cond = conditions.STEP_LE2
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        pool = DomainCircuit.sample(self, 8 * batch, seed)
        cleans, corrupts = [], []
        for row in pool:
            x = row.tolist()
            opts = [(i, d) for i in range(1, 9) for d in (-1, 1)
                if 0 <= x[i] + d <= 9
                and abs(x[i] + d - x[i - 1]) <= 2
                and abs(x[i + 1] - x[i] - d) <= 2]
            if not opts:
                continue
            i, d = opts[int(torch.randint(0, len(opts), (1,), generator=g))]
            corrupt = row.clone()
            corrupt[i] = x[i] + d
            cleans.append(row.clone())
            corrupts.append(corrupt)
            if len(cleans) == batch:
                return torch.stack(cleans), torch.stack(corrupts)
        raise CircuitError(f"{self.circuit_id}: only {len(cleans)}/{batch} calm rows admit an in-band nudge")
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return clean_row != corrupt_row


class Slew2BreakCircuit(Slew2, ContrastCircuit):
    """slew2_break on slew2_id: contrast vs slew2_calm [threshold-straddle] (integrated)."""
    circuit_id = 'slew2_break'
    algorithm = 'slew2_id'
    base_circuit = 'slew2_calm'
    local_predicate = True
    _cond = conditions.STEP_LE2
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        pool = DomainCircuit.sample(self, 8 * batch, seed)
        cleans, corrupts = [], []
        for row in pool:
            x = row.tolist()
            opts = [i for i in range(1, 10) if x[i - 1] <= 4 and x[i] != x[i - 1] + 2]
            if not opts:
                continue
            i = opts[int(torch.randint(0, len(opts), (1,), generator=g))]
            corrupt = row.clone()
            corrupt[i] = x[i - 1] + 5
            cleans.append(row.clone())
            corrupts.append(corrupt)
            if len(cleans) == batch:
                return torch.stack(cleans), torch.stack(corrupts)
        raise CircuitError(f"{self.circuit_id}: only {len(cleans)}/{batch} calm rows admit a limiter break")


class Slew2X0toggleCircuit(Slew2, ContrastCircuit):
    """slew2_x0toggle on slew2_id: contrast vs slew2_full [digit-toggle] (integrated)."""
    circuit_id = 'slew2_x0toggle'
    algorithm = 'slew2_id'
    base_circuit = 'slew2_full'
    local_predicate = False
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        clean = DomainCircuit.sample(self, batch, seed)
        clean[:, 0] = 0
        corrupt = clean.clone()
        corrupt[:, 0] = 9
        return clean, corrupt


class Slew2PsumFullCircuit(Slew2, DomainCircuit):
    """slew2_psum_full on slew2_psum_id: isolated via FULL (integrated)."""
    circuit_id = 'slew2_psum_full'
    algorithm = 'slew2_psum_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Rms3FullCircuit(Rms3, DomainCircuit):
    """rms3_full on rms3_id: isolated via FULL (integrated)."""
    circuit_id = 'rms3_full'
    algorithm = 'rms3_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Rms3BinaryCircuit(Rms3, DomainCircuit):
    """rms3_binary on rms3_id: isolated via BINARY (integrated)."""
    circuit_id = 'rms3_binary'
    algorithm = 'rms3_id'
    _cond = conditions.BINARY
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Rms3ImpulseCircuit(Rms3, DomainCircuit):
    """rms3_impulse on rms3_id: isolated via exactly_one_in([1, 2, 3, 4, 5, 6, 7, 8, 9]) (integrated)."""
    circuit_id = 'rms3_impulse'
    algorithm = 'rms3_id'
    _cond = conditions.exactly_one_in([1, 2, 3, 4, 5, 6, 7, 8, 9])
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Rms3FlatwinCircuit(Rms3, DomainCircuit):
    """rms3_flatwin on rms3_id: isolated via FLATWIN1_GE2 (integrated)."""
    circuit_id = 'rms3_flatwin'
    algorithm = 'rms3_id'
    _cond = conditions.FLATWIN1_GE2
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Rms3ZeroflankCircuit(Rms3, DomainCircuit):
    """rms3_zeroflank on rms3_id: isolated via ZEROFLANK (integrated)."""
    circuit_id = 'rms3_zeroflank'
    algorithm = 'rms3_id'
    _cond = conditions.ZEROFLANK
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Rms3Toggle09Circuit(Rms3, ContrastCircuit):
    """rms3_toggle09 on rms3_id: contrast vs rms3_full [digit-toggle] (integrated)."""
    circuit_id = 'rms3_toggle09'
    algorithm = 'rms3_id'
    base_circuit = 'rms3_full'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        rows = torch.arange(batch)
        p = torch.randint(0, self.input_length, (batch,), generator=g)
        clean[rows, p] = 0
        corrupt = clean.clone()
        corrupt[rows, p] = 9
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        return (torch.arange(10) - p).abs() <= 1


class Rms3ZfToggle09Circuit(Rms3, ContrastCircuit):
    """rms3_zf_toggle09 on rms3_id: contrast vs rms3_zeroflank [digit-toggle] (integrated)."""
    circuit_id = 'rms3_zf_toggle09'
    algorithm = 'rms3_id'
    base_circuit = 'rms3_zeroflank'
    local_predicate = True
    _cond = conditions.ZEROFLANK
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        clean = DomainCircuit.sample(self, batch, seed)
        rows = torch.arange(batch)
        p = 2 * torch.randint(0, self.input_length // 2, (batch,), generator=g)
        clean[rows, p] = 0
        corrupt = clean.clone()
        corrupt[rows, p] = 9
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        p = int((clean_row != corrupt_row).nonzero()[0])
        return (torch.arange(10) - p).abs() <= 1


class Rms3PsumFullCircuit(Rms3, DomainCircuit):
    """rms3_psum_full on rms3_psum_id: isolated via FULL (integrated)."""
    circuit_id = 'rms3_psum_full'
    algorithm = 'rms3_psum_id'
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Spn2TransparentCircuit(Spn2, DomainCircuit):
    """spn2_transparent on spn2_id: isolated via spn2_transparent() (integrated)."""
    circuit_id = 'spn2_transparent'
    algorithm = 'spn2_id'
    _cond = conditions.spn2_transparent()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return Id(self.input_length).eval(input_seq)


class Spn2Key1zeroCircuit(Spn2, DomainCircuit):
    """spn2_key1zero on spn2_id: isolated via spn2_key1zero() (integrated)."""
    circuit_id = 'spn2_key1zero'
    algorithm = 'spn2_id'
    _cond = conditions.spn2_key1zero()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Spn2Key2zeroCircuit(Spn2, DomainCircuit):
    """spn2_key2zero on spn2_id: isolated via spn2_key2zero() (integrated)."""
    circuit_id = 'spn2_key2zero'
    algorithm = 'spn2_id'
    _cond = conditions.spn2_key2zero()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class Spn2SwapCircuit(Spn2, ContrastCircuit):
    """spn2_swap on spn2_id: contrast vs spn2_key1zero [position-swap] (integrated)."""
    circuit_id = 'spn2_swap'
    algorithm = 'spn2_id'
    base_circuit = 'spn2_key1zero'
    local_predicate = True
    _cond = conditions.FULL
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        n = self.input_length
        spots = torch.tensor([(p, q) for p in range(n) for q in range(p + 2, n)])
        vals = torch.tensor([(a, b) for a in range(10) for b in range(10) if a != b])
        pq = spots[torch.randint(0, spots.shape[0], (batch,), generator=g)]
        ab = vals[torch.randint(0, vals.shape[0], (batch,), generator=g)]
        rows = torch.arange(batch)
        clean = torch.randint(0, 10, (batch, n), generator=g)
        clean[rows, pq[:, 0]] = ab[:, 0]
        clean[rows, pq[:, 1]] = ab[:, 1]
        corrupt = clean.clone()
        corrupt[rows, pq[:, 0]] = ab[:, 1]
        corrupt[rows, pq[:, 1]] = ab[:, 0]
        return clean, corrupt
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        return clean_row != corrupt_row


class XparCodewordCircuit(Xpar, DomainCircuit):
    """xpar_codeword on xpar_id: isolated via xpar_codeword() (integrated)."""
    circuit_id = 'xpar_codeword'
    algorithm = 'xpar_id'
    _cond = conditions.xpar_codeword()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)
    def eval(self, input_seq: torch.Tensor) -> torch.Tensor:
        return Id(self.input_length).eval(input_seq)


class XparOnerrorCircuit(Xpar, DomainCircuit):
    """xpar_onerror on xpar_id: isolated via xpar_onerror() (integrated)."""
    circuit_id = 'xpar_onerror'
    algorithm = 'xpar_id'
    _cond = conditions.xpar_onerror()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class XparErrrow0Circuit(Xpar, DomainCircuit):
    """xpar_errrow0 on xpar_id: isolated via xpar_errrow0() (integrated)."""
    circuit_id = 'xpar_errrow0'
    algorithm = 'xpar_id'
    _cond = conditions.xpar_errrow0()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample(self, batch: int, seed: int) -> torch.Tensor:
        return self._cond.sample(batch, seed)


class XparMsgflipCircuit(Xpar, ContrastCircuit):
    """xpar_msgflip on xpar_id: contrast vs xpar_onerror [value-perturb-control] (integrated)."""
    circuit_id = 'xpar_msgflip'
    algorithm = 'xpar_id'
    base_circuit = 'xpar_onerror'
    local_predicate = True
    _cond = conditions.xpar_onerror()
    def condition(self, input_seq: torch.Tensor) -> bool:
        return self._cond.holds(input_seq)
    def sample_pairs(self, batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = conditions.stream_key(seed, self.circuit_id)
        g = torch.Generator().manual_seed(int(seed))
        rows = torch.arange(batch)
        msg = torch.randint(0, 10, (batch, 4), generator=g)
        j = torch.randint(0, 4, (batch,), generator=g)
        delta = torch.randint(1, 10, (batch,), generator=g)
        flip = msg.clone()
        flip[rows, j] = (msg[rows, j] + delta) % 10
        pos = torch.randint(0, 10, (batch,), generator=g)
        err = torch.randint(1, 10, (batch,), generator=g)

        def received(m):
            top = torch.cat([m, (-m.sum(1, keepdim=True)) % 10], dim=1)
            x = torch.cat([top, (-top) % 10], dim=1)
            x[rows, pos] = (x[rows, pos] + err) % 10
            return x

        return received(msg), received(flip)
    def affected_outputs(self, clean_row: torch.Tensor, corrupt_row: torch.Tensor) -> torch.Tensor | None:
        j = int((clean_row != corrupt_row).nonzero()[0])
        m = torch.zeros(self.input_length, dtype=torch.bool)
        m[[j, 4, 5 + j, 9]] = True
        return m
