# DSL operator reference

Every zoo model is a DSL program. Its name is an `algorithm_id`, decoded by
`decode_algorithm_id` in [pipeline/models/algorithms.py](../../pipeline/models/algorithms.py); the
task, output length, `n_ctx`, answer positions and architecture are all recovered from the name.
Each operator composes as `Op(child)`, where the child is another operator or `Id` (the identity on
the raw 10-digit input). IDs are written outermost to innermost: `op1_op2_id` = Op1(Op2(Id)). An
optional `__<arch>` suffix (`__small`, `__wide`, `__deep`, `__xl`, `__mlp`, `__moe4`, `__moe8`; no
suffix = `large`) selects the architecture (see [../models.md](../models.md)).

This page has four parts: the core parameterized operators, the operators behind the showcase
circuits, example models, and an index of every other operator the decoder accepts. The code is the
source of truth; the index at the end is generated from it.

## Core operator table

| Name | ID Format | Params | Output Length | Category | Description |
|------|-----------|--------|---------------|----------|-------------|
| `Id` | `id` | — | L | Trivial | Identity, returns input unchanged |
| `Const` | `c{d}L{n}` | d=digit, n=length | n | Trivial | Constant sequence of digit d |
| `Reverse` | `rev` | — | L | Transform | Flip sequence |
| `Rotate` | `rot{k}` | k=steps | L | Transform | Rotate by k positions (wraps) |
| `Take` | `take{k}` | k | k | Transform | Take first k elements |
| `Drop` | `drop{k}` | k | L−k | Transform | Drop first k elements |
| `MapAdd` | `madd{c}` | c=constant | L | Transform | Add c mod 10 element-wise |
| `MapMul` | `mmul{c}` | c=constant | L | Transform | Multiply by c mod 10 element-wise |
| `PadTo` | `pad{M}f{fill}` | M=length, fill=digit | M | Transform | Pad/truncate to length M |
| `PrefixMax` | `pmax` | — | L | Scan | Running maximum |
| `PrefixSumMod` | `psum` | — | L | Scan | Running sum mod 10 |
| `RunningCountDigit` | `rcnt{d}` | d=digit | L | Scan | Count of digit d in prefix |
| `CompareShift` | `cmpsh{k}` | k=shift | L | Scan | 1 if x[i]==x[i-k] else 0 |
| `DiffShift` | `diffsh{k}` | k=shift | L | Scan | (x[i]−x[i-k]) mod 10 |
| `HashScan` | `hscan{a}b{b}c{c}s{s}` | a,b,c,seed | L | Scan | Linear recurrence: s=(a·s+b·x+c)%10 |
| `AccumulateReset` | `areset{t}` | t=threshold | L | Scan | Accumulate with reset at threshold |
| `ParityInterleave` | `parity{a}e{b}o{c}f{d}` | a,b,c,d | L | Scan | Different affine maps on even/odd positions |
| `CellularAutomaton` | `ca{s}a{a}b{b}c{c}` | steps,a,b,c | L | Scan | Parallel neighbor rule: (a·left+b·center+c·right)%10, circular |
| `WindowHash` | `whash{T}w{w}a{a}b{b}c{c}` | T,w,a,b,c | T | Windowed | Polynomial rolling hash over window of size w |
| `SelfConvolve` | `sconv{T}` | T | T | Windowed | Circular self-convolution (data×data products) |
| `JumpIndex` | `jump{T}` | T | T | Walk | Index walk: i=(i+x[i])%L |
| `CycleWalk` | `cwalk{T}` | T | T | Walk | Graph walk: next=(i+1+2·x[i])%L |
| `ConditionalJump` | `cjump{T}t{t}` | T, thresh | T | Walk | Bidirectional jump based on value threshold |
| `TwoPointerMerge` | `tpmerge{T}` | T | T | Walk | Two coupled pointers, output sum |
| `ChainedIndex` | `chain{T}d{d}` | T, depth | T | Walk | Multi-level pointer dereference |
| `IndexMix` | `imix{T}a{a}b{b}` | T,a,b | T | Walk | Coupled index+state walk |
| `PermuteByRank` | `prank{T}` | T | T | Sort | Data-dependent global reordering by score |
| `ChunkSort` | `csort{i}to{j}a` / `csort{i}to{j}d` | i,j,asc | L | Sort | Sort subrange [i,j) ascending or descending |
| `ReductionTree` | `rtree{T}a{a}b{b}` | T,a,b | T | Reduction | Binary tournament: (a·left+b·right)%10, concat levels |
| `ScatterWrite` | `scatter{T}a{a}` | T,a | T | Data-dep | Data-directed writes: value specifies target index |
| `InverseLookup` | `invlook{T}` | T | T | Data-dep | Find first occurrence position of each digit |
| `RunLengthEncode` | `rle{T}` | T | T | Encoding | RLE: output (digit, count) pairs |
| `SegmentReverse` | `segrev{d}` | d=boundary | L | Encoding | Reverse segments delimited by boundary digit |
| `TreeWalkDFS` | `tdfs{T}` | T | T | Graph | DFS of tree encoded by parent pointers |
| `TreeWalkBFS` | `tbfs{T}` | T | T | Graph | BFS of tree encoded by parent pointers |
| `LFSR` | `lfsr{T}w{w}t{t1}t{t2}c{c1}c{c2}` | T,w,tap1,tap2,c1,c2 | T | Register | Linear feedback shift register trace |
| `XORScan` | `xorscan` | — | L | Algebraic | Cumulative XOR mod 10; GF(2) arithmetic |
| `ModRemainder` | `modrem{m}` | m (e.g. 7,11,13,17) | L | Algebraic | Rolling base-10 value mod m (m≠10) |
| `SwitchedScan` | `swscan{a1}e{b1}f{a2}g{b2}t{thresh}` | a1,b1,a2,b2,thresh | L | Algebraic | State-gated switching between two linear recurrences |
| `ZigzagCount` | `zigzag` | — | L | Counting | Cumulative count of strict local extrema |
| `RecordCount` | `reccount` | — | L | Counting | Cumulative count of left-to-right maxima |
| `KMPFailure` | `kmpfail` | — | L | String | KMP failure function: longest proper prefix = suffix |
| `BWT` | `bwt` | — | L | String | Burrows-Wheeler Transform (sort rotations, take last column) |
| `LZ78Complexity` | `lz78` | — | L | Info theory | Cumulative LZ78 phrase count (Lempel-Ziv complexity) |
| `VanEckExtension` | `vaneck{T}` | T | T | Self-referential | Continue Van Eck sequence seeded by input |
| `TaskMux` | `mux{N}.{m1}.{m2}...` | N member ids ('_'→'-') | 10 | Multi-task | Token 0 (=10+i) selects member i, run on the 10-digit payload; input length 11, vocab 10+N |

Notes:

- **L** = length of the child's output (= 10 for an `Id` child); **T** = a fixed output-length
  parameter, typically 6-10.
- All outputs are digit sequences (values 0-9), except where an operator's output alphabet is
  explicitly different (e.g. 0/1 indicators).
- `TaskMux` is one underscore-free token whose members must be length-preserving (10 → 10), e.g.
  `mux3.rev-id.dblodd-id.psum-id` = TaskMux([rev_id, dblodd_id, psum_id]); the grammar is described
  in [design.md](design.md#multi-task-taskmux-models).

## Operators behind the showcase circuits

The five showcase circuits ([../../results/showcase/](../../results/showcase/)) come from these
models:

| model | program | what it computes |
|---|---|---|
| `schmitt_id__moe8` | `Schmitt(Id)` | Schmitt trigger: a switch with a high and a low threshold plus memory. The state starts high (9); a digit ≤ 3 sets it to 0, a digit ≥ 6 sets it to 9, and 4 or 5 holds the previous state; the output is the state at each position. |
| `mexhat_id__moe8` | `MexicanHatVeto(Id)` | Dip detector: an interior digit `x[i]` is replaced by 0 when `x[i-1] + x[i+1] − 2·x[i] ≥ 4` (a dip relative to its neighbours); every other digit is copied. |
| `graviclump_id__moe4` | `MeanRelax(Id)` | Gravity: `m = round(mean(x))`, and every digit moves one step toward `m` (up if below, down if above, unchanged if equal). |
| `psum_novel_id__moe8` | `PrefixSumMod(NoveltyIndicator(Id))` | Running sum mod 10 of the novelty flags (1 at the first occurrence of each value, 0 on repeats), i.e. the number of distinct digits seen so far, mod 10. Its BridgeCircuit probes the inner novelty flag (`bridge_variable('psum_novel_id', 1)`). |
| `mux2.pmstep-id.sr12-id__moe8` | `TaskMux([ParityRoute(Id), SRLatch(1, 2, Id)])` | Cue token 10 selects `pmstep` (add 1 to each even digit, subtract 1 from each odd digit, mod 10); cue token 11 selects `sr12` (a latch set to 1 by a 1, reset to 0 by a 2, held otherwise). Input length 11. |

The ProbeCircuit example in [circuit_types.md](circuit_types.md) uses `Maxrun(Id)` (`maxrun_id`):
the length of the longest run of equal adjacent digits seen so far, capped at 9.

## Single-operator models

Examples of depth-2 models (one operator applied to `Id`); the full run list is
`pipeline/models/models.txt`.

| Model id | Description |
|----------|-------------|
| `areset13_id` | AccumulateReset(13, Id): running sum that wraps to `sum mod 13` whenever it reaches 13; emits the accumulator mod 10. |
| `reccount_id` | RecordCount(Id): running count (mod 10) of left-to-right maxima (records) seen so far. |
| `psumodd_id` | PrefixSumOdd(Id): running sum (mod 10) of the odd-valued digits seen so far; even digits do not contribute. |
| `dblodd_id` | DoubleOdd(Id): doubles each odd-valued digit (mod 10), leaving even digits unchanged; every odd digit becomes even, so the whole output is even-valued. |
| `prod_id` | RunningProduct(Id): running product (mod 10) of the digits seen so far. |
| `cntodd_id` | CountOdd(Id): running count (mod 10) of odd-valued digits seen so far. |
| `recordind_id` | RecordIndicator(Id): 1 at each new left-to-right maximum (record), 0 otherwise. |
| `novel_id` | NoveltyIndicator(Id): 1 at the first occurrence of each value (novel), 0 on repeats. |
| `fixedpoint_id` | FixedPointIndicator(Id): 1 where a digit equals its position index, 0 otherwise. |
| `reset3_id` | ResetRamp(3, Id): positions since the last 3 (a ramp resetting to 0 at every 3). |
| `tresetmax3_id` | TokenResetMax(3, Id): running maximum, resetting to 0 at every 3. |
| `walk12_id` | SignedWalk(1,2, Id): +1 on each 1, -1 on each 2; running total mod 10. |
| `sr12_id` | SRLatch(1,2, Id): bit set to 1 on a 1, reset to 0 on a 2, else held. |
| `holdge5_id` | SampleHold(ge,5, Id): holds the most recent digit >= 5. |
| `dep9f0_id` | Deplete(9,0, Id): counts down from 9, subtracting each digit, floored at 0. |
| `anc0_id` | AnchorSubtract(0, Id): subtracts the digit at position 0 from every digit (mod 10). |
| `fclamp0_id` | AnchorClamp(floor,0, Id): clamps every digit up to at least the digit at position 0. |
| `drot0_id` | DataRotate(0, Id): cyclically left-rotates by the digit at position 0. |
| `refl0_id` | DataReflect(0, Id): mirrors about a pivot read from position 0. |
| `stride0_id` | StrideGather(0, Id): strided gather with stride read from position 0. |
| `pref0_id` | PrefixReverseK(0, Id): reverses the first k elements, k read from position 0. |
| `psort0_id` | PrefixSortK(0, Id): sorts the first k elements ascending, k read from position 0. |
| `cmpswap0p9_id` | CompareSwap(0,9, Id): swaps positions 0 and 9 iff x[0] > x[9]. |
| `bmax1_id` | WindowMax(1, Id): sliding max over the current and 1 preceding digit (a delay-line on monotone input). |
| `pmstep_id` | ParityRoute(Id): +1 on each even digit, -1 on each odd digit (mod 10). |
| `carryk1_id` | RippleCarry(1, Id): adds 1 to the last digit and ripples the carry leftward. |
| `argminmax_id` | ArgMinMax(Id): argmax indicator concatenated with argmin indicator (length-20). |
| `cjump10t5_id` | ConditionalJump(10,5, Id): a 10-step cursor walk jumping forward when the value exceeds 5 and backward otherwise. |
| `segrev3_id` | SegmentReverse(3, Id): reverses each maximal run of non-3 digits; every 3 stays in place as a boundary (the model of the [segrev3 case study](../segrev3_case_study.md)). |

## Example composite and special-purpose models

| algorithm_id | description |
|---|---|
| `dfa3t010020001_id` | 3-state finite automaton over the digit stream: digits are grouped into three classes (0-2, 3-6, 7-9), each class drives a state transition from the `t010020001` table, and the output at each position is the current state. |
| `indecho_id` | Induction / previous-occurrence recall: at each position, emit the digit that followed the current digit's most recent earlier occurrence (the digit itself if it has not occurred before). |
| `cosetprod_id` | Partially observed S₃ register: each digit (mod 6) names an element of the symmetric group S₃, the running product is folded left to right, and only a 3-valued coset label of the running element is emitted. |
| `dblodd_walk10_recordind_id` | 3-step composite: mark left-to-right records (RecordIndicator), take a signed walk over that indicator (SignedWalk(1, 0): +1 on a 1, −1 on a 0, mod 10), then double each odd digit mod 10 (DoubleOdd). |
| `pmaxv9_id` | Running maximum in which the digit 9 is vetoed: 9s never update the maximum. |
| `mux3.sr12-id.sr34-id.sr56-id` | 3-way task multiplexer: a leading cue token selects which set/reset latch (set/reset digits 1/2, 3/4 or 5/6) to apply to the payload. |

## Index of all other operators

Every other operator the decoder accepts, as `id` and class name. Most of these are single-purpose
operators added for the zoo; their docstrings and reference implementations (`_ref_<id>`) are in
`algorithms.py`, and plain-language explanations of many of them are in
[../brainstorming/operator_glossary.md](../brainstorming/operator_glossary.md). `{d}` is one digit,
`{n}` a multi-digit integer.

| | | |
|---|---|---|
| `sarank` Sarank | `pmaxv9` Pmaxv9 | `dfa3t010020001` SmallDFATrace |
| `indecho` Indecho | `cosetprod` GroupCosetProductS3 | `kanc2` Kanc2 |
| `kanc4` Kanc4 | `kanc8` Kanc8 | `kanc10` Kanc10 |
| `sanc2` Sanc2 | `tmap5` Tmap5 | `addfix` Addfix |
| `sjf1` Sjf1 | `duelsum` Duelsum | `netrm` Netrm |
| `convfix` Convfix | `usum` Usum | `perase0` Perase0 |
| `lagf3` Lagf3 | `nearidx` Nearidx | `tresetsum3` Tresetsum3 |
| `oepass0` Oepass0 | `pge` Pge | `cgpmax09` Cgpmax09 |
| `lanesum` Lanesum | `addpmax` Addpmax | `maxrun` Maxrun |
| `dualtie` Dualtie | `cap59` Cap59 | `cgrev09` Cgrev09 |
| `dupmax` Dupmax | `rmaj5` Rmaj5 | `majmask` Majmask |
| `bcast0` AnchorBroadcast | `zpeak` Zpeak | `mulanc0` Mulanc0 |
| `recdist` Recdist | `pmrep` Pmrep | `mulpair` Mulpair |
| `prevsegmax3` PrevSegmentMax | `occcnt` Occcnt | `segdist3` SegmentDistinct |
| `dclamp` Dclamp | `firstuniq` FirstUnique | `runmed` Runmed |
| `cntge3` Cntge3 | `twosum10` Twosum10 | `holdodd` Holdodd |
| `addlastodd` Addlastodd | `bmcand` Bmcand | `hold2ge5` Hold2ge5 |
| `sjf2` Sjf2 | `gapge5` Gapge5 | `riseerase` Riseerase |
| `ebud3` Ebud3 | `tieswap` Tieswap | `risemax` Risemax |
| `delc2` Delc2 | `riskeep` Riskeep | `mixtie` Mixtie |
| `pmaxv89` Pmaxv89 | `pmaxv789` Pmaxv789 | `togsum0` Togsum0 |
| `signsum0` Signsum0 | `med3` Median3 | `maj3` Majority3 |
| `spile4` Spile4 | `spile5` Spile5 | `spile6` Spile6 |
| `cyca` Cyca | `r184t4` Rule184Thr4 | `r184t5` Rule184Thr5 |
| `r184t6` Rule184Thr6 | `ewma2` Ewma2 | `zcross5` Zcross5 |
| `heapmax` Heapmax | `vig2` Vig2 | `vig3` Vig3 |
| `vig5` Vig5 | `hamk3` Hamk3 | `altrun` Altrun |
| `blkpar` BlockSwapParity | `blkoe2` BrickSort2 | `blkadd` FeistelAdd |
| `blkmax` BlockMax2 | `errdiff2` ErrorDiffuse2 | `errdiff3` ErrorDiffuse3 |
| `errdiff5` ErrorDiffuse5 | `boxsum2` BoxSum2 | `boxsum3` BoxSum3 |
| `boxsum4` BoxSum4 | `boxsum5` BoxSum5 | `schmitt` Schmitt |
| `railfence2` RailFence2 | `railfence3` RailFence3 | `railfence4` RailFence4 |
| `qlen3` QueueWalk3 | `qlen4` QueueWalk4 | `qlen5` QueueWalk5 |
| `qlen6` QueueWalk6 | `bpath` MinBottleneck | `wpath` MaxCapacity |
| `hsub` HeapSubtreeMax | `incrun` IncRunMax | `haar` HaarStep |
| `runmad` RunMAD | `contfrac` ContFrac | `s3prod` GroupProductS3 |
| `d4prod` GroupProductD4 | `q8prod` GroupProductQ8 | `d5prod` GroupProductD5 |
| `d4char` D4Character | `d5act` GroupActionD5 | `rangeadd` RunningRangeAdd |
| `medadd` RunningMedianAdd | `tribscan` ThirdOrderScan | `rotor3` LinearRotor3 |
| `modecntadd` RunningModeCountAdd | `oscpos` Oscpos | `oscmom` Oscmom |
| `refrac` RefractoryVeto | `mexhat` MexicanHatVeto | `maxfreq` MaxFrequency |
| `ducci` DucciStep | `pgcd` PairGcd | `fibscan` SecondOrderScan |
| `meanint` NivenByCount | `euclstep` EuclidStep | `paircarry` PairCarry |
| `invcount` InversionCount | `majindex` MajorIndex | `nimhist` NimHistogram |
| `graviclump` MeanRelax | `entrain` AnchorNudge | `phaselock` PhaseNudge |
| `heatx` HeatExchange | `e2sym` ElemSym2 | `runmex` RunningMex |
| `paladd` MirrorAdd | `palmax` MirrorMax | `palmin` MirrorMin |
| `rankadd` OnlineRankAdd | `signdiff` SignDiffuse | `dblge5` DoubleGE |
| `midadd` MidrangeAdd | `seedsum` SpacedSeedSum | `argposadd` ArgPosAdd |
| `lanemax` LaneRunningMax | `pancake` PancakeFlip | `bitmaj` BitPlaneMajority |
| `skipn5` PredicatedSkip | `bittv` BitTransitionCount | `vm1` OpcodeMachine |
| `tourney` KnockoutDepth | `envmax` EnvelopeMax | `cmin` CountMinSketch |
| `bittrans` BitTransposeGather | `intern` InternRank | `dupxor` MultiplicityParityXor |
| `nbrdeg` ProximityDegree | `cfreq` CumFreqRank | `dsub` DistinctSubseqCount |
| `codonskip` CodonSkip | `fraglen` FragLen | `divzeta` DivisorZeta |
| `mobinv` MobiusInvert | `factnorm` FactoradicNormalize | `nonadj4` NonAdjacentMax4 |
| `hkconf` BoundedConfidenceMean | `baksnep` ExtremalReplace | `hsifte` Hsifte |
| `lprobe` LinearProbe | `dutch` FlagPartition | `slew2` Slew2 |
| `rms3` Rms3 | `spn2` Spn2 | `xpar` Xpar |

To list every operator from the code (class and id pattern), read the branches of `parse_op` inside
`decode_algorithm_id`; for example:

```bash
python - <<'PY'
import re
src = open("pipeline/models/algorithms.py").read()
body, pat, n = src[src.index("def decode_algorithm_id"):], None, 0
for ln in body.split("\n"):
    if ln.startswith("    def ") and "parse_op" not in ln and n:
        break                                   # end of parse_op
    m = re.search(r"part == '([^']+)'", ln) or re.search(r"re\.match\(r'([^']+)', part\)", ln)
    if m:
        pat = m.group(1)
        continue
    m2 = re.search(r"return \('(\w+)'", ln)
    if m2 and pat:
        print(m2.group(1), pat)
        pat, n = None, n + 1
PY
```
