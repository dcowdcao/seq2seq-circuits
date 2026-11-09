# Operator glossary — what each operator actually computes

Plain-language companion to [brainstormed_models.md](brainstormed_models.md) (proposed operators);
the operators that are implemented are listed in [../reference/operators.md](../reference/operators.md).
The brainstorm table often names an operator after its
real-world SOURCE (a physics model, a group, a named algorithm); this doc says, in ordinary words, what
each one actually does to a row of ten digits (positions 0–9, left to right), with a worked example.
Every operator maps a length-10 row of digits 0–9 to another length-10 row. Where a famous name applies,
it appears only in a "(Known as: …)" tag at the end — the explanation never leans on it.

Operators are grouped by the KIND of mechanism, not by the batch that proposed them. Analysis/measurement
rows in the master table (the "ZT-…", "protocol", "code", "instrument", "diagnosis" kinds) are experiments
run ON operators, not operators themselves, and are not listed here. Op-families / ladders (e.g.
`ca-ladder`, `kanc{K}`) are a set of operators sharing a rule with a tunable knob; the shared rule is
explained once and one rung is shown. Find any operator by searching (Ctrl-F) its `id`.

Two conventions used throughout: **"mod 10"** means digits wrap like a 10-hour clock (after 9 comes 0),
i.e. "keep the last digit" (13 → 3). A digit's **"parity"** is whether it is even or odd.

## Contents
1. Per-digit maps · 2. Running sums, counts & accumulators · 3. Extremes & order statistics ·
4. Neighbour & local-window arithmetic · 5. Reversals, rotations & permutations ·
6. Sorting & comparison networks · 7. Cellular automata & local-sweep rules *(physics)* ·
8. Oscillators, relaxation & thermalization *(physics)* · 9. Signal processing *(filters/envelopes)* ·
10. Number theory · 11. Group theory & symmetry *(math)* · 12. Recurrences · 13. State machines & tiny
machines · 14. Bit-level · 15. Coding, ciphers & error-correction · 16. Cache, scheduling & regulators ·
17. Suppression / veto · 18. Data structures in the stream · 19. Multi-step compositions & bridges ·
20. Multi-task routers (mux)

---

## 1. Per-digit maps
Each output digit depends only on the input digit in the same spot (no memory, no neighbours) — a fixed
lookup table applied everywhere. The simplest family; mostly building blocks and controls.

**cneg1_id** (nines-complement if any 1 is present) — Scan the whole row once: if a digit 1 appears
anywhere, replace every digit x with 9−x; if no 1 appears, copy the row unchanged. One global yes/no
decision drives a per-digit flip. (Known as: the nines' complement, gated on the presence of a 1.)
  Example: 3141592653 → 6858407346 (a 1 is present, so every digit becomes 9−x).

**posadd_id** (add the position number) — Add each digit's position index to it, mod 10: y[i] = (x[i] + i)
mod 10. Position 0 is unchanged, position 9 gets +9. The position is used as a number, not an address.
  Example: 3141592653 → 3264948332 (position 5's 9 → 9+5=14→4; position 7's 6 → 6+7=13→3).

**mulanc0_id** (multiply every digit by the first digit) — Read the first digit as a fixed multiplier,
then replace every digit x with (first digit × x) mod 10. The first digit picks which times-table applies
to the whole row. (Known as: mod-10 multiplication by an anchor; because 2×5=0 mod 10, this can't be
relabelled into an addition.)
  Example: 3141592653 → 9323576859 (first digit 3, so every digit is tripled mod 10).

**tmap5_id** (fold the high half down) — Each digit is mapped on its own: a digit 5 or more gets 5 added
mod 10 (landing in 0–4); a digit below 5 passes through unchanged. The output only ever uses 0–4.
  Example: 3141592653 → 3141042103 (5→0, 9→4, 6→1; the low digits are untouched).

**dblge5_id** (threshold-gated doubling) — Double every digit that is 5 or more (keeping the last digit,
mod 10); leave digits under 5 unchanged.
  Example: 3141592653 → 3141082203 (5→0, 9→8, 6→2; the under-5 digits pass through).

**kanc{K} ladder** (class-routed anchor map — a family, K = 2, 4, 8, 10) — Sort each digit into one of K
classes by its remainder mod K. Every class owns a fixed rule y = (this digit + m·(first digit) + b) mod
10 with its own multiplier m and offset b. The K rules are the first K rows of one master table, so the
only thing that changes as you climb the ladder is how finely digits are split into classes — K is the
dial (K=2 splits even/odd, K=10 gives every digit its own rule).
  Example (K=2, first digit 3): 3141592653 → 0848262620 (odd digits use (m,b)=(5,2), even use (3,1); the
  leading 3 is odd → (3 + 5·3 + 2) mod 10 = 0).

## 2. Running sums, counts & accumulators
Keep a running total (or count) while sweeping left to right, and print it — usually one digit with mod-10
wraparound. The value at each position summarizes everything seen so far.

**runmode/pmode_id** (running most-common digit) — At each position, output the digit that has occurred
most often in the row so far. On a tie the earliest-seen value wins (the alternative smallest-digit
tiebreak makes an all-distinct input collapse to the running minimum). (Known as: the streaming mode.)
  Example: 2727712257 → 2222772222 (2 and 7 stay tied early, broken toward the earlier-seen 2; 7 leads at
  positions 4–5, then 2 retakes the lead).

**mode9_id** (whole-row most-common digit, printed once) — Find the single most frequent digit over the
entire row, then emit it at one fixed slot only (illustrative: the last position) with 0 elsewhere.
  Example: 2775772772 → 0000000007 (7 occurs six times; printed only in the final slot).

**wsum_id** (running weighted checksum) — Keep a running total in which each digit is multiplied by a
fixed per-position weight before being added, reported mod 10 at every step. (Parameters illustrative:
weight = position+1.) (Known as: a weighted checksum, like the ISBN-10 check digit.)
  Example: 3141592653 → 3571604277 (running (Σ (i+1)·x[i]) mod 10).

**psum_novel_id** (running count of distinct values) — Keep a set of the values seen so far; each time a
brand-new value appears, bump a counter (mod 10); repeats leave it flat. (Known as: running count-distinct
/ cardinality.)
  Example: 3141592653 → 1233456777 (new values 3,1,4 raise it to 3; then 5,9,2,6 push it to 7; trailing
  5,3 are repeats).

**rcnt1_cmpsh1_id** (running count of "same as left") — Each time a digit equals the one immediately
before it, bump a counter (mod 10); otherwise hold. Position 0 never counts.
  Example: 7725551883 → 0111233344 (the second 7, the run of 5s, and the pair of 8s each tick it up).

**rmaj5_id** (running majority of high digits) — At each position output 1 if strictly more than half of
the digits so far are 5 or greater, else 0 (an exact half is 0). Only the fraction of high digits matters.
  Example: 3141592653 → 0000000000 (the high digits 5 and 9 never exceed half the prefix, so it stays 0).

**lanesum_id** (two interleaved running sums) — Keep two separate running sums mod 10: one over the even
positions, one over the odd positions. Each output reports only its own lane's sum so far.
  Example: 3141592653 → 3172214790 (position 4 sums even-position digits 3,4,5 → 2; position 5 sums
  odd-position digits 1,1,9 → 1).

**addfix_id** (digit plus coincidence count) — Keep a running tally of how many positions so far held a
digit equal to their own position number (position 0 holding 0, position 1 holding 1, …). Output each
digit plus that tally, mod 10. (Known as: counting fixed points, where value = index.)
  Example: 3143596659 → 3255719983 (coincidences at positions 1,3,6,9 step the tally, added to each digit).

**usum_id** (running sum of distinct values) — Running sum mod 10, but each distinct value contributes only
the first time it appears; repeats add nothing.
  Example: 3141592653 → 3488324000 (the second 1, 5, and 3 are ignored, so the sum freezes at the end).

**tresetsum3_id** (running sum that resets at every 3) — Running sum mod 10, except every 3 clears the sum
to 0 (and that 3 itself outputs 0).
  Example: 4132539137 → 4502709007 (each 3 knocks the total back to 0).

**cntge3_id** (running count of big digits) — Keep a tally of how many digits so far are 3 or bigger; print
its last digit at each step. (Known as: a threshold counter.)
  Example: 3141592653 → 1122344567 (the tally ticks up at 3,4,5,9,6,5,3 and stalls at the 1s and the 2).

**duelsum_id** (evens-vs-odds scoreboard) — Keep two running totals, one of the even digits and one of the
odd digits. At each position whichever total is larger prints its own last digit; neither is ever shown
directly.
  Example: 3141592653 → 3445099947 (odd digits pull ahead early, so most positions print the odd total).

**occcnt_id** (repeat counter) — For each digit, print how many times that same digit value has already
appeared earlier — effectively ten little tallies, one per value, of which each position bumps and reads
only one.
  Example: 3141592653 → 0001000011 (the second 1, second 5, and second 3 each report one earlier match).

**eqcount_id + coll_cum_id** (repeat counter / collision total) — **eqcount** reports, for each digit, how
many earlier digits equal it — this is the same computation as `occcnt` above, and was dropped
as a duplicate. **coll_cum**, the new operator, is the running TOTAL of those counts — how many equal-digit
collisions have piled up so far. (Known as: the birthday-paradox collision count / Simpson diversity /
Herfindahl concentration.)
  Example: 3141592653 → eqcount 0001000011, coll_cum 0001111123 (the repeats at positions 3, 8, 9 each
  score 1 in eqcount; coll_cum accumulates them).

**invcount_id + majindex_id** (running "unsortedness" pair) — Both track how far from sorted the digits
are so far, mod 10. **invcount** counts every pair out of order (a larger digit before a smaller one).
**majindex** instead sums the POSITIONS where a digit drops below the one just before it. A theorem forces
these two to share the same overall distribution though they differ row by row. (Known as: the two Mahonian
statistics — inversions and MacMahon's major index.)
  Example: 3141592653 → inv 0113337805, maj 0114440087.

**runmad_id** (distance from the running average) — Output how far the current digit lands from the average
of all digits seen so far, with that average rounded to a whole number (absolute difference). (Known as:
mean absolute deviation / z-score.)
  Example: 3141592653 → 0111252211 (the 9 at position 5 is 5 away from the running average of ≈4).

**e2sym_id** (running sum of pairwise products) — Keep a running total, mod 10, of the products of every
pair of digits seen so far; equivalently, at each new digit add (new digit × sum of all previous digits).
(Known as: the 2nd elementary symmetric polynomial / Newton's identities.)
  Example: 3141592653 → 0397284497 (0 at first — no pair yet; 3,1 give product 3; adding 4 contributes
  3·4 + 1·4 = 16 → 9).

**modecntadd_id** (running mode-count add) — Add to each digit the size of the largest group so far — the
height of the tallest bar in the running histogram of the digits — mod 10.
  Example: 3141592653 → 4253714875 (once the second 1 arrives, the tallest bar is 2, added from then on).

**maxfreq_id** (running mode count) — Print how many times the most common digit has appeared so far (the
tallest bar of the running histogram). (Known as: the largest part of the running count-partition.)
  Example: 3141592653 → 1112222222 (stays 1 until the second 1 lifts the tallest bar to 2; nothing reaches
  three copies).

**psumk-ladder** (iterated running sum — a family, k = 1, 2, 3) — Apply "running sum mod 10" to the row,
then feed the result back and do it again — k times in total. k is the dial. (Known as: iterated prefix
sums.)
  Example (k=2): 3141592653 → 3754816732 (one prefix-sum, then a prefix-sum of that).

## 3. Extremes & order statistics
Track the largest / smallest / most-common / rank-ordered value seen so far (or across the whole row).
"Order statistics" means facts about the sorted order — max, min, median (middle), range (max−min), mode
(most frequent), rank (how many are below).

**prank_id** (rank so far) — For each position, output how many earlier digits are strictly smaller than
the current one — its rank within the prefix before it. (Known as: online rank; these counts sum to the
number of inversions.)
  Example: 3141592653 → 0020452653 (the 5 at position 4 has four earlier smaller digits; the 9 has five).

**rrange_id** (running spread) — Track the largest and smallest digit seen so far and output their
difference (max−min) at every step. (Known as: the running range.)
  Example: 3141592653 → 0233488888 (once min locks to 1 and max climbs to 9, the gap settles at 8).

**addpmax_id** (add the best-so-far) — At each position, add the largest of the earlier digits to the
current digit, mod 10 (0 before any earlier digit). The running maximum is used everywhere but never shown.
  Example: 3141592653 → 3475941542 (position 6's 2 gets the earlier max 9 → 1).

**recdist_id** (steps since the last record high) — Keep the position of the most recent left-to-right
record (a digit larger than everything before it; position 0 always counts) and output how many positions
since. A new record resets it to 0. (Known as: time-since-last-record.)
  Example: 3141592653 → 0101001234 (records at 3,4,5,9; after the 9 nothing beats it, so it counts up).

**runmed_id** (running median) — Output the lower median of all digits seen so far: sort the prefix and
report the element at index ⌊i/2⌋.
  Example: 3141592653 → 3131333343 (after "3,1,4" the sorted prefix 1,3,4 has middle 3).

**cgpmax09_id** (compare-ends, then running max or copy) — Look at the first and last digit. If the first
is bigger, output the running maximum at each position; otherwise output the row unchanged.
  Example: 9141592650 → 9999999999 (first 9 > last 0 → running max, and a leading 9 makes it 9 everywhere).

**dupmax_id** (max among repeated values) — Output the largest value that has appeared at least twice so
far (0 until any value repeats); a value qualifies at its second occurrence.
  Example: 3141592653 → 0001111155 (1 first repeats at position 3; 5 repeats at position 8).

**netrm_id** (nearer of the two extremes) — Track the largest and smallest digits seen so far (including
the current one). At each position print whichever is numerically closer to the current digit; exact tie →
the largest.
  Example: 3141592653 → 3141591991 (with max 9 and min 1, the tail snaps to whichever it sits nearer).

**nearidx_id** (where the nearer extreme first appeared) — Same choice as netrm (nearer of running
max/min, tie → max) but print the POSITION where that winning extreme first showed up.
  Example: 3141592653 → 0121451551 (min 1 first at position 1, max 9 first at position 5; output is one of
  those indices).

**dclamp_id** (clamp into the earlier range) — For each digit, look at the highest and lowest digits
BEFORE it. If it exceeds the earlier high, print the high; if below the earlier low, print the low;
otherwise copy. The first digit copies; the second always becomes the first.
  Example: 3141592653 → 3331452653 (positions 2 and 4 exceed the running high and are capped).

**wmax ladder (bmax2_id, bmax3_id; anchors bmax1, pmax)** (sliding-window maximum) — Print the largest
digit among the last k+1 positions (the current one plus the k before it). k dials the window width: k=1 is
each adjacent pair, k=9 covers the whole prefix (ordinary running max).
  Example (k=2, window 3): 3141592653 → 3344599966 (each output is the biggest of the current digit and
  its two left neighbours).

**heapmax_id** (parent-vs-two-children max) — Lay the row out as a complete binary tree (position i's
children sit at 2i+1 and 2i+2). Each position becomes the largest of itself and its two children; the
childless bottom half copies. (Known as: one level of an implicit binary heap / tournament tree.)
  Example: 3141592653 → 4596592653 (position 0's children 1,4 → 4; position 2's children 9,2 → 9; leaves
  5–9 copy).

**hsub_id** (whole-subtree max) — Same tree layout, but each position becomes the largest digit anywhere
in its ENTIRE subtree — so position 0 becomes the largest in the whole row. (Known as: a range-max
reduction over an implicit heap / segment tree.)
  Example: 3141592653 → 9696592653 (position 0 sees the whole row → 9; leaves copy).

**bpath_id + wpath_id** (best-bottleneck / best-widest toll pair) — Read each digit as a toll and step
left to right, each move going forward 1 or 2 places. **bpath**: over all paths reaching this position,
take the one whose LARGEST toll is as small as possible, and print that smallest-possible worst toll —
b[i] = max(x[i], min(b[i−1], b[i−2])). **wpath** is the mirror (largest-possible smallest toll). Neither is
a plain running max/min — a low pair of predecessors lets a later value duck back down. (Known as:
bottleneck / widest shortest-path on the min-max semiring.)
  Example (bpath): 3141592653 → 3343595655 (at position 6 the predecessors are 9 and 5, so the route uses
  min(9,5)=5 then max(2,5)=5 — the output DROPS 9→5, which a running max never does). wpath → 3131332333.

**maxrun_id** (longest identical-run so far) — Track the length of the current run of identical adjacent
digits (it resets to 1 whenever a digit differs from the one before) and output the longest such run seen
up to here, capped at 9.
  Example: 7725551883 → 1222233333 (the "77" reaches 2, then the "555" run reaches 3 and stays the record).

**incrun_id** (longest rising-run so far) — Track the longest run of strictly increasing digits; any digit
not bigger than the one before it restarts the count at 1. Print the longest such run seen up to here.
(Known as: the longest-increasing-run DP.)
  Example: 3141592653 → 1122233333 (the longest rise is 1,5,9, length 3).

**altrun_id** (longest zigzag-run so far) — Track the longest run where digits strictly alternate up, down,
up, down… (each step reversing direction); a flat step or two the same way restarts it. Print the longest
such run seen up to here. (Known as: the longest alternating-subarray / Euler-zigzag statistic.)
  Example: 3141592653 → 1234555555 (3,1,4,1,5 alternates down-up-down-up, length 5).

**blkmax_id** (block-max, repeated) — Chop the row into fixed groups (default: adjacent pairs) and replace
every digit in a group by that group's largest, so the output is flat within each group. A family that also
dials the SUMMARY (biggest / smallest / sum / most-common) and the GROUP SIZE (2 or 5 digits — a
coarse-graining scale). (Known as: renormalization-group block-spin decimation / max-pooling.)
  Example (groups of 2, "biggest"): 3141592653 → 3344996655 (pairs 31,41,59,26,53 collapse to their larger
  digit).

**medadd_id** (add the running median) — Take the middle value of all digits seen so far (lower median)
and add it to the current digit, mod 10. (Known as: order statistics / robust location.)
  Example: 3141592653 → 6272825996 (first digit 3 + median{3} = 6).

**rangeadd_id** (running range add) — Add the spread-so-far (largest minus smallest seen up to here) to
each digit, mod 10.
  Example: 3141592653 → 3374970431 (when the 9 arrives the spread locks at 8, added to every later digit).

**midadd_id** (running-midrange add) — Add the running midrange — the average of the biggest and smallest
digit so far, rounded down — to each digit, mod 10. (Known as: the midrange, a robust location estimator.)
  Example: 3141592653 → 6363847108 (once the 9 arrives the midrange is (9+1)/2 = 5, added from then on).

**argposadd_id** (argmax-position add) — Add the POSITION (an address 0–9) of the biggest-so-far digit to
each digit, mod 10; ties keep the earliest position.
  Example: 3141592653 → 3163947108 (from position 5 on, the max 9 sits at position 5, so 5 is added).

**rankadd_id** (online-rank add) — Add to each digit its running rank — the count of earlier digits
strictly smaller than it — mod 10. (Known as: online rank / inversion counting.)
  Example: 3141592653 → 3161944206 (the 4 at position 2 has two smaller earlier digits → 4+2=6).

**lanemax_id** (parity-lane running max) — Running maximum, but each position only looks at earlier
positions of the SAME parity: even positions track the max over even positions, odd over odd. (Known as:
polyphase per-phase peak-hold.)
  Example: 3141592653 → 3141595959 (from position 5 the odd lane has seen the 9, so later odd positions
  print 9).

## 4. Neighbour & local-window arithmetic
Each output looks at a digit and a few immediate neighbours (usually the one to its left, or a small fixed
window) and combines them.

**zpeak_id** (erase local peaks) — Compare each digit with its two immediate neighbours (a missing
neighbour counts as −1, so the ends can qualify). If it is strictly greater than both, it is a local peak
and becomes 0; otherwise copy. All judged from the original row. (Known as: non-maximum suppression.)
  Example: 3141592653 → 0101502053 (the 3, 4, 9, 6 are strict peaks → 0).

**mulpair_id** (multiply by your left neighbour) — Output each digit times the digit immediately to its
left, mod 10; the first digit is copied.
  Example: 3141592653 → 3344558205 (9·5=45→5, 2·9=18→8, …).

**cmpsh1_id** (equal-to-left-neighbour flag) — Output 1 at each position whose digit equals the digit
immediately to its left, else 0 (position 0 is always 0).
  Example: 7725551883 → 0100110010 (marks the second 7, the middle two 5s, the second 8).

**diffsh1_id** (neighbour difference) — Output each digit minus the digit to its left, mod 10; the first
digit is copied.
  Example: 3141592653 → 3837443498 (position 1 is (1−3) mod 10 = 8).

**diffshk ladder (diffsh1_diffsh1_id, …)** (repeated neighbour differences) — Replace each digit by its
neighbour difference (mod 10, first digit kept), then do that whole pass again — k times over. It is the
exact undo of adding up running sums k times.
  Example (k=2): 3141592653 → 3554709159 (difference once → 3837443498, difference again → 3554709159).

**hamk3_id** (matches to a fixed 1-2-3 stamp) — Slide a fixed 3-slot stamp reading "1, 2, 3" along the row.
At each position count how many of the last three digits (two-back vs 1, one-back vs 2, here vs 3) equal
the stamp; print that count (0–3). (Known as: bounded Hamming distance / approximate string matching.)
  Example: 3141592653 → 1001010101 (position 3 scores 1 — its two-back digit is a 1 hitting the "1" slot).

**ducci_id** (neighbour differences around a ring) — Replace each digit by the absolute difference between
it and its RIGHT neighbour, with the last digit pairing back to the first (a ring). (Known as: the Ducci
sequence / "diffy boxes".)
  Example: 3141592653 → 2334474120 (|3−1|=2, …, final |3−3|=0 wraps position 9 to position 0).

**paircarry_id** (decimal carry bit) — Output 1 at each position where the digit plus its left neighbour
reach 10 or more (the point where an addition would carry), else 0; the first position is always 0.
(Known as: the base-10 half-adder carry-out.)
  Example: 3141592653 → 0000011010 (only 5+9, 9+2, and 6+5 reach ten).

**paladd_id** (mirror-pair sum) — Add each digit to its mirror partner — the digit the same distance in
from the other end (position i pairs with 9−i) — mod 10. The output always reads the same forwards and
backwards. (Known as: a palindromic fold-sum.)
  Example: 3141592653 → 6603443066 (position 4's 5 pairs with position 5's 9 → 14 → 4).

**palmax_id / palmin_id** (mirror-pair extremum) — Keep the larger (palmax) or smaller (palmin) of each
digit and its mirror partner (position i vs 9−i). The two are twins that between them recover the higher
and lower of every mirror pair. (Known as: morphological symmetrization.)
  Example: 3141592653 → palmax 3562992653, palmin 3141551413.

**seedsum_id** (spaced-seed sum) — Last digit of (this digit + the one just before it + the one three
positions back) — a windowed sum that deliberately SKIPS the digit two back, leaving a hole. Missing
look-backs at the start count as 0. (Known as: a spaced seed / lacunary FIR filter from bio-alignment.)
  Example: 3141592653 → 3458782300 (at position 3: 1 + 4 + 3 = 8, ignoring the digit two back).

## 5. Reversals, rotations & permutations
Move digits to new positions without changing their values. The multiset of digits is preserved; only the
order changes.

**riffle_id** (perfect card shuffle) — Split the ten positions into a first half (0–4) and second half
(5–9), then interleave them — new order 0,5,1,6,2,7,3,8,4,9. The first card stays first. (Known as: a
perfect out-shuffle / faro shuffle.)
  Example: 3141592653 → 3912461553 (3 then 9, 1 then 2, …).

**swapmm_id** (swap the biggest and smallest) — Find the position holding the row's largest digit and the
one holding its smallest, and swap just those two; everything else copies.
  Example: 3141592653 → 3941512653 (the 9 at position 5 trades places with the 1 at position 1).

**cgrev09_id** (reverse only if front beats back) — Compare the first digit with the last: if the first is
strictly larger, output the whole row reversed; otherwise copy. One comparison decides between two
whole-row behaviours.
  Example: 9141592650 → 0562951419 (first 9 > last 0, so the row is reversed end-to-end).

**rot9_id** (rotate right by one) — Shift the whole row one place right: every output equals its left
neighbour, and position 0 wraps around to take the last digit.
  Example: 3141592653 → 3314159265 (the trailing 3 wraps to the front).

**lagf3_id** (count-controlled look-back) — Count how many 3s have appeared up to and including the current
position; call it k. Output the digit sitting k places to the left (wrapping around the front). A running
count of 3s decides how far back to reach.
  Example: 3141592653 → 3314159266 (early digits copy with no 3 counted yet; each 3 shifts the read one
  further back, wrapping around).

**railfence_id** (zigzag-rail rescramble, ladder R=2,3,4) — Write the digits by bouncing down and back up
across R rails (rail sequence 0,1,2,1,0,1,2,… for R=3), then read them off one whole rail at a time, top
rail first. Only positions move. R dials how scrambled it gets. (Known as: the rail-fence transposition
cipher.)
  Example (R=3): 3141592653 → 3551196342 (top rail = positions 0,4,8 = 3,5,5; middle = 1,3,5,7,9; bottom
  = 2,6).

**pancake_id** (pancake-sort flip) — Find the biggest digit, then reverse the whole run from the start up
to and including it — bringing that biggest digit to the front, like sliding a spatula under the largest
pancake and flipping the stack above it. (Known as: a pancake-sort prefix flip.)
  Example: 3141592653 → 9514132653 (the 9 at position 5 → the prefix "314159" reverses to "951413"; the
  tail "2653" is left alone).

## 6. Sorting & comparison networks
Partially or fully sort by a fixed schedule of "compare two positions, put the smaller first" steps.

**oepass0_id** (one pair-sort pass) — Split the row into fixed adjacent pairs (0,1),(2,3),(4,5),(6,7),(8,9)
and sort each pair so the smaller digit is on the left — five independent comparators, one layer of a
sorting network. (Known as: one phase of an odd-even transposition sort.)
  Example: 3141592653 → 1314592635 (pair (3,1)→(1,3), (4,1)→(1,4), (5,3)→(3,5)).

**blkoe2_id** (two-pass pair sort, phase ladder) — Sort each fixed adjacent pair (0,1)(2,3)…, then re-pair
one slot over (1,2)(3,4)… and sort those. Two passes on offset groupings. The ladder (1 pass = oepass, 2,
…, up to a full sort) dials how many passes run. (Known as: a brick-wall / odd-even transposition sorting
network — fixed, non-overlapping pairs.)
  Example: 3141592653 → 1134529365 (pass 1 sorts each pair, pass 2 sorts the shifted pairs).

**tourney_id** (knockout bracket depth) — Run a single-elimination tournament: pair the digits, the higher
value in each match advances (ties → earlier position), repeat with the winners until one champion remains.
Output, for each digit, how many matches it won before being knocked out (0–4). With 10 digits an odd round
gives one entrant a free pass that turn, which doesn't count as a win. (Known as: knockout-bracket depth.)
  Example: 3141592653 → 1020040110 (the 9 wins all four matches → 4; the 4 at position 2 wins twice → 2).

## 7. Cellular automata & local-sweep rules  *(physics)*
A cellular automaton updates every position at once by a fixed LOCAL rule that looks at a cell and its
immediate neighbours; all updates use the ORIGINAL values (one synchronous sweep). Different rules model
different physics — smoothing, spreading, toppling, traffic. Each op here is a single sweep (not iterated
to a fixed point, which would be untrainable).

**ca-ladder** (three-neighbour sum sweep — a family, s = 1, 2, 3, 5) — In one sweep, replace every digit
with the sum of itself and its two neighbours, mod 10 (the row wraps at both ends; all positions update at
once from the pre-sweep values). Repeat s times; s dials how far information spreads. (Known as: an
additive cellular automaton, rule (1,1,1).)
  Example (s=2): 3141592653 → 6141186482 (one sweep → 7860567341-style row; a second identical sweep).

**med3_id + maj3_id** (median-of-3 / majority-of-3) — Both look at each digit with its left and right
neighbour (ends reuse the edge digit), updating all at once. **med3**: replace with the MIDDLE of the three
values — this erases lone spikes. **maj3**: if two of the three are equal, take that value; otherwise leave
the digit alone. They agree when a value repeats but split on an all-distinct trio. (Known as: a median /
rank-order filter, and a 1-D Ising / majority-voter automaton.)
  Example: med3 3141592653 → 3314556553 (position 2's trio 1,4,1 → middle 1). maj3 → 3111592653 (that same
  trio has a repeated 1 → 1; an all-distinct trio keeps the original).

**spile4_id** (sandpile topple, ladder threshold 4,5,6) — Any digit that is 4 or more "topples" in one
simultaneous sweep: it drops by 4 and hands 1 to each of its two neighbours (chips off the two ends are
lost). Every position uses its ORIGINAL value; the output stays in 0–7 with no wraparound. The threshold
dials how many digits are unstable. (Known as: the Bak–Tang–Wiesenfeld abelian sandpile / self-organized
criticality.)
  Example: 3141592653 → 3203264324 (the 4,5,9,6,5 each shed 4 to their neighbours).

**cyca_id** (a digit ticks up to chase a neighbour) — For each digit, look at its "next value" (one higher,
wrapping 9→0). If either neighbour already holds that next value, the digit ticks up by 1 to match;
otherwise it stays. Activity spreads like a wave. (Known as: the Fisch–Gravner–Griffeath cyclic cellular
automaton / a Greenberg–Hastings excitable medium.)
  Example: 3141592653 → 3141592663 (only position 8's 5 wants a 6, and its left neighbour is a 6, so it
  ticks 5→6).

**r184t5_id** (one-lane traffic step, ladder cutoff 4,5,6) — Read big digits (5 or more) as cars and small
digits as empty road. In one simultaneous step, every car with empty road immediately ahead moves forward
one slot (swapping with the gap); blocked cars stay. No digit is created or destroyed. The cutoff sets what
counts as a car. (Known as: Wolfram rule 184 / the TASEP traffic exclusion process.)
  Example: 3141592653 → 3141529635 (the 9 with a 2 ahead drives forward; the 5 at position 4 is blocked by
  the 9).

**blkpar_id** (swap mismatched fixed pairs) — Cut the row into fixed adjacent pairs (0,1)(2,3)… that never
overlap or shift. Swap a pair's two digits only when one is even and the other odd; leave both-even or
both-odd pairs alone. No value is created or destroyed. (Known as: a reversible Margolus-neighbourhood
block cellular automaton.)
  Example: 3141592653 → 3114592653 (only the (4,1) block, one even one odd, flips → (1,4)).

**signdiff_id** (sign-diffusion sweep) — Each digit takes one step toward the average of its two
neighbours: +1 if that average is higher, −1 if lower, unchanged if equal, clamped to 0–9. All positions
update at once from the original values; the ends use the digit itself as the missing neighbour. (Known as:
a discrete Allen–Cahn / sign-of-curvature relaxation.)
  Example: 3141592653 → 2232583544 (the 3 at position 0 steps down toward a lower neighbour-average; the 5
  at position 4 sits exactly at its neighbours' midpoint, so it holds).

## 8. Oscillators, relaxation & thermalization  *(physics)*
Physical systems that move toward balance. "Relaxation" = each value steps toward some average.
"Oscillators" = treat digits as clock hands or springs that nudge each other. "Thermalization" = a
conserved quantity (total heat) shuffles between hot and cold spots. Common thread: small local nudges.

**graviclump_id** (one step toward the row's average) — Compute the whole row's average, rounded to a whole
number; then move every digit ONE toward it — up by 1 if below, down by 1 if above, unchanged if equal (no
wraparound). (Known as: mean-field gravitational collapse / DeGroot consensus / Ornstein–Uhlenbeck
mean-reversion.)
  Example: 3141592653 → 4242483544 (average 3.9 rounds to 4; digits below 4 rise a step, above 4 fall a
  step, the 4 stays).

**entrain_id** (clock-hands toward the first hand) — Treat each digit 0–9 as a hand on a 10-hour clock.
Turn every hand ONE notch toward the direction of the FIRST digit's hand, along the shorter way round (stay
if already there). (Known as: injection locking / master-slave synchronization / entrainment to a
pacemaker.)
  Example: 3141592653 → 3232403543 (anchor 3 stays; 1→2 and 4→3 step toward it; 9→0 wraps the short way up
  toward 3).

**phaselock_id** (clock-hands toward the average) — Treat each digit 0–9 as a hand on a 10-hour clock. Turn
every hand ONE notch (mod 10) toward the AVERAGE direction of all ten hands, the shorter way round. (Known
as: the Kuramoto model of coupled oscillators.)
  Example: 3141592653 → 4232403544 (each digit steps one notch toward the group's mean clock direction).

**oscpos_id + oscmom_id** (a nudged spring's position and velocity) — Model a spring with a POSITION and a
VELOCITY, both starting at 0 on a 0–9 dial. Each digit is a push: every step the position advances by the
current velocity (mod 10), then the velocity is updated by the restoring pull plus that push (mod 10).
**oscpos** prints the position; **oscmom** prints the velocity — same dynamics, different readout. (Known
as: the Störmer–Verlet / leapfrog symplectic integrator; a phase-space oscillator.)
  Example: 3141592653 → oscpos 0345229962, oscmom 3117070767.

**heatx_id** (hottest gives one to coldest) — Copy the row unchanged except: the single largest digit drops
by 1 and the single smallest digit rises by 1 (first occurrence of each; if all digits are equal, nothing
changes). The row total is preserved. (Known as: microcanonical / Kac-model thermalization — conserved-
energy relaxation toward equipartition.)
  Example: 3141592653 → 3241582653 (the max 9 → 8 and the min 1 → 2; everything else untouched).

## 9. Signal processing (filters, envelopes, quantizers)  *(signal)*
Treat the row as a signal in time. Filters smooth or average it; envelopes track its recent peak with a
fading memory; quantizers snap it to a grid while carrying the rounding error forward.

**ewma2_id** (fading running total) — Keep a running register, but HALVE it (rounding down) before adding
each new digit, then take the last digit. Old contributions fade by half each step, so only recent digits
matter. (Known as: an exponentially-weighted moving average / leaky integrator / first-order IIR filter.)
  Example: 3141592653 → 3253623787 (at position 5 the register 6 halves to 3, plus digit 9 = 12 → 2).

**boxsum_id** (moving sum over the last k, ladder k=2,3,4,5) — Print the last digit of the sum of the
current digit and the k−1 before it — a sliding window adding up the most recent k digits. k dials the
window width. (Known as: a boxcar / moving-average / CIC comb FIR filter.)
  Example (k=3): 3141592653 → 3486056734 (position 5 sums 1,5,9 = 15 → 5).

**errdiff_id** (round to a grid, carry the leftover, ladder step 2,3,5) — Round each digit to the nearest
multiple of the grid step (default 3 → nearest of 0,3,6,9), but first add in the leftover rounding error
carried from the previous digit; then remember THIS digit's new leftover to pass on. No error is ever
dropped — it gets pushed forward and eventually corrects itself. (Known as: delta-sigma modulation /
Floyd–Steinberg error-diffusion dithering.)
  Example: 3141592653 → 3060690663 (the 1 rounds down to 0, leaving +1, which pushes the next 4 up to 5 →
  rounds to 6).

**zcross5_id** (count of half-switches) — Split digits into a low half (under 5) and a high half (5 or
more). Count how many times, scanning left to right, the stream flips from one half to the other; print
that running count. (Known as: the zero-crossing rate, a speech/audio voicing feature.)
  Example: 3141592653 → 0000112334 (the count bumps at each low↔high flip → 4 flips total).

**haar_id** (pair sum-and-difference) — Split the row into fixed pairs; turn each pair (a, b) into (a+b mod
10, a−b mod 10) — the pair's total and its gap. (Known as: the Haar wavelet / one-level discrete wavelet
transform / a Walsh–Hadamard butterfly.)
  Example: 3141592653 → 4253468682 (pair (5,9) → (14 mod 10 = 4, −4 mod 10 = 6)).

**envmax_id** (peak-hold with decay) — Print a running maximum whose memory fades: the running value is the
larger of the current digit and (the previous running value minus 1). A tall peak is held but leaks away by
1 each step until a bigger digit refreshes it. (Known as: a peak detector / envelope follower / automatic
gain control — equivalently a max-plus "tropical" scan with unit decay.)
  Example: 3141592653 → 3243598765 (after the 9, the later digits are all smaller, so the memory decays 9,
  8, 7, 6, 5).

## 10. Number theory
Operations built from divisibility, remainders, gcd, and digit arithmetic — the whole-number structure of
the digits rather than their size.

**pgcd_id** (gcd with the left neighbour) — Replace each digit by the greatest common divisor of it and the
digit to its left (the largest whole number dividing both); the first digit is kept. (Known as: the
pairwise Euclidean gcd.)
  Example: 3141592653 → 3111111211 (most neighbour pairs are coprime → 1; the 2 is gcd(6,2)).

**euclstep_id** (remainder of the left neighbour) — Replace each digit by the remainder when its left
neighbour is divided by it (if the current digit is 0, keep the left neighbour); the first digit is kept.
This is the single reduction step the gcd algorithm repeats. (Known as: the Euclidean step, (a,b)→(b, a
mod b).)
  Example: 3141592653 → 3010151212 (position 1: 3 ÷ 1 leaves 0; position 5: 5 ÷ 9 leaves 5).

**meanint_id** (whole-number running average) — Output 1 whenever the average of all digits seen so far is
a whole number (the running sum divides evenly by the count), else 0. (Known as: a Harshad / Niven
condition.)
  Example: 3141592653 → 1100000010 (sum 3 over 1 and sum 4 over 2 are whole; later, sum 36 over 9 = 4).

## 11. Group theory & symmetry  *(math)*
A "group" here is the set of symmetry moves of a small shape — the ways you can rotate/flip a triangle (6
moves), a square (8), a pentagon (10) and land it back on itself. Each digit names one move; you do the
moves in stream order and report where the shape ends up (or a coarse feature of it). Doing moves in
sequence is "multiplying in the group"; order matters (flip-then-rotate ≠ rotate-then-flip), which makes
these richer than plain addition.

**s3prod_id** (running 3-card-shuffle product) — Picture 3 cards in a row. There are 6 ways to rearrange
them; each digit (mod 6) names one rearrangement, applied in order to track where the cards now sit. Output
a number 0–5 naming the current arrangement. (Known as: the running product in the symmetric group S₃, a
non-abelian group.)
  Example: 3141592653 → 3401421142.

**d4prod_id** (running square-symmetry product) — Picture a square you can rotate and flip; 8 moves in all.
Each digit (mod 8) names one move; keep a running record of the square's orientation. Output 0–7. (Known
as: the dihedral group D₄ product.)
  Example: 3141592653 → 3045013721.

**q8prod_id** (running quaternion product) — There are 8 "signed units": 1, i, j, k and their negatives.
Each digit (mod 8) picks one; multiply them in order by the quaternion rule (i·j = k, j·i = −k, …); output
0–7 naming the running product. (Known as: the quaternion group Q₈ product.)
  Example: 3141592653 → 3263635365.

**cosetprod_id** (partially-hidden 3-card shuffle) — Shuffle 3 cards exactly as s3prod, but each step
report only a COARSER fact: which of 3 groups the current arrangement falls into (two arrangements that
differ merely by swapping a particular pair of cards count as the same group). The exact arrangement stays
hidden. Output 0, 1 or 2. (Known as: projection onto the cosets of a non-normal subgroup of S₃.)
  Example: 3141592653 → 0212002200.

**d4char_id** (running square-symmetry, coarse label) — Read the digits as moves that rotate/flip a square
(digit d uses move d mod 8) and track where it sits. But report only a coarse label: 2 if the square is
back exactly where it started, 8 (standing for −2) if it has been turned a half-turn (180°), 0 for every
other rotation and every flip. (Known as: the character of the 2-D irreducible representation of D₄.)
  Example: 3141592653 → 0200200080 (the running square lands back at start twice → two 2s, at a half-turn
  once → the 8).

**d5prod_id** (running pentagon symmetry) — Read the digits as moves on a regular pentagon: digits 0–4
rotate it (by that many fifths of a turn), digits 5–9 flip it. Keep a running "where the pentagon sits now"
and print it as 0–9. (Known as: the dihedral group D₅ product; it underlies the Verhoeff check-digit
scheme, which catches more transcription typos than a plain mod-10 checksum precisely because the moves
don't commute.)
  Example: 3141592653 → 3434630642.

**d5act_id** (running pentagon symmetry, applied to a corner) — Same running pentagon moves, but don't
print the symmetry itself. Instead take the symmetry accumulated from all EARLIER digits and use it to move
the current digit's own corner (corner = digit mod 5); print which corner it lands on (0–4). (Known as: the
action of D₅ on the pentagon's 5 vertices.)
  Example: 3141592653 → 3434420112.

## 12. Recurrences (Fibonacci-style scans)
Keep the last one, two, or three running totals and combine them each step (like Fibonacci, each number the
sum of the previous two). The "order" is how many past totals feed the next one.

**fibscan_id** (Fibonacci-style rolling sum) — Keep the last two running totals; each new total is (current
digit + the previous two totals) mod 10. (Known as: an input-driven Fibonacci / Lucas recurrence.)
  Example: 3141592653 → 3416271407.

**tribscan_id** (three-term rolling sum) — Like fibscan but keep the last THREE running totals; each new
total is (current digit + the previous three totals) mod 10. (Known as: an input-driven tribonacci /
order-3 linear recurrence.)
  Example: 3141592653 → 3419988124.

**contfrac_id** (running continued-fraction denominator) — Keep the last two running values; each new value
is (current digit × the previous value + the one before that) mod 10. Seeded so the first value is 1, this
reproduces the denominators of the continued fraction whose terms are the digits. (Known as: continued-
fraction convergent denominators.)
  Example: 3141592653 → 1156517329 (the true convergent denominators 1,1,5,6,35,321,… taken mod 10).
  *Implementation note:* the operator as implemented (`ContFrac` in `pipeline/models/algorithms.py`)
  seeds the two registers the other way round, yielding `3493492172` on this input, which is NOT the
  continued-fraction sequence; the description above is the intended reading.

**rotor3_id** (3-component linear rotor) — Keep three coupled running tallies u, v, w (all starting at 0).
Each new digit updates them in a chain — u += w + digit, then v += u, then w += v (each mod 10) — but only
u is ever shown; v and w stay hidden. (Known as: a 3-phase linear rotor / three coupled integrators.)
  Example: 3141592653 → 3742083905 (printing u; the two hidden tallies feed forward into it later).

## 13. State machines, latches & tiny machines
Carry a small hidden STATE (an on/off latch, a mode, an accumulator) that the current digit updates by a
fixed rule, and print something about it. Includes tiny "computers" where each digit is an instruction.

**maxgate_id** (running-max flips a rule switch) — Carry one on/off switch. Each time a digit sets a new
running maximum the switch flips, and its current position decides which of two multiply-then-add rules
(y = a·x+b mod 10) applies to that digit. (Parameters illustrative: OFF = (x+2) mod 10, ON = (3x+1) mod 10.)
  Example: 3141592653 → 0463614875 (records at 3,4,5,9 flip the switch; each digit uses whichever rule is
  active).

**prevsegmax3_id** (max of the previous chunk) — Cut the row into segments at every 3. Output the maximum
of the most recently COMPLETED segment, held steady until the next 3; before the first 3, output 0. When a
3 arrives, the just-finished segment's max latches into the output and the accumulator resets.
  Example: 4132539137 → 0044455599 (segment [4,1] max 4 shows until the next 3; then [2,5] max 5; then
  [9,1] max 9).

**sanc2_id** (parity-latch-routed anchor map) — Keep a one-bit switch that flips every time an odd digit is
seen (tracking whether an even or odd number of odds has appeared). That switch selects one of two rules
y = (current digit + m·(first digit) + b) mod 10.
  Example (first digit 3): 3141592653 → 0148569350 (the leading odd 3 flips the switch → rule (5,2): (3 +
  5·3 + 2) mod 10 = 0).

**dfa3t010020001_id** (3-state machine) — Run a machine with three states (0,1,2), starting at 0, reading
digits left to right. A low digit (0–2) resets it to 0; a mid digit (3–6) advances it one step, wrapping
2→0; a high digit (7–9) steps it down one, not below 0. Output the state after each digit. (Known as: a
small deterministic finite automaton.)
  Example: 3141592653 → 1010100120 (mid 3 → state 1, low 1 → reset to 0, …; the high 9 steps state 1 down
  to 0).

**bmcand_id** (majority front-runner) — Hold one digit as the current "leader" plus a score. Seeing the
leader again adds 1 to the score; a different digit subtracts 1; at zero the next digit becomes the new
leader. Always print whoever you are currently holding. (Known as: the Boyer–Moore majority-vote
algorithm.)
  Example: 3141592653 → 3344552255 (the held leader flips as the score is knocked down and re-seeded).

**hold2ge5_id** (second-most-recent big digit) — Watch for "big" digits (5 or more). At each position print
the value of the SECOND-most-recent big digit seen; print 0 until two big digits have appeared. A two-slot
delay line: the newest big value is stored but not yet emitted, the one before it is what prints.
  Example: 3141592653 → 0000055966 (the big digits are 5,9,6,5; once 9 arrives the older 5 prints, then 9,
  then 6).

**gapge5_id** (spacing of the last two big digits) — Watch for digits 5 or more and remember the positions
of the two most recent. Print the gap between those positions; print 0 until two big digits have been seen.
hold2ge5's twin that tracks WHERE rather than WHICH value.
  Example: 3141592653 → 0000011211 (big digits at positions 4,5,7,8; the last-two spacing is 1,1,2,1).

**holdodd_id** (hold the latest odd digit) — Remember and print the most recent odd digit seen; print 0
until the first odd appears. An odd digit updates the held value; an even digit keeps showing the last odd
one. (Known as: a sample-and-hold latch triggered by odd values.)
  Example: 3141592653 → 3111599953 (after the 9, the even 2 and 6 keep printing 9 until the next odd, 5).

**addlastodd_id** (add the latest odd digit in) — Hold the most recent odd digit (as holdodd) and add it to
the current digit, mod 10; nothing is added until the first odd appears.
  Example: 3141592653 → 6252081506 (the held odd is 9 across positions 5–7, so the 2 and 6 become 1 and 5;
  the leading odd 3 doubles to 6).

**togsum0_id** (two totals, a 0 flips which one prints) — Keep two running totals: one of the even digits,
one of the odd. Normally print the even total's last digit, but every 0 in the stream toggles which total
is printed (a 0 never changes either total). (Known as: an event-latched readout selector.)
  Example: 3140592053 → 0044988666 (the first 0 switches the printout to the ODD lane; the second switches
  it back).

**signsum0_id** (running total whose sign a 0 flips) — Keep one running total. Digits normally add; every 0
flips the direction so digits SUBTRACT until the next 0, which flips it back. Print the total's last digit.
  Example: 3140592053 → 3488342270 (after the 0 at position 3 the 5 and 9 are subtracted, not added).

**schmitt_id** (two-threshold on/off latch) — A switch that stays ON (prints 9) until a digit of 3 or less
arrives (turning it OFF, prints 0), and stays OFF until a digit of 6 or more arrives (turning it ON).
Digits 4 and 5 are a dead-band that leave it unchanged. Starts OFF. (Known as: a Schmitt trigger /
hysteresis comparator / bang-bang controller.)
  Example: 3141592653 → 0000090990 (the 9 flips it ON; the 2 (≤3) flips it OFF; the 6 flips it back ON;
  the 5 is dead-band → stays ON).

**vm1_id** (digit-as-opcode register machine) — Treat each digit as an instruction to a one-number
calculator (its running accumulator): digits 0–4 load that value; 5 adds one; 6 subtracts one; 7 doubles;
8 replaces the value with nine-minus-it; 9 holds. Print the accumulator after each instruction, mod 10 —
so the input is itself the program. (Known as: a one-instruction register machine / bytecode interpreter.)
  Example: 3141592653 → 3141222123 (loads 3,1,4,1; then 5→2, 9 holds at 2, 2 loads 2, 6→1, 5→2, 3 loads 3).

**skipn5_id** (predicated-skip flip-flop) — Copy each digit, but whenever you emit a digit that is 5 or
more, the very next position is blanked to 0. A blanked position emits 0, and since 0 is under 5 it can't
itself trigger the next blank. (Known as: control-flow predication / a branch-delay slot.)
  Example: 3141592653 → 3141502603 (the emitted 5 blanks the next 9; the emitted 6 blanks the next 5).

## 14. Bit-level (operating on the binary of a digit)
Work on the BITS of each digit (0–9 is four binary bits) rather than its numeric value. The rest of the
catalog only ever XORs bits; this family reads binary structure directly.

**bitmaj_id** (bit-plane majority) — Read each digit as 4 binary bits. For each of the 4 bit-positions
separately, take a running majority vote of that bit across all digits so far (ties → 0), then reassemble
the 4 winning bits into a digit, mod 10. (Known as: bit-plane-parallel majority voting.)
  Example: 3141592653 → 3111111111 (after the first digit, every bit-plane's running majority settles so
  the reassembled digit stays 1).

**bittv_id** (bit-transition count) — Running total, mod 10, of how many bits flip between each pair of
adjacent digits (the Hamming distance of neighbours). It starts at 0 and accumulates. (Known as: cumulative
switching activity.)
  Example: 3141592653 → 0135681246 (3→1 flips 1 bit, 1→4 flips 2, 4→1 flips 2, … each added to the total).

**bittrans_id** (cross-digit bit transpose) — Lay the row out as a 10×4 grid of bits (10 positions down,
each digit's 4 bits across), transpose it to 4×10, read it back row by row, and regroup into 10 new digits.
Every output digit is packed from single bits gathered across four different input positions. (Known as: a
SIMD / packet bit-transpose.)
  Example: 3141592653 → 1370149180 (a pure rewiring of individual bits — verified NOT to be any
  rearrangement of the whole digits).

**dupxor_id** (multiplicity-parity involution) — Count how many times each value occurs across the whole
row. Then flip the lowest bit (the ±1 that swaps even↔odd) of every digit whose value occurs an ODD number
of times; leave the rest. On an all-distinct row it is its own inverse (applying it twice restores the
row — bit-flips just permute the distinct values), but NOT in general: flipping bits changes which
values have odd counts, so a second pass need not undo the first. (Known as: a multiplicity-parity
flip.)
  Example: 3141592653 → 3151583753 (the values 4,9,2,6 each appear once → those digits flip to 5,8,3,7;
  the 3s, 1s, 5s each appear twice → untouched).

## 15. Coding, ciphers & error-correction
Reversible scrambles (ciphers), redundancy that detects/fixes a flipped digit (error-correction), and
block transforms borrowed from cryptography — each with a clear encode/decode structure.

**hamfix_id** (fix one flipped bit) — Take the low bit (x mod 2) of a block of digits and read those bits
as a Hamming code block: recompute the parity checks, and if they point at one position, flip that single
bit back — locating and correcting one bit error. (Parameters illustrative: seven low bits form a
Hamming(7,4) block.) (Known as: Hamming single-error correction / syndrome decoding.)
  Example (low-bit plane): received bits 1110010 → the checks point at position 6 → flip it → 1110000.

**convfix_id** (single-error-correcting decoder) — Read each digit's two low bits as one step of a
transmitted code: the lowest bit is a data bit, the next a check bit. Recompute a "syndrome" per step (the
check bit XORed against the current and previous data bits); wherever two consecutive syndromes are both 1,
a single bit-flip is localized and corrected. Output the corrected data-bit stream, delayed one step.
(Known as: threshold/Viterbi decoding of a rate-1/2 convolutional code; a clean transmission changes
nothing — a built-in invariance control.)
  Example: 3141592653 → 0111111010.

**vig3_id** (shift by a repeating secret key, ladder key-length 2,3,5) — Add a repeating fixed key to the
digits, position by position, mod 10: position i is shifted by key[i mod L] (default key 3,1,4). For L=3
every third position gets the same shift. The key-length dials how long before the pattern repeats. (Known
as: the Vigenère periodic-key cipher.)
  Example: 3141592653 → 6284635796 (shifts 3,1,4 repeat: 3+3=6, 1+1=2, 4+4=8, then 1+3=4, …).

**blkadd_id** (add left into right, in fixed pairs) — Cut the row into fixed adjacent pairs (0,1)(2,3)…. In
each pair leave the left digit untouched and replace the right with (left + right) mod 10. Reversible —
subtracting the left back recovers the original. (Known as: a Feistel controlled-add / reversible
block-cipher half-round; a CNOT analog.)
  Example: 3141592653 → 3445542858 (pair (5,9) → left 5 kept, right becomes 14 → 4).

## 16. Cache, scheduling & control regulators
Systems that manage a limited resource over time: a small cache that evicts old entries, a scheduler that
picks who runs next, a regulator that grants/denies requests to keep a level in bounds.

**lru3_id** (3-slot least-recently-used cache) — Maintain a cache of the 3 most-recently-used distinct
values. For each digit output 1 if it is already cached (a hit) or 0 if not (a miss); on a miss, if the
cache is full, evict the value unused longest, then insert the new one. (Known as: an LRU cache, capacity
3.)
  Example: 3141592653 → 0001000000 (only the second 1 is still cached when it returns; the earlier repeats
  were evicted).

**tbucket_id** (token-bucket rate limiter) — Keep a refillable budget of tokens. Each step the bucket first
refills a fixed amount (capped), then the request is granted (output 1) if enough tokens remain, which
spends them, or denied (0) if not — grants feed back into the level. (Parameters illustrative: capacity 9,
refill +2/step, cost = the digit.) (Known as: the token-bucket rate limiter.)
  Example: 3141592653 → 1111101101 (small requests are granted and drain the bucket; the costly 9 and a
  later 5 are denied).

**aimd8_id** (rise-slow, fall-fast counter) — Keep a counter that climbs by 1 most steps but is halved when
a congestion event hits — additive increase, multiplicative decrease. (Parameters illustrative: a digit ≥8
is the congestion event.) (Known as: AIMD, the rule behind TCP congestion control.)
  Example: 3141592653 → 1234523456 (the counter ramps 1..5, the 9 halves it to 2, then it ramps again).

**rrgrant3_id** (round-robin arbiter) — Three requesters share one grant per step; the grant rotates among
them in fixed turn order, so each step names whose request is served. (Parameters illustrative: all three
always requesting, so the grant just cycles 0,1,2.) (Known as: round-robin arbitration.)
  Example: 3141592653 → 0120120120 (grants cycle through requesters 0,1,2).

**sjf1_id** (one-slot shortest-job-first) — A single waiting slot holds the largest digit seen so far. At
each new digit, send out whichever is smaller — the stored value or the new digit — and keep the larger.
Equivalently, output min(largest-so-far, current digit). (Known as: shortest-job-first scheduling with a
1-slot buffer.)
  Example: 3141592653 → 3131452653 (a new record like the 4 gets buffered and the smaller stored max
  emits).

**sjf2_id** (two-seat waiting room, smallest served) — A waiting room holding up to two digits. Each new
digit walks in; once a third would overflow, the smallest of the three present is ejected and printed;
while the room is still filling, print 0. Equivalently, each output is the smaller of the current digit and
the second-largest seen so far. (Known as: shortest-job-first with a 2-slot buffer.)
  Example: 3141592653 → 0011342553.

**qlen_id** (queue length that can't go negative, ladder threshold 3,4,5,6) — Read each digit as an arrival
(4 or more → queue +1) or a departure (under 4 → queue −1), but the length can never drop below 0 or above
9. Print the current length. A run of departures on an empty queue just stays at 0. The threshold dials how
often digits count as arrivals. (Known as: the Lindley recursion / an M/M/1 queue length / a reflected
random walk.)
  Example: 3141592653 → 0010121232 (early small digits pin it at 0; the 5 pushes it to 1, the 9 to 2, then
  it drifts).

## 17. Suppression / veto (compute-then-cancel)
The output has a "default" action that runs everywhere, plus an exception that must ACTIVELY CANCEL it at
certain spots. Interesting because the model has to compute the thing it then suppresses.

**recval_id** (print only new record highs) — Output the digit whenever it sets a new running maximum, and
0 everywhere else. The record-detecting machinery runs on every digit; non-records are cancelled to 0.
  Example: 3141592653 → 3040590000 (3, 4, 5, 9 are successive record highs; everything else is zeroed).

**dualtie_id** (erase ties with the running extremes) — Copy each digit, except output 0 whenever it
exactly equals the largest OR the smallest digit seen so far (strictly beating either still copies).
Position 0 always copies.
  Example: 3141592653 → 3140592653 (the second 1 ties the running minimum → erased to 0; nothing else ties
  an extreme).

**pmaxv9_id** (running max with 9s disqualified) — Output the running maximum, but 9s never count — once a
9 appears it is permanently ignored, so the reported max is the biggest non-9 seen so far (0 if none). The
9-max is computed only to be cancelled.
  Example: 3141592653 → 3344555666 (the 9 is skipped, leaving the max at 5 before the later 6 lifts it).

**pmaxv-dose family (pmaxv89_id, pmaxv789_id)** (running max with banned values) — Same as pmaxv9 but a SET
of values is barred from ever counting as the maximum. The family holds the trigger fixed and dials how
much content must be cancelled: pmaxv89 bars {8,9}, pmaxv789 bars {7,8,9}.
  Example (pmaxv89): 3841592653 → 3344555666 (the 8 and 9 are ignored, so the max climbs only through the
  allowed digits).

**cap59_id** (circuit breaker at the fifth high digit) — Copy digits through until the count of "high"
digits (5 or more) reaches five; from that point output 9 at every remaining position. (Known as: a
saturating circuit-breaker.)
  Example: 5267189435 → 5267189999 (the fifth high digit trips the breaker so every later output is 9).

**majmask_id** (zero the minority parity) — Track how many even and how many odd digits have been seen
(including the current one). Output 0 for any digit whose parity is the strict minority at that moment; copy
the rest.
  Example: 3141592653 → 3101590053 (the even 4, 2, 6 arrive while evens are outnumbered → zeroed).

**pmrep_id** (parity-step, vetoed by a repeat) — By default add 1 to an even digit and subtract 1 from an
odd digit (mod 10). The exception: if a digit exactly equals its left neighbour, copy it raw instead.
Position 0 always takes the step map.
  Example: 7725551883 → 6734550982 (the repeated 7, 5s, and 8 copy unchanged; everything else steps).

**ebud3_id** (censor the first three big digits) — Copy every digit, except the first three "big" digits (5
or more) are blanked to 0. From the fourth big digit onward the censoring stops — an eraser that runs out
of budget.
  Example: 3141592653 → 3141002053 (the big digits are 5,9,6,5; the first three are zeroed, the fourth
  survives).

**tieswap_id** (tie flips to the opposite extreme) — Copy each digit, unless it exactly equals the
largest-so-far, in which case print the smallest-so-far instead — and if it equals the smallest, print the
largest. Only ties with a running extreme are redirected.
  Example: 3141592653 → 3144592653 (only position 3's 1 ties the earlier low and is swapped up to the
  earlier high 4).

**riseerase_id** (erase upticks) — Copy each digit, except erase to 0 any digit larger than the one
immediately to its left. The first digit always copies.
  Example: 3141592653 → 3101002053 (the 4, 5, 9, 6 each rise above their left neighbour and are zeroed).

**risemax_id** (uptick prints the running max) — riseerase's twin: any digit larger than its left neighbour
prints as the largest digit seen so far (including itself) instead of its own value; the rest copy.
  Example: 3141592653 → 3141592953 (the 6 at position 7 rises above the 2 and prints the running max 9).

**riskeep_id** (keep only the upticks) — The exact complement of riseerase: keep a digit only if it is
larger than its left neighbour, erase everything else to 0 (position 0 is 0). riskeep + riseerase gives
back the original row digit-for-digit.
  Example: 3141592653 → 0040590600 (only the rises 4, 5, 9, 6 are kept).

**delc2_id** (delayed censor) — Copy every digit, but each "big" digit (5 or more) forces the output TWO
positions later to 0. The big digit itself copies; the cancellation lands downstream at a fixed offset of
two.
  Example: 3141592653 → 3141590050 (the big digits at positions 4,5,7 zero out positions 6,7,9).

**mixtie_id** (two tie-rules in one) — Copy each digit, with two exceptions: a digit that ties the
largest-so-far prints 0, and a digit that ties the smallest-so-far prints the largest-so-far. Fuses
tieswap's min-tie branch and a blank-on-max-tie branch.
  Example: 1345564143 → 1345064643 (position 4's 5 ties the earlier high → 0; position 7's 1 ties the
  earlier low → prints the running high 6).

**refrac_id** (neural refractory veto) — Copy each digit, but a big digit (7, 8, or 9) briefly silences the
small digits right after it, the silencing wearing off step by step. A following digit is set to 0 only
while its value plus the number of steps since the last big digit stays under 5 — so right after a big
digit even a 3 is silenced, a couple of steps later only smaller digits are, and soon nothing is. (Known
as: a neuron's relative refractory period / afterhyperpolarization.)
  Example: 3141592653 → 3141590653 (the 9 silences the 2 right after it — 2 + gap 1 = 3 < 5 — but the
  following 6, 5, 3 have recovered enough to survive).

**mexhat_id** (lateral-inhibition veto) — Copy each digit, except silence (set to 0) any digit sitting in a
sharp valley — at least 2 below the midpoint of its two immediate neighbours. Row-ends are always copied.
(Known as: center-surround lateral inhibition in the retina / the Marr–Hildreth Laplacian-of-Gaussian edge
detector, a [−1, 2, −1] kernel.)
  Example: 3141592653 → 3040590653 (the 1 between 3 and 4 has midpoint 3.5, and 3.5 − 1 = 2.5 ≥ 2 →
  silenced; likewise the other deep dips).

## 18. Data structures in the stream (stacks, sketches, dictionaries, seen-sets)
The hidden state IS a data structure — a stack, a hash-bucket count, a value→label dictionary, a set of
values seen so far — maintained as the row streams by and read out per position.

**parmatch_id** (bracket partner-matching) — Treat two chosen digits as open and close brackets and keep a
stack: push each open bracket's position; when a close bracket arrives, pop and report its matching open's
position. Non-brackets and unmatched closes output 0. (Parameters illustrative: 1 = open, 2 = close, output
= the matched open's index.) (Known as: balanced-parenthesis matching with a stack.)
  Example: 1132122400 → 0001040000 (opens at positions 0,1,4; the close at position 3 pops the top open,
  position 1; the close at position 5 pops the open at position 4).

**hb3cnt_id** (hash-bucket collision count) — Hash each digit into a small number of buckets and, at each
position, output how many earlier digits fell into the same bucket. (Parameters illustrative: hash = digit
mod 3.) (Known as: hash-bucket collision counting / a counting sketch.)
  Example: 3141592653 → 0012011223 (the final 3 shares bucket 0 with the earlier 3, 9, 6 → count 3).

**pfetch0_id** (one-hop pointer fetch) — Read the first digit as an address, look up the digit stored at
that position, and place it in slot 0; every other slot copies. A single indirect load. (Known as: a
one-hop pointer dereference, y[0] = x[x[0]].)
  Example: 3141592653 → 1141592653 (first digit 3 addresses position 3, which holds 1, so slot 0 becomes 1).

**indecho_id** (induction: echo what followed last time) — For each digit, find its most recent earlier
occurrence and output the digit that came right after it there; if it has never appeared before, copy it.
Content-addressed lookahead: match on value, then fetch the successor. (Known as: the transformer induction
head, as a whole task.)
  Example: 3141592653 → 3144592691 (the second 1 recalls the first 1, whose successor was 4, so it outputs
  4).

**segdist3_id** (distinct count since the last landmark) — Count how many distinct values have appeared
since the most recent 3, growing a seen-set as new values arrive; each 3 wipes the seen-set and outputs 0.
  Example: 4132539137 → 1201201201 (4 →1 distinct, 1 →2; the 3 resets to 0; and so on).

**pge_id** (nearest earlier larger digit) — For each digit, output the value of the closest digit to its
left that is strictly larger than it, or 0 if no earlier digit is bigger ("who is blocking me from the
left"). (Known as: the previous-greater-element / monotonic-stack primitive.)
  Example: 3141592653 → 0304009965 (the 2 at position 6 looks back and the nearest bigger digit is the 9).

**perase0_id** (pointer-erase) — Read the first digit as an address; copy the whole row unchanged except
write 0 at the position it names.
  Example: 3141592653 → 3140592653 (first digit 3 → position 3 erased to 0).

**bcast0_id** (broadcast the first digit) — Output the first digit at every position — a pure copy of x[0]
across the whole row.
  Example: 3141592653 → 3333333333.

**chain-ladder** (pointer chasing — a family, d = 1, 2, 3) — From each starting position, repeat d times:
move to the position whose number is the digit currently under you; after d moves, output the digit you
land on. d is the number of address hops. (Known as: linked-list / pointer-chasing traversal.)
  Example (d=2): 3141592653 → 1191315431 (from position 2: digit 4 → jump to position 4, digit 5 → jump to
  position 5, digit 9 → output 9).

**firstuniq_id** (first non-repeater so far) — At each position, print the earliest digit that has appeared
exactly once up to here; print 0 if every digit so far has repeated. As the front-runner gets a duplicate
it is dropped and the next still-unique digit takes over. (Known as: the "first non-repeating character"
problem.)
  Example: 3141592653 → 3333333334 (the leading 3 stays earliest-unique until the final 3 repeats it, then
  the 4 takes over).

**twosum10_id** (complement-seen flag) — For each digit, print 1 if some earlier digit combines with it to
make a multiple of ten (an earlier digit equals ten-minus-this-one, with two 0s also counting), else 0.
(Known as: the two-sum problem, targeting a sum of ten.)
  Example: 3141592653 → 0000010110 (the 9 finds an earlier 1, the 6 an earlier 4, the 5 an earlier 5).

**runmex_id** (smallest digit not yet seen) — Output the smallest digit 0–9 that has NOT appeared anywhere
up to and including the current position (capped at 9). (Known as: the mex / minimum excludant of
Sprague–Grundy theory.)
  Example: 3041592653 → 0112226777 (once 0,1,2 have all appeared the answer jumps to 3, then climbs).

**nimhist_id** (Nim-value of the digit tallies) — Keep a tally of how many times each digit has appeared so
far; output the bitwise XOR of those ten tallies, mod 10. (Known as: the Sprague–Grundy Nim-value of the
digit-count heaps.)
  Example: 3141592653 → 1012323212.

**cmin_id** (count-min sketch) — A tiny approximate frequency counter. Keep two banks of four counters. Two
fixed hash rules send each digit to one counter in each bank; bump both, then output the smaller of those
two counts. Because different values can share a counter, the output can over-estimate. (Known as: the
Count-Min sketch, a streaming probabilistic data structure.)
  Example: 3141592653 → 1112341252 (with hashes "value mod 4" and "(value+1) mod 4"; where two values
  share a counter the count runs high). The exact output depends on the freely-chosen hash functions, so
  this example is illustrative — a different hash pair gives a different (equally valid) sketch.

**intern_id** (first-appearance dictionary coder) — Relabel each digit by the order in which its value
first appeared: first distinct value → 1, second → 2, … (wrapping mod 10); a repeat reuses the label its
value already earned. (Known as: a first-appearance dictionary coder — pandas.factorize / the "abba"
word-pattern / LZW indices.)
  Example: 3141592653 → 1232456741 (3→1, 1→2, 4→3, the next 1 reuses 2, 5→4, …).

**nbrdeg_id** (value-proximity degree) — For each digit, count how many other digits anywhere in the whole
row have a value within 1 of it (itself not counted). (Known as: the degree in an ε-neighbourhood graph — a
DBSCAN core-point / kernel-density count.)
  Example: 3141592653 → 3242304233 (the 3 at position 0 finds one 2, the other 3, and one 4 → 3; the 9 has
  nothing within 1 → 0).

## 19. Multi-step compositions & bridges
Two or three operators chained: run the first on the row, then the second on that result, etc. The
intermediate ("bridge") value is computed but never printed.

**dblodd_walk10_cmpsh1_id** (match-bit → signed walk → double-odds) — Three steps: (1) mark 1 where a digit
equals its left neighbour, else 0; (2) run a tally over those marks, +1 on a match and −1 otherwise, mod
10; (3) double every odd tally, giving an all-even output.
  Example: 7725551883 → 8088808888.

**dblodd_psum_novel_id** (novelty-bit → distinct-count → double-odds) — (1) mark 1 the first time each
value appears; (2) running count of distinct values, mod 10; (3) double the odd counts. All-even output.
  Example: 3141592653 → 2266406444 (distinct-count 1,2,3,3,4,5,6,7,7,7 → doubling odds 1,3,5,7 → 2,6,0,4).

**reset1_cmpsh1_id** (positions since the last neighbour-match) — (1) mark where a digit equals its left
neighbour; (2) count positions since the most recent such match (reset to 0 at each match, else increment).
  Example: 7725551883 → 0012001201.

**reset1_novel_id** (staleness clock) — (1) mark the first time each value appears; (2) count positions
since the last never-before-seen digit (reset to 0 at each new value, else increment).
  Example: 3141592653 → 0001000012.

**holdge5_diffsh1_id** (last big neighbour-jump) — (1) take neighbour differences (each digit minus its
left neighbour, mod 10; first kept); (2) hold onto the most recent difference that was 5 or more and output
it (0 before any big difference).
  Example: 3141592653 → 0887777798 (differences 3,8,3,7,4,4,3,4,9,8; the register latches 8, then 7, then
  9 and 8).

**dblodd_walk10_recordind_id** (record-marks → signed walk → double-odds) — (1) mark 1 at each digit
strictly bigger than everything before it (position 0 counts); (2) tally +1 on a 1 and −1 otherwise, mod
10; (3) double the odd tallies. The record-mark can be recomputed on the spot, but its running tally
cannot.
  Example: 3141592653 → 2020222088.

**holdge4_gapsp_id** (hold the last long echo) — Inner: for each digit, measure how far back the same value
last occurred (0 if new). Outer: keep emitting the most recent look-back distance that was 4 or more; 0
until one occurs.
  Example: 3141592653 → 0000000049 (distances 0,0,0,2,0,0,0,0,4,9; only the last two reach 4).

**pmax_gapsp_id** (longest echo so far) — Same look-back-distance inner (how far back each value last
repeated, 0 if new), but the outer stage takes the running MAXIMUM of those distances.
  Example: 3141592653 → 0002222249 (distances …,2,…,4,9; running max climbs 2 → 4 → 9).

**psum_gapsp_id** (echo distances, summed) — Same look-back-distance inner, but the outer stage keeps a
running TOTAL of the distances, mod 10.
  Example: 3141592653 → 0002222265.

**dblodd_holdge4_gapsp_id** (hold the last long echo, then double-odds) — The look-back distance of each
digit (0 if new) → hold the most recent distance that was 4 or more → double it if it is odd. holdge4_gapsp
with a final double-the-odds pass.
  Example: 3141592653 → 0000000048 (the held distances end 4,9; doubling the odd 9 gives 8).

**pmax_cwalk_id** (clamped-score high-water mark) — A running score starts at 0. Each step it changes by
(current digit − 5) but is kept inside 0–9 (can't go below 0 or above 9). Print the highest score the walk
has reached so far. (Known as: Kadane's maximum-subarray, with a saturating score.)
  Example: 3141592653 → 0000044444 (the score stays at 0 through the low digits, jumps to 4 at the 9, and
  that 4 remains the high-water mark).

## 20. Multi-task routers (mux)
The first token is a COMMAND that picks which of several per-digit tasks to run on the rest of the row
(the payload). Written `muxN.taskA.taskB…`; the model must read the command and route, and (interestingly)
may compute a wrong task's answer and then silence it. Examples show one selected member on the payload;
`pmstep` = add 1 to each even digit, subtract 1 from each odd; `sr12` = a latch that turns ON at each 1 and
OFF at each 2; `dblodd` = double the odd digits; `fixedpoint` = 1 where a digit equals its position, else 0.

**mux3.pmstep-id.dblodd-id.tmap5-id** — The command picks one of three memoryless digit maps: parity-step
(pmstep), double-the-odds (dblodd), or fold-the-high-half (tmap5: digit ≥5 → +5 mod 10). Nothing shared but
the router and a copy backbone.
  Example (pmstep member): [t0]3141592653 → 2050483742.

**mux4.pmstep-id.sr12-id.drot0-id.cmpswap0p9-id** — Four unrelated tasks: parity-step; the sr12 on/off
latch; rotate the whole row left by the first digit's value (drot0); compare first and last digit and swap
if the first is larger (cmpswap0p9).
  Example (pmstep member): [t0]3141592653 → 2050483742 (for reference the others give sr12 → 0111110000,
  drot0 → 1592653314, cmpswap0p9 → 3141592653).

**mux3.cmpswap0p9-id.dblodd-id.anc0-id** — Three tasks spanning copy / transform / anchor: compare-and-swap
the ends (mostly a copy); double the odds; subtract the first digit from every digit (anc0).
  Example (dblodd member): [t1]3141592653 → 6242082606.

**mux3.pmstep-id.sr12-id.drot0-id** — Built to test the "silencing" theory: parity-step (always active, so
it must be silenced when not selected), the sr12 latch (cheap held state, weakly silenced), and rotate-by-
first-digit (predicted NOT to need silencing — gated at the router, not computed then cancelled).
  Example (pmstep member): [t0]3141592653 → 2050483742.

**mux3.psum-novel-id.pmstep-id.dblodd-id** — Three tasks, one genuinely two-step: running count of distinct
values (psum_novel), parity-step, and double-the-odds — a light/light/heavy mix.
  Example (count-distinct member): [t0]3141592653 → 1233456777.

**mux3×moe kanc2 cross** — Three tasks where only one needs dedicated experts: the class-routed anchor map
kanc2 (even/odd picks one of two rules y = digit + m·(first digit) + b), double-the-odds, and fold-the-high-
half.
  Example (kanc2 member, first digit 3): [t0]3141592653 → 0848262620.

**mux2 shared-feature ladder (mux2.pmstep-id.fixedpoint-id / .pmstep-id.dblodd-id / .sr12-id.sr21-id)** —
Three two-task routers built so the members share progressively more machinery: rung A pairs parity-step
with the position-match flag (nothing shared); rung B pairs parity-step with double-the-odds (both branch
on even/odd); rung C pairs two mirror-image latches, sr12 (on-at-1/off-at-2) and sr21 (the reverse).
  Example (rung A, position-match member): [t1]3141592653 → 0100000000.

**mux4.fixedpoint-id.sr12-id.dblodd-id.fclamp0-id** — Four tasks chosen so each pair disagrees on a
different known fraction of positions: the position-match flag, the sr12 latch, double-the-odds, and raise
every digit up to at least the first digit's value (fclamp0).
  Example (double-odds member): [t2]3141592653 → 6242082606.

**mux3.pmstep-id.pmstep-id.dblodd-id** — Three command tokens but only two distinct tasks: two cues both
run parity-step and the third runs double-the-odds — a test of whether the router treats two synonymous
commands as one action.
  Example: [t0]/[t1] → 2050483742; [t2] → 6242082606.

**mux2.pmstep-id.sr12-id** — The smallest two-task router: one command bit chooses parity-step or the sr12
latch.
  Example: [t0] → 2050483742; [t1] → 0111110000.

**mux2.pmstep-id.id** — One command bit chooses parity-step or "do nothing" (copy the row). Since
parity-step moves every digit, the two disagree everywhere, so any leak of the map on a copy row is caught
immediately.
  Example: [t0] → 2050483742; [t1] → 3141592653.

**madd-ring ladder (mux2.madd3.madd7 ⊂ mux3.madd3.madd5.madd7 ⊂ mux5.madd1…madd9)** — Every command means
"add K to every digit (mod 10)" for a different fixed K; the ladder grows the menu of Ks. The router could
branch to N separate adders or build one adder and feed the command in as the number to add.
  Example (add-3 cue): 3141592653 → 6474825986; (add-7, its exact inverse): → 0818269320.

**mux4.madd3-id.madd7-id.mmul3-id.mmul7-id** — Four whole-row maps from two families: add 3, add 7,
multiply by 3, multiply by 7 (mod 10; each family's pair is an exact inverse). Adding can share one adder;
multiplying needs the command combined with the digit.
  Example (add-3 cue): 3141592653 → 6474825986; (×3 cue): → 9323576859.

**mux2.pmstep-id.c5L10** — One command runs parity-step; the other ignores the row entirely and prints
5555555555 no matter the digits (a payload-blind constant member).
  Example: [t0] → 2050483742; [t1] → 5555555555.

**mux2.rcnt1-cmpsh1-id.reset1-cmpsh1-id** — Both commands watch the same hidden event ("this digit equals
the one before it"); one prints a running COUNT of such matches, the other how many STEPS since the last
match. A test of whether one detector feeds two readouts.
  Example: (count cue) 7725551883 → 0111233344; (time-since cue) → 0012001201.

**mux2.dblodd-id.dblodd-diffsh1-id** — Both commands end in double-the-odds, over different inputs: one
doubles the odds of the row itself, the other first replaces the row by its neighbour-differences and then
doubles the odds of that.
  Example: (row cue) 3141592653 → 6242082606; (differences cue) → 6864446488.

**mux2.diffsh1-id.diffsh1-diffsh1-id** — The command sets how many times to take neighbour-differences:
once, or twice. The command is an iteration count rather than a choice of unrelated tasks.
  Example: (once cue) 3141592653 → 3837443498; (twice cue) → 3554709159.

**mux3.pmstep-id.sr12-id.fixedpoint-id** — Three unrelated tasks: parity-step, the sr12 on/off latch, and
the position-match flag.
  Example (fixedpoint member): [t2]3141592653 → 0100000000.

**mux3.sr12-id.sr34-id.sr56-id** — One on/off latch, but the command picks which trigger-pair drives it:
sr12 uses 1/2, sr34 uses 3/4, sr56 uses 5/6. The tasks share the latch machinery and differ only in which
two digit-values switch it.
  Example (sr34 member): [t1]3141592653 → 1100000001.
