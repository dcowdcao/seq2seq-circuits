"""Edge Pruning circuit discovery — the third method (parallel to acdc/eap).

A faithful PORT of "Finding Transformer Circuits with Edge Pruning" (upstream:
https://github.com/princeton-nlp/Edge-Pruning, MIT) onto the frozen
TransformerLens models: each reader's masked input is built as a CORRECTION on
the real residual stream — Σ_w (1−z[w])·(corr_x[w] − live_x[w]) swaps every
ablated edge's writer to its corrupted-input state (interchange patching) and
leaves kept edges untouched, so no attention/MLP math is re-derived and the
reconstruction is exact (see assert_self_consistency). The L0 mask machinery
is imported from find_circuits/l0.py (vendored verbatim from upstream
src/modeling/l0.py).

Per target sparsity the run writes one candidate folder (same layout as
acdc/eap, via write_candidate_folder): circuits_data/<model>/<circuit>/circuits/
edge_pruning_sparsity_<s>/... Dirs are named by the TARGET sparsity; the
ACHIEVED sparsity is recorded in summary.json. --smoke quarantines outputs
under .smoke/run3/, builds pairs in memory, shrinks to 1 target / <=60 steps.

--zero-ablation (C4) sets corr_x=None so _delta returns -live_v (ablate every
edge to zero, no corrupted patch). --use-pos-embed (C4) splits the always-on
embedding reader-edges into tok/pos roots in the drawn circuit. --probe <var>
(C2) repoints the final reader to the probe site blocks.{l}.hook_resid_post over
the writers in layers <= l (causal cutoff), swaps the teacher/objective to
probe-readout KL and faithfulness to probe argmax agreement (helpers imported
from eap.py); candidates write under edge_pruning_probe_<var>_sparsity_<s>. The
flags add _zero / _pos suffixes so canonical artifacts are never clobbered.

Must NOT be imported under a colliding name; run as a script or load via
importlib under a unique module name (mirrors acdc.py/eap.py).
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
from functools import partial
from typing import Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # .../find_circuits
PIPELINE_ROOT = os.path.dirname(_THIS_DIR)                    # .../pipeline
EP_L0 = os.path.join(_THIS_DIR, "l0.py")    # the L0 primitives (vendored from Edge-Pruning)

if os.path.join(PIPELINE_ROOT, "models") not in sys.path:
    sys.path.insert(0, os.path.join(PIPELINE_ROOT, "models"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm import tqdm  # noqa: E402

import model_registry  # noqa: E402  (bootstraps interp_tools/data_gen/metrics onto sys.path)
from model_registry import (  # noqa: E402
    apply_gpu_memory_cap, recovery_ratio, resolve_device, wait_for_headroom,
)
from canon import (  # noqa: E402
    CANON_EDGE_PRUNING_SPARSITIES, DEFAULT_SEED,
)

GENERATED_BY = "pipeline/find_circuits/edge_pruning.py"

#: Early-stop policy. SPARSITY_TOL is the slack (in sparsity units) inside which the
#: constraint counts as met — it therefore decides how far ``achieved_edge_sparsity``,
#: and so ``n_edges_surviving``, may sit from the target, and it is recorded per run.
SPARSITY_TOL = 0.05
KL_MIN_DELTA = 1e-4
CHECK_EVERY = 25


# The shared helper hub is find_circuits/eap.py, loaded under a collision-proof
# module name (see acdc.py::_load_fc_eap for the canonical recipe).
def _load_fc_eap():
    name = "pipeline_find_circuits_eap"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(_THIS_DIR, "eap.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


_fc_eap = _load_fc_eap()
set_checkpoint_env = _fc_eap.set_checkpoint_env
artifact_meta = _fc_eap.artifact_meta
build_dataloader_from_pairs = _fc_eap.build_dataloader_from_pairs
slice_pairs = _fc_eap.slice_pairs
make_logit_diff_metric = _fc_eap.make_logit_diff_metric
load_run_pairs = _fc_eap.load_run_pairs
write_candidate_folder = _fc_eap.write_candidate_folder
# probe-metric helpers (Direction 3) — shared with eap.py/acdc.py
load_probe_artifact = _fc_eap.load_probe_artifact
probe_meta = _fc_eap.probe_meta
probe_logprobs = _fc_eap.probe_logprobs
probe_capture_hook = _fc_eap.probe_capture_hook
install_probe_capture = _fc_eap.install_probe_capture
make_probe_metric = _fc_eap.make_probe_metric
probe_baselines = _fc_eap.probe_baselines
check_probe_layer = _fc_eap.check_probe_layer


# L0 primitives from the vendored l0.py (imported by file path under a unique
# module name) + a local copy of get_mask (upstream src/modeling/modeling_fpt2.py:106).
def _load_l0():
    name = "pipeline_edge_pruning_l0"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, EP_L0)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


_l0 = _load_l0()
sample_z_from_log_alpha = _l0.sample_z_from_log_alpha
deterministic_z_from_log_alpha = _l0.deterministic_z_from_log_alpha


def get_mask(log_alpha, training=False, threshold_for_deterministic=None,
             apply_one=False) -> torch.Tensor:
    """Verbatim copy of modeling_fpt2.py::get_mask (l. 106) — training samples
    the stochastic gate; eval uses the deterministic (top-k / expected-L0) gate,
    optionally binarized at a threshold."""
    if training:
        mask = sample_z_from_log_alpha(log_alpha)
    else:
        mask = deterministic_z_from_log_alpha(log_alpha, apply_one=apply_one)
        if threshold_for_deterministic is not None:
            mask = (mask > threshold_for_deterministic).to(mask.dtype)
    return mask


def _sparsity_str(s: float) -> str:
    return "%g" % s


def _circuit_name(s: float, probe: Optional[str] = None,
                  zero_ablation: bool = False, use_pos_embed: bool = False) -> str:
    base = (f"edge_pruning_probe_{probe}_sparsity_{_sparsity_str(s)}" if probe
            else f"edge_pruning_sparsity_{_sparsity_str(s)}")
    return base + ("_zero" if zero_ablation else "") + ("_pos" if use_pos_embed else "")


# =========================================================================== #
# DisentangledModel — the Edge-Pruning forward, riding a frozen HookedTransformer
# =========================================================================== #
class DisentangledModel(nn.Module):
    """Wraps a FROZEN HookedTransformer and learns per-edge L0 masks over its
    writer→reader graph (edges only; no node masks). The masked, interchange-
    patched forward is implemented as a residual-stream correction applied
    through TransformerLens hooks (see the module docstring).

    Writers (always WITHOUT embedding nodes, mirroring ACDC):
        a{l}.h{h}  (per-head attn output, pre-bias)  and, for attn+MLP models,
        m{l}       (per-layer MLP output).
    Readers:
        a{l}.h{h}.{q,k,v}  (per-head q/k/v inputs),  m{l}  (MLP input, attn+MLP
        only),  and  logits (the final read into resid_post → ln_final).
    Edges run writer→reader, restricted to causally-valid pairs (a reader reads
    only the writers already in the residual stream at its hook).
    """

    def __init__(self, model_name: str, device: Optional[str] = None,
                 probe: Optional[Dict] = None) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = resolve_device(device)
        model = model_registry.load_model(model_name, device=self.device)
        model.requires_grad_(False)
        model.eval()
        # Local-instance flags that expose the per-head / per-MLP hook points.
        model.cfg.use_attn_result = True
        model.cfg.use_split_qkv_input = True
        self.attn_only = bool(model.cfg.attn_only)
        if not self.attn_only:
            model.cfg.use_hook_mlp_in = True
        self.model = model

        self.L = int(model.cfg.n_layers)
        self.H = int(model.cfg.n_heads)
        self.d_model = int(model.cfg.d_model)
        dev = self.device

        # ---- writer indexing -------------------------------------------------
        # MLP models:  slot per layer = [h0..h_{H-1}, m]  -> stride H+1.
        # attn-only :  slot per layer = [h0..h_{H-1}]      -> stride H.
        self._stride = self.H + (1 if not self.attn_only else 0)
        self.n_writers = self.L * self._stride

        # ---- per-layer read log_alphas (edge masks) -------------------------
        def _p(shape):
            t = nn.Parameter(torch.empty(*shape, dtype=torch.float32, device=dev))
            t.data.normal_(mean=10.0, std=0.01)
            return t

        self.q_read = nn.ParameterList([_p((self.n_writers, self.H)) for _ in range(self.L)])
        self.k_read = nn.ParameterList([_p((self.n_writers, self.H)) for _ in range(self.L)])
        self.v_read = nn.ParameterList([_p((self.n_writers, self.H)) for _ in range(self.L)])
        if not self.attn_only:
            self.mlp_read = nn.ParameterList([_p((self.n_writers,)) for _ in range(self.L)])
        else:
            self.mlp_read = None
        self.final_read = _p((self.n_writers,))

        # ---- Lagrange multipliers (trained with gradient ASCENT) ------------
        self.lambda_1 = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=dev))
        self.lambda_2 = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=dev))

        # ---- common masks + valid-writer index tensors ----------------------
        # attn reader at layer l reads writers in layers < l;
        # mlp  reader at layer l reads writers in layers < l + this layer's heads;
        # final reader reads every writer.
        attn_off = [l * self._stride for l in range(self.L)]              # first writer of layer l
        mlp_off = [l * self._stride + self.H for l in range(self.L)]      # m{l} slot
        self._attn_valid = [torch.arange(o, dtype=torch.long, device=dev) for o in attn_off]
        self._mlp_valid = [torch.arange(o, dtype=torch.long, device=dev) for o in mlp_off]
        self._final_valid = torch.arange(self.n_writers, dtype=torch.long, device=dev)

        def _common(n_valid):
            m = torch.zeros(self.n_writers, dtype=torch.float32, device=dev)
            m[:n_valid] = 1.0
            return m

        self.attn_common = [_common(o).unsqueeze(1) for o in attn_off]    # (n_writers,1)
        self.mlp_common = [_common(o) for o in mlp_off]                   # (n_writers,)
        self.final_common = torch.ones(self.n_writers, dtype=torch.float32, device=dev)

        # ---- C2 probe: repoint the FINAL reader to the probe site -----------
        # In probe mode the final reader reads blocks.{l}.hook_resid_post over the
        # writers in layers <= l (causal cutoff); layers > l feed nothing the probe
        # sees, so they are excluded from masking + the sparsity denominator.
        self.probe = probe
        if probe is not None:
            l = int(probe["layer"])
            self.probe_layer = l
            self._final_hook = probe["hook"]                  # blocks.{l}.hook_resid_post
            n_le = (l + 1) * self._stride                     # writers in layers <= l
            self._final_valid = torch.arange(n_le, dtype=torch.long, device=dev)
            fc = torch.zeros(self.n_writers, dtype=torch.float32, device=dev)
            fc[:n_le] = 1.0
            self.final_common = fc
        else:
            self._final_hook = f"blocks.{self.L - 1}.hook_resid_post"

        # ---- exact graph edge count (the sparsity denominator) --------------
        # In probe mode only edges among layers <= probe_layer (+ the probe reader) count.
        n_edges = 0.0
        top = (self.probe_layer + 1) if probe is not None else self.L
        for l in range(top):
            n_edges += 3 * self.H * float(self.attn_common[l].sum())
            if not self.attn_only:
                n_edges += float(self.mlp_common[l].sum())
        n_edges += float(self.final_common.sum())
        self.n_edges = int(round(n_edges))

    # ---- writer-slot helpers ------------------------------------------------
    def _w_attn(self, layer: int, head: int) -> int:
        return layer * self._stride + head

    def _w_mlp(self, layer: int) -> int:
        return layer * self._stride + self.H

    def _writer_record(self, w: int) -> Dict:
        layer = w // self._stride
        within = w % self._stride
        if (not self.attn_only) and within == self.H:
            return {"label": f"m{layer}", "kind": "mlp_out", "layer": layer, "head": None}
        return {"label": f"a{layer}.h{within}", "kind": "result", "layer": layer, "head": within}

    # ---- writer states (per-writer real outputs of ONE clean forward) -------
    @torch.no_grad()
    def writer_states(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return ``(n_writers, B, S, d_model)`` of each writer's real output on
        ``tokens`` (one frozen forward). Used as ``corr_x`` (the corrupted side
        of interchange patching) and by the self-consistency check."""
        names = [f"blocks.{l}.attn.hook_result" for l in range(self.L)]
        if not self.attn_only:
            names += [f"blocks.{l}.hook_mlp_out" for l in range(self.L)]
        name_set = set(names)
        _, cache = self.model.run_with_cache(
            tokens, names_filter=lambda n: n in name_set, return_type=None)
        B, S = tokens.shape[0], tokens.shape[1]
        ws = torch.zeros(self.n_writers, B, S, self.d_model,
                         device=tokens.device, dtype=cache[names[0]].dtype)
        for l in range(self.L):
            res = cache[f"blocks.{l}.attn.hook_result"]      # (B,S,H,d)
            for h in range(self.H):
                ws[self._w_attn(l, h)] = res[:, :, h, :]
            if not self.attn_only:
                ws[self._w_mlp(l)] = cache[f"blocks.{l}.hook_mlp_out"]   # (B,S,d)
        return ws.detach()

    # ---- masks (sampled in training, deterministic in eval) -----------------
    def _compute_masks(self, deterministic: bool,
                       threshold: Optional[float] = None,
                       force_value: Optional[float] = None) -> Dict:
        """Per-reader-group gates z = mask * common_mask, plus the running
        z_edges_sum (Σ over all valid edges). ``force_value`` (0.0/1.0) bypasses
        the gate entirely (the self-consistency check)."""
        train = not deterministic
        out = {"q_z": [], "k_z": [], "v_z": [], "mlp_z": []}

        def gate(log_alpha, common, count: bool = True):
            """``(z, z_sum)``: z is the per-edge gate for the interchange
            correction (1 = keep, 0 = ablate -> swap to corr_x). ``count=False``
            (probe mode, reader above the cutoff) keeps every edge and adds nothing
            to z_sum — those readers are downstream of the probe site."""
            if force_value is not None:
                z = torch.full_like(log_alpha, float(force_value)) * common
                return z, z.new_zeros(())
            if not count:
                return torch.ones_like(log_alpha) * common, log_alpha.new_zeros(())
            z = get_mask(log_alpha, training=train,
                         threshold_for_deterministic=threshold) * common
            return z, z.sum()

        z_sum = torch.zeros((), device=self.final_read.device)
        for l in range(self.L):
            count = not (self.probe is not None and l > self.probe_layer)
            qz, qs = gate(self.q_read[l], self.attn_common[l], count)
            kz, ks = gate(self.k_read[l], self.attn_common[l], count)
            vz, vs = gate(self.v_read[l], self.attn_common[l], count)
            out["q_z"].append(qz)
            out["k_z"].append(kz)
            out["v_z"].append(vz)
            z_sum = z_sum + qs + ks + vs
            if not self.attn_only:
                mz, ms = gate(self.mlp_read[l], self.mlp_common[l], count)
                out["mlp_z"].append(mz)
                z_sum = z_sum + ms
            else:
                out["mlp_z"].append(None)
        fz, fs = gate(self.final_read, self.final_common)
        out["final_z"] = fz
        z_sum = z_sum + fs
        out["z_edges_sum"] = z_sum
        return out

    def _edge_loss(self, z_edges_sum, target_edge_sparsity):
        model_sparsity = 1.0 - (z_edges_sum / self.n_edges)
        if target_edge_sparsity is None:
            return None, model_sparsity
        d = model_sparsity - target_edge_sparsity
        edge_loss = self.lambda_1.reshape([]) * d + self.lambda_2.reshape([]) * d ** 2
        return edge_loss, model_sparsity

    # ---- the disentangled forward -------------------------------------------
    def forward(self, tokens: torch.Tensor, corr_x: Optional[torch.Tensor] = None,
                target_edge_sparsity: Optional[float] = None,
                deterministic: bool = False, threshold: Optional[float] = None,
                force_value: Optional[float] = None) -> tuple:
        """Masked, interchange-patched logits.

        Returns ``(logits, edge_loss, model_edge_sparsity)``. With every gate at
        1 and ``corr_x=None`` the corrections vanish and the output is the real
        model's logits (the self-consistency property).
        """
        masks = self._compute_masks(deterministic, threshold=threshold,
                                    force_value=force_value)
        live: List[Optional[torch.Tensor]] = [None] * self.n_writers
        use_corr = corr_x is not None

        def cap_result(value, hook, layer):                # capture attn writers
            for h in range(self.H):
                live[self._w_attn(layer, h)] = value[:, :, h, :]
            return value

        def cap_mlp(value, hook, layer):                   # capture MLP writer
            live[self._w_mlp(layer)] = value
            return value

        def _delta(valid):
            live_v = torch.stack([live[int(i)] for i in valid], dim=0)     # (V,B,S,d)
            if use_corr:
                return corr_x.index_select(0, valid) - live_v
            return -live_v                                  # corr_x=None -> zero-ablation

        def corr_qkv(value, hook, layer, chan):            # add per-head correction
            valid = self._attn_valid[layer]
            if valid.numel() == 0:
                return value
            mzz = masks[f"{chan}_z"][layer].index_select(0, valid)         # (V,H)
            corr = torch.einsum("vh,vbsd->bshd", (1.0 - mzz), _delta(valid))
            return value + corr

        def corr_mlp(value, hook, layer):
            valid = self._mlp_valid[layer]
            if valid.numel() == 0:
                return value
            mz = masks["mlp_z"][layer].index_select(0, valid)              # (V,)
            corr = torch.einsum("v,vbsd->bsd", (1.0 - mz), _delta(valid))
            return value + corr

        def corr_final(value, hook):
            valid = self._final_valid
            mz = masks["final_z"].index_select(0, valid)
            corr = torch.einsum("v,vbsd->bsd", (1.0 - mz), _delta(valid))
            return value + corr

        fwd_hooks = []
        for l in range(self.L):
            fwd_hooks.append((f"blocks.{l}.attn.hook_result", partial(cap_result, layer=l)))
            fwd_hooks.append((f"blocks.{l}.hook_q_input", partial(corr_qkv, layer=l, chan="q")))
            fwd_hooks.append((f"blocks.{l}.hook_k_input", partial(corr_qkv, layer=l, chan="k")))
            fwd_hooks.append((f"blocks.{l}.hook_v_input", partial(corr_qkv, layer=l, chan="v")))
            if not self.attn_only:
                fwd_hooks.append((f"blocks.{l}.hook_mlp_out", partial(cap_mlp, layer=l)))
                fwd_hooks.append((f"blocks.{l}.hook_mlp_in", partial(corr_mlp, layer=l)))
        fwd_hooks.append((self._final_hook, corr_final))    # probe mode: blocks.{l}.hook_resid_post

        logits = self.model.run_with_hooks(tokens, return_type="logits",
                                           fwd_hooks=fwd_hooks)
        edge_loss, model_sparsity = self._edge_loss(masks["z_edges_sum"],
                                                    target_edge_sparsity)
        return logits, edge_loss, model_sparsity

    # ---- self-consistency check --------------------------------------------
    @torch.no_grad()
    def assert_self_consistency(self, tokens: torch.Tensor, tol: float = 1e-4) -> float:
        """The whole interchange machinery must be exact: with EVERY edge ablated
        and ``corr_x`` = the writer states of the SAME (clean) input, each reader
        is fully reconstructed from the gathered writers and must reproduce the
        real model. Proves writer indexing, common masks, biases and the per-head
        LN reassembly are all correct."""
        ref = self.model(tokens, return_type="logits")
        corr = self.writer_states(tokens)
        got, _, _ = self.forward(tokens, corr_x=corr, force_value=0.0)
        max_abs = (got - ref).abs().max().item()
        if not (max_abs < tol):
            raise AssertionError(
                f"edge_pruning self-consistency FAILED: max|Δlogit|={max_abs:.3e} "
                f">= {tol:.0e} (the disentangled reconstruction does not match the "
                f"real model — writer/reader bookkeeping is wrong)")
        return max_abs

    # ---- deterministic edge extraction --------------------------------------
    @torch.no_grad()
    def surviving_edges(self, threshold: float = 0.5) -> List[Dict]:
        """The discrete circuit: every valid edge whose deterministic gate
        survives (> ``threshold``). Each record mirrors the ACDC schema fields the
        smallest-faithful selection + the drawer consume."""
        def det(log_alpha):
            return deterministic_z_from_log_alpha(log_alpha)      # soft, top-k zeroed

        records: List[Dict] = []

        def emit(w, reader_label, reader_kind, reader_layer, reader_head, qkv, score):
            wr = self._writer_record(w)
            records.append({
                "sender_label": wr["label"], "sender_kind": wr["kind"],
                "sender_layer": wr["layer"], "sender_head": wr["head"],
                "receiver_label": reader_label, "receiver_kind": reader_kind,
                "receiver_layer": reader_layer, "receiver_head": reader_head,
                "qkv": qkv, "effect_size": float(score),
            })

        for l in range(self.L):
            if self.probe is not None and l > self.probe_layer:
                continue                                      # downstream of the probe site
            valid = self._attn_valid[l]
            for chan, par in (("q", self.q_read), ("k", self.k_read), ("v", self.v_read)):
                soft = det(par[l])                            # (n_writers,H)
                for w in valid.tolist():
                    for h in range(self.H):
                        sc = float(soft[w, h])
                        if sc > threshold:
                            emit(w, f"a{l}.h{h}_{chan}", chan, l, h, chan, sc)
            if not self.attn_only:
                soft = det(self.mlp_read[l])                  # (n_writers,)
                for w in self._mlp_valid[l].tolist():
                    sc = float(soft[w])
                    if sc > threshold:
                        emit(w, f"m{l}", "mlp_in", l, None, None, sc)
        soft = det(self.final_read)
        for w in self._final_valid.tolist():
            sc = float(soft[w])
            if sc > threshold:
                emit(w, "logits", "resid_post", None, None, None, sc)

        records.sort(key=lambda e: -abs(e["effect_size"]))
        return records


# --------------------------------------------------------------------------- #
# surviving edges -> drawer graph (EAP vocabulary)                            #
# --------------------------------------------------------------------------- #
def _receiver_node(label: str) -> str:
    """'a0.h3_q' -> 'a0.h3' ; 'm1' -> 'm1' ; 'logits' -> 'logits'."""
    for suf in ("_q", "_k", "_v"):
        if label.endswith(suf):
            return label[:-2]
    return label


def _embed_reader_edges(records: List[Dict], attn_only: bool,
                        use_pos_embed: bool = False) -> List[Dict]:
    """The always-on EMBEDDING readers, as ``embed``->reader edges in the ACDC
    record schema, so BOTH this method's own circuit.png AND the smallest-faithful
    re-parse (which reads ``summary.json``'s ``surviving_edges`` in the
    ``acdc_summary`` dialect) surface an ``input`` node — exactly like eap.py and
    acdc.py already do. ``use_pos_embed`` (C4) splits each into a ``tok_embed`` and
    a ``pos_embed`` sender (the embedding is not a maskable writer, so this is a
    visualization split of the always-on input node into token/position roots).

    WHY this is needed: Edge Pruning never makes the embedding a MASKED writer
    (each reader's residual baseline is ``embed + biases + Σ_w z[w]·…``), so a
    reader CHANNEL that no surviving writer feeds reads SOLELY from the embedding.
    That is the visual analog of EAP's ``input->a{l}.h{h}<chan>`` source edges and
    ACDC's ``embed``->reader edges; we synthesize one ``embed``->reader edge per
    such channel (q/k/v of a kept head, a kept MLP reader, or the final ``logits``
    read). ``effect_size`` is 1.0 (the embedding gate is always on) and each record
    carries ``always_on: True``, because these are NOT maskable edges of this
    method's graph: they are deliberately EXCLUDED from ``n_edges_surviving`` /
    achieved sparsity (which count only trainable writer→reader edges)."""
    fed = {r["receiver_label"] for r in records}         # reader channels a writer feeds
    heads, mlps = set(), set()
    for r in records:
        for lab in (r["sender_label"], _receiver_node(r["receiver_label"])):
            if lab.startswith("a") and ".h" in lab:
                l, h = lab[1:].split(".h")
                heads.add((int(l), int(h)))
            elif len(lab) > 1 and lab[0] == "m" and lab[1:].isdigit():
                mlps.add(int(lab[1:]))

    out: List[Dict] = []
    senders = (["tok_embed", "pos_embed"] if use_pos_embed else ["embed"])

    def emit(receiver_label, qkv, receiver_kind, layer, head):
        for sender in senders:
            out.append({
                "sender_label": sender, "sender_kind": "embed",
                "sender_layer": None, "sender_head": None,
                "receiver_label": receiver_label, "receiver_kind": receiver_kind,
                "receiver_layer": layer, "receiver_head": head,
                "qkv": qkv, "effect_size": 1.0, "always_on": True,
            })

    for (l, h) in sorted(heads):
        for chan in ("q", "k", "v"):
            if f"a{l}.h{h}_{chan}" not in fed:           # channel fed only by embed
                emit(f"a{l}.h{h}_{chan}", chan, f"{chan}_input", l, h)
    if not attn_only:
        for l in sorted(mlps):
            if f"m{l}" not in fed:
                emit(f"m{l}", None, "mlp_in", l, None)
    if "logits" not in fed:      # same rule for the FINAL reader, or prune_dangling
        emit("logits", None, "resid_post", None, None)   # eats a circuit with no logits node
    return out


def _to_drawer_graph(records: List[Dict]):
    """Convert surviving edges into the FROZEN drawer's (nodes, edges) form
    — nodes ordered INPUT, then heads-by-(layer,head), then MLPs, then logits;
    edges [(parent, child, qkv, score)]. An ``embed``-sender record (the always-on
    embedding readers from _embed_reader_edges) maps to the EAP/ACDC ``input``
    node, so the rendered graph is rooted at ``input`` exactly like eap.py /
    acdc.py."""
    edges, internal, seen = [], set(), set()
    has_input = has_pos = has_logits = False
    for e in records:
        parent = e["sender_label"]                       # a{l}.h{h}, m{l}, embed, tok/pos_embed
        if parent in ("embed", "input", "tok_embed"):
            parent = "input"                             # token/combined embedding root
        child = _receiver_node(e["receiver_label"])      # a{l}.h{h} / m{l} / logits
        qkv = e["qkv"]
        if parent == child:
            continue
        key = (parent, child, qkv)
        if key in seen:
            continue
        seen.add(key)
        edges.append((parent, child, qkv, float(e["effect_size"])))
        for nm in (parent, child):
            if nm == "input":
                has_input = True
            elif nm == "pos_embed":                      # C4 positional root
                has_pos = True
            elif nm == "logits":
                has_logits = True
            else:
                internal.add(nm)

    def node_key(nm):
        if nm.startswith("a"):                           # "a2.h7" -> (2, 0, 7)
            l, h = nm[1:].split(".h")
            return (int(l), 0, int(h))
        return (int(nm[1:]), 1, 0)                        # "m{l}" sorts after that layer's heads

    nodes = ((["input"] if has_input else []) + (["pos_embed"] if has_pos else [])
             + sorted(internal, key=node_key))
    if has_logits:
        nodes.append("logits")
    return nodes, edges


def _head_str(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


# --------------------------------------------------------------------------- #
# Training one target sparsity                                                 #
# --------------------------------------------------------------------------- #
def _warmup_target(step: int, warmup_steps: int, target: float,
                   start: float = 0.0) -> float:
    """Logarithmic sparsity warmup (modeling_fpt2 trainer, warmup_type=
    'logarithmic'): ramp from ``start`` to ``target`` over ``warmup_steps``."""
    if step >= warmup_steps or warmup_steps <= 0:
        return target
    log_one_minus = (math.log(1 - start)
                     + (math.log(1 - target) - math.log(1 - start))
                     * step / warmup_steps)
    return 1.0 - math.exp(log_one_minus)


def _precompute_train_cache(masked_model: DisentangledModel, batches, ans,
                            zero_ablation: bool = False, probe: Optional[Dict] = None):
    """Cache the mask-INDEPENDENT tensors for each train batch ONCE: the
    corrupted-side writer states (``corr_x``) and the teacher KL target
    (log-softmax of the frozen full model's clean logits over the answer
    positions). Both depend only on the input batch, never on the masks, so they
    are reused across EVERY step and EVERY target sparsity — turning the per-step
    cost from three forward passes (corr_x + teacher + masked) into one (masked).
    Bit-identical to recomputing them every step (same frozen no-grad forwards;
    these draw no randomness, so the mask-sampling RNG stream is unchanged).

    Entries are ``(clean, corr_x, tlog, pos, mask)``. Legacy labels ([B,2,P],
    shared answer positions ``ans``): tlog is [(B*P),V], pos/mask are None.
    Per-pair labels ([B,4,P], eap.pack_labels): tlog is [B,P,V] gathered at the
    per-row positions (labels row 2), mask is the valid-position mask (row 3)."""
    dev = masked_model.device
    cache = []
    with torch.no_grad():
        for clean, corrupted, labels in batches:
            clean = clean.to(dev)
            corrupted = corrupted.to(dev)
            corr_x = None if zero_ablation else masked_model.writer_states(corrupted)
            if probe is not None:                          # C2: teacher = clean probe log-probs
                masked_model.model(clean, return_type=None)
                tlog = probe_logprobs(masked_model._probe_capture["resid"], probe).float()  # [B,P,C]
                cache.append((clean, corr_x, tlog, None, None))
                continue
            teacher = masked_model.model(clean, return_type="logits")
            V = teacher.shape[-1]
            if labels.shape[1] == 4:                       # per-row answer positions
                pos = labels[:, 2, :].long().to(dev)                        # [B,P]
                sel = teacher.gather(1, pos.unsqueeze(-1).expand(-1, -1, V))
                tlog = F.log_softmax(sel, dim=-1)                           # [B,P,V]
                mask = labels[:, 3, :].to(dev).float()                      # [B,P]
                cache.append((clean, corr_x, tlog, pos, mask))
            else:
                tlog = F.log_softmax(teacher[:, ans, :].reshape(-1, V), dim=-1)
                cache.append((clean, corr_x, tlog, None, None))
    return cache


def _precompute_eval_cache(masked_model: DisentangledModel, batches,
                           zero_ablation: bool = False, probe: Optional[Dict] = None):
    """Cache the eval batches' mask-independent tensors (``corr_x`` + labels) and
    the full-model clean/corrupted BASELINES (also mask-independent). Only the
    circuit's own metric is recomputed per target (one deterministic masked
    forward). ``zero_ablation`` sets ``corr_x=None`` (forward zero-ablates); in
    C2 probe mode the labels are per-batch clean probe log-probs and the baselines
    are clean/corrupted probe agreement. Returns ``(cache, clean_base, corrupt_base)``."""
    dev = masked_model.device
    cache = []
    if probe is not None:
        capture = masked_model._probe_capture
        metric_eval = partial(make_probe_metric(probe, capture), loss=False, mean=False)
        labeled = []
        with torch.no_grad():
            for clean, corrupted, _ in batches:
                clean = clean.to(dev)
                corrupted = corrupted.to(dev)
                masked_model.model(clean, return_type=None)
                ref = probe_logprobs(capture["resid"], probe).float()       # [B,P,C]
                corr_x = None if zero_ablation else masked_model.writer_states(corrupted)
                cache.append((clean, corr_x, ref))
                labeled.append((clean, corrupted, ref))
        clean_ag, corrupt_ag = probe_baselines(masked_model.model, labeled, metric_eval)
        return cache, clean_ag, corrupt_ag
    metric = make_logit_diff_metric(masked_model.answer_positions)
    clean_ld = corrupt_ld = 0.0
    n = 0
    with torch.no_grad():
        for clean, corrupted, labels in batches:
            clean = clean.to(dev)
            corrupted = corrupted.to(dev)
            labels = labels.to(dev)
            b = clean.shape[0]
            corr_x = None if zero_ablation else masked_model.writer_states(corrupted)
            clean_ld += metric(masked_model.model(clean, return_type="logits"),
                               None, None, labels, mean=True).item() * b
            corrupt_ld += metric(masked_model.model(corrupted, return_type="logits"),
                                 None, None, labels, mean=True).item() * b
            cache.append((clean, corr_x, labels))
            n += b
    return cache, clean_ld / n, corrupt_ld / n


def train_one_sparsity(masked_model: DisentangledModel, train_cache, ans, target: float,
                       steps: int, edge_lr: float, reg_lr: float,
                       warmup_frac: float, amp: bool, seed: int,
                       early_stop: bool = True, patience: int = 6,
                       verbose: bool = True) -> Dict:
    """Optimize the per-edge log_alphas (AdamW, minimize) + the two Lagrange
    multipliers (AdamW, maximize) so the masked model matches the full model
    (KL over the answer positions) at the requested edge sparsity. The teacher
    target + corrupted writer states come PRECOMPUTED in ``train_cache``
    (mask-independent — one forward each instead of one per step).

    With ``early_stop`` the run halts once it is (a) past warmup, (b) the achieved
    (expected) sparsity is within ``SPARSITY_TOL`` of the target — i.e. the
    constraint is met — and (c) the windowed KL has not improved by ``KL_MIN_DELTA``
    for ``patience`` consecutive ``CHECK_EVERY``-step windows. Conservative: if the
    constraint is never met it runs the full ``steps``."""
    masked_model.train()
    # reset masks + multipliers fresh for each target (independent search)
    log_alpha_params = (list(masked_model.q_read) + list(masked_model.k_read) + list(masked_model.v_read)
                        + ([] if masked_model.attn_only else list(masked_model.mlp_read))
                        + [masked_model.final_read])
    for p in log_alpha_params:
        p.data.normal_(mean=10.0, std=0.01)
    masked_model.lambda_1.data.zero_()
    masked_model.lambda_2.data.zero_()

    opt_alpha = torch.optim.AdamW(log_alpha_params, lr=edge_lr)
    opt_lambda = torch.optim.AdamW([masked_model.lambda_1, masked_model.lambda_2], lr=reg_lr,
                                   maximize=True)

    dev = masked_model.device
    use_amp = amp and str(dev).startswith("cuda")
    warmup_steps = max(1, int(warmup_frac * steps))
    g = torch.Generator(device="cpu").manual_seed(seed)
    nb = len(train_cache)

    best_kl, wait = float("inf"), 0
    win_kl, win_sp = [], []
    stopped_at = steps
    last = {"kl": float("nan"), "model_sparsity": float("nan"),
            "edge_loss": float("nan"), "target_t": 0.0}
    pbar = tqdm(range(steps), desc=f"EP train [s={target:g}]", unit="step",
                leave=False, disable=not verbose)
    for step in pbar:
        clean, corr_x, tlog, pos, mask = train_cache[int(torch.randint(nb, (1,), generator=g))]
        target_t = _warmup_target(step, warmup_steps, target)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits, edge_loss, model_sparsity = masked_model(
                clean, corr_x=corr_x, target_edge_sparsity=target_t)
            if masked_model.probe is not None:                   # C2: probe-readout KL at the site
                mlog = probe_logprobs(masked_model._probe_capture["resid"], masked_model.probe).float()
                kl = (tlog.exp() * (tlog - mlog)).sum(-1).mean()           # [B,P] -> scalar
            elif pos is None:                              # shared answer positions
                mlog = F.log_softmax(
                    logits[:, ans, :].reshape(-1, logits.shape[-1]).float(), dim=-1)
                kl = F.kl_div(mlog, tlog, reduction="batchmean", log_target=True)
            else:                                          # per-row positions, masked mean over valid slots
                sel = logits.gather(1, pos.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))
                mlog = F.log_softmax(sel.float(), dim=-1)
                kl_bp = (tlog.exp() * (tlog - mlog)).sum(-1)             # [B,P]
                kl = (kl_bp * mask).sum() / mask.sum().clamp(min=1)
            loss = kl + edge_loss

        opt_alpha.zero_grad(set_to_none=True)
        opt_lambda.zero_grad(set_to_none=True)
        loss.backward()
        opt_alpha.step()
        opt_lambda.step()

        klv = float(kl.detach())
        spv = float(model_sparsity.detach())
        last = {"kl": klv, "model_sparsity": spv,
                "edge_loss": float(edge_loss.detach()), "target_t": float(target_t)}
        if verbose and step % 20 == 0:
            pbar.set_postfix(kl=round(klv, 4), sp=round(spv, 3), tgt=round(target_t, 3))

        # convergence early stopping (only past warmup, only once the sparsity
        # constraint is met; then require the windowed KL to plateau).
        win_kl.append(klv)
        win_sp.append(spv)
        if early_stop and (step + 1) % CHECK_EVERY == 0 and (step + 1) > warmup_steps:
            avg_kl = sum(win_kl) / len(win_kl)
            avg_sp = sum(win_sp) / len(win_sp)
            win_kl.clear()
            win_sp.clear()
            if abs(avg_sp - target) <= SPARSITY_TOL:
                if avg_kl < best_kl - KL_MIN_DELTA:
                    best_kl, wait = avg_kl, 0
                else:
                    wait += 1
                    if wait >= patience:
                        stopped_at = step + 1
                        break
            else:
                wait = 0                       # constraint not yet met — keep going
    pbar.close()
    masked_model.eval()
    last["stopped_at_step"] = stopped_at
    last["early_stopped"] = bool(stopped_at < steps)
    return last


# --------------------------------------------------------------------------- #
# Faithfulness of the deterministic circuit (mirrors EAP's logit-diff ratio)   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _faithfulness(masked_model: DisentangledModel, eval_cache, clean_ld: float,
                  corrupt_ld: float, threshold: float) -> Dict:
    """Logit-diff faithfulness of the deterministic circuit (EAP's ratio). The
    clean/corrupted baselines + ``corr_x`` come PRECOMPUTED (mask-independent);
    only the circuit's own logit-diff (one deterministic masked forward per batch)
    is computed here."""
    if masked_model.probe is not None:                           # C2: probe argmax agreement
        circuit_ag = 0.0
        n = 0
        for clean, corr_x, ref in eval_cache:
            b = clean.shape[0]
            masked_model.forward(clean, corr_x=corr_x, deterministic=True, threshold=threshold)
            cur = probe_logprobs(masked_model._probe_capture["resid"], masked_model.probe)   # [B,P,C]
            circuit_ag += (cur.argmax(-1) == ref.argmax(-1)).float().mean().item() * b
            n += b
        circuit_ag /= n
        return {                                           # clean_ld/corrupt_ld carry agreement here
            "probe_agreement_clean_baseline": clean_ld,
            "probe_agreement_corrupted_baseline": corrupt_ld,
            "probe_agreement_circuit": circuit_ag,
            "faithfulness_ratio": recovery_ratio(circuit_ag, clean_ld, corrupt_ld),
            "eval_examples": n,
        }
    metric = make_logit_diff_metric(masked_model.answer_positions)
    circuit_ld = 0.0
    n = 0
    for clean, corr_x, labels in eval_cache:
        b = clean.shape[0]
        circ_logits, _, _ = masked_model.forward(
            clean, corr_x=corr_x, deterministic=True, threshold=threshold)
        circuit_ld += metric(circ_logits, None, None, labels, mean=True).item() * b
        n += b
    circuit_ld /= n
    return {
        "clean_baseline_logit_diff": clean_ld,
        "corrupted_baseline_logit_diff": corrupt_ld,
        "circuit_logit_diff": circuit_ld,
        "faithfulness_ratio": recovery_ratio(circuit_ld, clean_ld, corrupt_ld),
        "eval_examples": n,
    }


# --------------------------------------------------------------------------- #
# Write one candidate folder                                                   #
# --------------------------------------------------------------------------- #
def _write_candidate(masked_model, model_name, circuit, target, smoke, seed, threshold,
                     provenance, train_stats, faith, n_examples, verbose,
                     zero_ablation=False, use_pos_embed=False, probe=None):
    records = masked_model.surviving_edges(threshold=threshold)
    n_edges_surviving = len(records)                     # METRIC: maskable edges only
    achieved = 1.0 - (n_edges_surviving / masked_model.n_edges)
    # Always-on embedding readers -> an `input` node, for parity with eap.py /
    # acdc.py (these are structure/visualization edges, NOT counted in the
    # sparsity metric above). Prepended so input sorts first in the node order.
    all_edges = _embed_reader_edges(records, masked_model.attn_only, use_pos_embed) + records
    nodes, drawer_edges = _to_drawer_graph(all_edges)
    # PRUNE dangling heads/MLPs (EAP-IG Graph.prune semantics): Edge Pruning leaves no-outgoing heads (output feeds nothing), inflating n_heads. n_edges_surviving (the training diagnostic) is left as-is; the DRAWN circuit is the connected one.
    nodes, drawer_edges = _fc_eap.prune_dangling(nodes, drawer_edges)
    heads_in_circuit, _mlp_layers = _fc_eap.heads_mlps_from_nodes(nodes)
    mlps_in_circuit = [f"m{l}" for l in _mlp_layers]

    meta = artifact_meta(
        model_name, method="edge_pruning", generated_by=GENERATED_BY,
        n_examples=int(n_examples), seed=seed,
        scheme=provenance["scheme"], source_dataset_path=provenance["path"])
    if probe is not None:
        meta["probe"] = probe_meta(probe)

    metric_str = (f"probe-readout KL at {probe['hook']} over positions "
                  f"{probe['positions'].tolist()} + L0 edge sparsity (writers "
                  f"<= layer {int(probe['layer'])}); eval = probe argmax agreement"
                  if probe is not None else
                  "KL(full||masked) over the answer positions + L0 edge sparsity")
    cname = _circuit_name(target, probe=(probe["variable"] if probe else None),
                          zero_ablation=zero_ablation, use_pos_embed=use_pos_embed)
    # Edge Pruning's score is the L0 gate (~0.8-1.0), so the drawer's min/max penwidth scaling would pin every edge at the 5.0 ceiling; render all edges at one uniform width (EDGE_PRUNING_PENWIDTH).
    result = write_candidate_folder(
        model_name, circuit, cname, smoke=smoke, meta=meta,
        nodes=nodes, edges=drawer_edges, model_or_cfg=masked_model.model,
        circuit_json_method="edge_pruning",
        uniform_penwidth=_fc_eap.EDGE_PRUNING_PENWIDTH,
        summary={
            "method": "edge_pruning",
            "metric": metric_str,
            "target_edge_sparsity": float(target),
            "achieved_edge_sparsity": float(achieved),
            # method-INDEPENDENT denominator (canon.live_edge_budget) so a "fraction of
            # the graph" score is comparable with eap / acdc. n_edges_total is this
            # method's own MASKABLE count, which excludes the always-on embedding writer
            # (1184 vs 1281 at 4Lx8H) and is the denominator of achieved_edge_sparsity.
            "n_edges_universe": model_registry.live_edges(
                model_registry.get_spec(model_name)),
            "n_edges_total": int(masked_model.n_edges),
            "n_edges_surviving": n_edges_surviving,
            "extraction_threshold": float(threshold),
            "heads_in_circuit": [_head_str(l, h) for (l, h) in heads_in_circuit],
            "n_heads": len(heads_in_circuit),
            "mlps_in_circuit": mlps_in_circuit,
            "n_mlps": len(mlps_in_circuit),
            # The full circuit structure incl. always-on `embed`->reader edges (so the
            # smallest-faithful re-parse renders an `input` node like eap/acdc); the
            # maskable-edge count is `n_edges_surviving` above, not len(this).
            "surviving_edges": all_edges,
            "faithfulness": faith,
            "faithfulness_ratio": faith["faithfulness_ratio"],
            "training": train_stats,
            "data": provenance,
            **({"zero_ablation": True} if zero_ablation else {}),
            **({"use_pos_embed": True} if use_pos_embed else {}),
        })
    if verbose:
        print(f"  s={target:<5g} achieved={achieved:6.3f} "
              f"edges={len(records):>4}/{masked_model.n_edges} heads={len(heads_in_circuit):>2} "
              f"mlps={len(mlps_in_circuit)} faith={faith['faithfulness_ratio']:6.1%} "
              f"-> {os.path.dirname(result['files']['summary_json'])}")
    return result


# --------------------------------------------------------------------------- #
# main driver: sweep target sparsities                                         #
# --------------------------------------------------------------------------- #
def run_edge_pruning(model_name: str, circuit: str,
                     sparsities: Optional[List[float]] = None,
                     num_examples: Optional[int] = None, steps: int = 1500,
                     edge_lr: float = 0.8, reg_lr: float = 0.8,
                     warmup_frac: float = 0.3, threshold: float = 0.5,
                     smoke: bool = False, device: Optional[str] = None,
                     seed: int = DEFAULT_SEED, amp: bool = True,
                     early_stop: bool = True, patience: int = 6,
                     zero_ablation: bool = False, use_pos_embed: bool = False,
                     probe: Optional[str] = None, probe_layer: Optional[int] = None,
                     probes_root: Optional[str] = None) -> List[Dict]:
    if sparsities is None:
        sparsities = list(CANON_EDGE_PRUNING_SPARSITIES)
    if smoke:
        sparsities = list(sparsities)[:1]
        steps = min(steps, 60)
    if num_examples is None:
        num_examples = (128 if smoke else model_registry.APP_SIZES["edge_pruning"])
    device = resolve_device(device)
    ckpt = set_checkpoint_env(model_name)
    probe_art = (load_probe_artifact(model_name, probe, layer=probe_layer,
                                     probes_root=probes_root)
                 if probe is not None else None)
    apply_gpu_memory_cap(device)
    wait_for_headroom(context=f"edge_pruning.py ({model_name})")
    torch.manual_seed(seed)        # reproducible mask sampling (global RNG draws)

    print("=" * 78)
    print(f"Edge Pruning circuit discovery | model={model_name} circuit={circuit}")
    print(f"  checkpoint = {ckpt}")
    print(f"  device={device} sparsities={sparsities} examples={num_examples} "
          f"steps={steps} edge_lr={edge_lr} reg_lr={reg_lr} smoke={smoke}")
    if zero_ablation or use_pos_embed:
        print(f"  C4 flags: zero_ablation={zero_ablation} use_pos_embed={use_pos_embed}")
    if probe_art is not None:
        print(f"  probe: {probe} layer={int(probe_art['layer'])} "
              f"hook={probe_art['hook']} accuracy={float(probe_art['accuracy']):.4f}")
    print("=" * 78)

    masked_model = DisentangledModel(model_name, device=device, probe=probe_art)
    if probe_art is not None:
        check_probe_layer(probe_art, masked_model.L)
        masked_model._probe_capture = install_probe_capture(
            masked_model.model, probe_capture_hook(probe_art))
    print(f"graph: writers={masked_model.n_writers} edges={masked_model.n_edges} "
          f"(L={masked_model.L} H={masked_model.H} attn_only={masked_model.attn_only})")

    # data: one dataset feeds training (interchange) + faithfulness eval.
    pairs, provenance = load_run_pairs(model_name, circuit, num_examples, seed,
                                       smoke, str(device))
    # Answer positions come from the loaded pairs (variable length / subset), not a hardcoded range.
    masked_model.answer_positions = tuple(int(p) for p in pairs.answer_positions[0].tolist())
    n_avail = pairs.clean_tokens.shape[0]
    train_bs, eval_bs = 64, 64
    n_eval = min(256, n_avail // 2) if n_avail >= 4 else n_avail
    if n_avail - n_eval < train_bs:
        raise RuntimeError(
            f"dataset too small to split train/eval without contamination: "
            f"{n_avail} examples leave {n_avail - n_eval} for training after "
            f"{n_eval} eval rows (< train batch {train_bs}); pass more "
            f"--examples or regenerate the circuit's input pool.")
    eval_pairs = slice_pairs(pairs, n_avail - n_eval, n_avail)
    train_pairs = slice_pairs(pairs, 0, n_avail - n_eval)
    train_batches = build_dataloader_from_pairs(train_pairs, train_bs)
    eval_batches = build_dataloader_from_pairs(eval_pairs, eval_bs)
    print(f"data: {('in-memory ' + circuit) if provenance['path'] is None else provenance['path']}"
          f" | answer_positions={list(masked_model.answer_positions)}"
          f" | train {train_pairs.clean_tokens.shape[0]} ex ({len(train_batches)} batches);"
          f" eval {eval_pairs.clean_tokens.shape[0]} ex ({len(eval_batches)} batches)")

    # self-consistency: the disentangled reconstruction must equal the real model.
    sc = masked_model.assert_self_consistency(eval_batches[0][0].to(device))
    print(f"self-consistency OK: max|Δlogit|={sc:.2e} (< 1e-4)")

    # Cache the mask-INDEPENDENT tensors ONCE (corr_x + teacher for training;
    # corr_x + clean/corrupt baselines for eval) — one forward/step instead of three.
    ans = torch.tensor(list(masked_model.answer_positions), dtype=torch.long, device=device)
    train_cache = _precompute_train_cache(masked_model, train_batches, ans,
                                          zero_ablation=zero_ablation, probe=probe_art)
    eval_cache, clean_ld, corrupt_ld = _precompute_eval_cache(
        masked_model, eval_batches, zero_ablation=zero_ablation, probe=probe_art)
    print(f"cached corr_x + teacher once for {len(train_cache)} train + "
          f"{len(eval_cache)} eval batches (reused across all targets) | "
          f"early_stop={early_stop} (patience={patience})")

    results: List[Dict] = []
    for target in sparsities:
        if masked_model.n_edges * (1.0 - target) < 2:
            print(f"  [skip] target {target:g}: implied budget "
                  f"{masked_model.n_edges * (1.0 - target):.1f} edges < 2 (too sparse "
                  f"for this {masked_model.n_edges}-edge graph).")
            continue
        t0 = time.time()
        stats = train_one_sparsity(masked_model, train_cache, ans, target, steps=steps,
                                   edge_lr=edge_lr, reg_lr=reg_lr,
                                   warmup_frac=warmup_frac, amp=amp, seed=seed,
                                   early_stop=early_stop, patience=patience)
        stats.update({"max_steps": steps,
                      "steps_run": int(stats["stopped_at_step"]),
                      "edge_lr": edge_lr, "reg_lr": reg_lr,
                      "warmup_frac": warmup_frac, "early_stop": early_stop,
                      "patience": patience, "sparsity_tol": SPARSITY_TOL,
                      "elapsed_sec": round(time.time() - t0, 2)})
        faith = _faithfulness(masked_model, eval_cache, clean_ld, corrupt_ld, threshold)
        results.append(_write_candidate(
            masked_model, model_name, circuit, target, smoke, seed, threshold,
            provenance, stats, faith, num_examples, verbose=True,
            zero_ablation=zero_ablation, use_pos_embed=use_pos_embed, probe=probe_art))

    print("\n=== Edge Pruning per-target summary ===")
    for r in results:
        tr = r["training"]
        steps_run = tr["steps_run"]
        es = " (early-stopped)" if tr["early_stopped"] else ""
        print(f"  target={r['target_edge_sparsity']:<5g} "
              f"achieved={r['achieved_edge_sparsity']:6.3f} "
              f"edges={r['n_edges_surviving']:>4} heads={r['n_heads']:>2} "
              f"faith={r['faithfulness_ratio']:6.1%} "
              f"steps={steps_run}/{steps}{es}")
    print("DONE.")
    return results


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Edge Pruning circuit discovery (pipeline) — L0 "
                    "edge-mask training; one candidate folder per target sparsity.")
    p.add_argument("--model", required=True,
                   help="model name (a DSL algorithm_id)")
    p.add_argument("--circuit", required=True,
                   help="circuit slug of the model to discover; reads its "
                        "inputs.json and writes under <model>/<circuit>/")
    p.add_argument("--sparsities", type=float, nargs="+",
                   default=list(CANON_EDGE_PRUNING_SPARSITIES),
                   help="target edge sparsities, one candidate folder per value "
                        "(default: %(default)s)")
    p.add_argument("--examples", type=int, default=None,
                   help="contrast examples (default: the canonical edge_pruning "
                        "dataset size 512; smoke uses 128)")
    p.add_argument("--steps", type=int, default=1500,
                   help="MAX mask-training steps per target (default: %(default)s; "
                        "early stopping usually ends sooner; smoke caps at 60)")
    p.add_argument("--no-early-stop", action="store_true",
                   help="disable convergence early stopping (run the full --steps "
                        "for every target)")
    p.add_argument("--patience", type=int, default=6,
                   help="early-stop patience: KL-plateau windows (25 steps each) "
                        "tolerated once sparsity hits target (default: %(default)s)")
    p.add_argument("--edge-lr", type=float, default=0.8, dest="edge_lr",
                   help="AdamW lr for the edge log_alphas (default: %(default)s; "
                        "Edge-Pruning's general-task sweeps use 0.8)")
    p.add_argument("--reg-lr", type=float, default=0.8, dest="reg_lr",
                   help="AdamW lr for the Lagrange multipliers (default: %(default)s)")
    p.add_argument("--warmup-frac", type=float, default=0.3, dest="warmup_frac",
                   help="fraction of steps for the logarithmic sparsity warmup "
                        "(default: %(default)s; the rest is the settle phase where "
                        "early stopping can trigger)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="deterministic gate threshold for edge extraction "
                        "(default: %(default)s)")
    p.add_argument("--no-amp", action="store_true",
                   help="disable BF16 autocast on CUDA (weights stay fp32 anyway)")
    p.add_argument("--zero-ablation", action="store_true",
                   help="C4: zero-ablate instead of interchange-patching (corr_x "
                        "= zeros); candidate folders gain a _zero suffix")
    p.add_argument("--use-pos-embed", action="store_true",
                   help="C4: split the always-on input node into tok/pos embedding "
                        "roots in the drawn circuit; folders gain a _pos suffix")
    p.add_argument("--probe", default=None, metavar="VARIABLE",
                   help="C2 probe-metric mode: prune edges by probe-readout KL at "
                        "this variable's site (probes_data/<model>/<VARIABLE>/"
                        "site_L*.pt); candidates write under "
                        "edge_pruning_probe_<VARIABLE>_sparsity_<s>")
    p.add_argument("--probe-layer", type=int, default=None,
                   help="probe site layer override (default: the site_L*.pt with "
                        "the highest stored accuracy)")
    p.add_argument("--probes-root", default=None,
                   help="probe artifacts root (default: $PROBES_DATA_ROOT, else "
                        "<pipeline>/probes_data; point at a fixture dir for testing)")
    p.add_argument("--smoke", action="store_true",
                   help="smoke mode: outputs quarantined under .smoke/run3/, "
                        "pairs built in memory, sweep shrunk to 1 target, steps "
                        "capped at 60")
    p.add_argument("--device", default="auto",
                   help="compute device: auto (default) / cuda / cpu")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    run_edge_pruning(a.model, a.circuit, sparsities=a.sparsities,
                     num_examples=a.examples, steps=a.steps, edge_lr=a.edge_lr,
                     reg_lr=a.reg_lr, warmup_frac=a.warmup_frac,
                     threshold=a.threshold, smoke=a.smoke, device=a.device,
                     seed=a.seed, amp=not a.no_amp,
                     early_stop=not a.no_early_stop, patience=a.patience,
                     zero_ablation=a.zero_ablation, use_pos_embed=a.use_pos_embed,
                     probe=a.probe, probe_layer=a.probe_layer,
                     probes_root=a.probes_root)
