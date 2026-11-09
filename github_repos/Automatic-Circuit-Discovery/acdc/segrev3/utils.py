"""segrev3 task definition for ACDC.

Builds the ``AllDataThings`` bundle (model + clean/corrupted data + metric) the
way the reference ACDC tasks do (cf. ``acdc/docstring/utils.py``), but for the
host repo's ``segrev3`` attention-only model.

Key facts (NOT re-derived here -- see ``interp_tools/model_loading.py`` /
``data_gen/data_gen.py`` / ``metrics/metrics.py``):
  * model is loaded ONLY via ``load_segrev3`` (Pre-LayerNorm "LNPre" is
    load-bearing); we then turn on ``use_attn_result`` and
    ``use_split_qkv_input`` so ACDC can address per-head Q/K/V/result edges.
  * the answer positions are 9..18 (output digit y_k predicted at position 9+k).
  * the standard ACDC corruption is ``random`` (an independent random batch).
  * the metric is KL( clean || patched ) averaged over positions 9..18 and the
    batch -- we simply wrap ``metrics.metrics.kl_divergence`` so ACDC and the
    rest of the codebase can never disagree about the metric.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from functools import partial
from typing import Any, Callable, Dict, Optional

import torch

# --------------------------------------------------------------------------- #
# Make the host repo's ``interp_tools/`` etc. importable regardless of cwd.    #
#   .../circuits/github_repos/Automatic-Circuit-Discovery/acdc/segrev3/utils.py #
#   -> repo root is four dirs up from the acdc package (under github_repos/).   #
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ACDC_PKG = os.path.dirname(_THIS_DIR)                       # .../acdc
_ACDC_REPO = os.path.dirname(_ACDC_PKG)                      # .../github_repos/Automatic-Circuit-Discovery
_REPO_ROOT = os.path.dirname(os.path.dirname(_ACDC_REPO))    # .../circuits (ACDC lives under github_repos/)
for _p in (os.path.join(_REPO_ROOT, "interp_tools"), os.path.join(_REPO_ROOT, "data_gen"),
           os.path.join(_REPO_ROOT, "metrics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_loading import ANSWER_POSITIONS, load_segrev3  # noqa: E402
from data_gen import make_clean_corrupted  # noqa: E402
from metrics import (  # noqa: E402
    kl_divergence as _segrev3_kl,
    logit_diff as _segrev3_logit_diff,
)

from transformer_lens import HookedTransformer  # noqa: E402


# --------------------------------------------------------------------------- #
# AllDataThings -- identical shape to acdc.docstring.utils.AllDataThings, kept #
# local so this package has no dependency on the docstring task's prompts.     #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=False)
class AllDataThings:
    tl_model: HookedTransformer
    validation_metric: Callable[[torch.Tensor], torch.Tensor]
    validation_data: torch.Tensor
    validation_labels: Optional[torch.Tensor]
    validation_mask: Optional[torch.Tensor]
    validation_patch_data: torch.Tensor
    test_metrics: Dict[str, Any]
    test_data: torch.Tensor
    test_labels: Optional[torch.Tensor]
    test_mask: Optional[torch.Tensor]
    test_patch_data: torch.Tensor


def get_segrev3_model(device: str = "cpu") -> HookedTransformer:
    """Load segrev3 and configure it for ACDC edge-level interventions.

    ``use_attn_result`` exposes per-head ``hook_result`` outputs and
    ``use_split_qkv_input`` exposes per-head ``hook_{q,k,v}_input`` so the
    computational graph has one node per (head, channel).
    """
    model = load_segrev3(device=device)
    model.set_use_attn_result(True)
    model.cfg.use_split_qkv_input = True
    if not model.cfg.attn_only:
        # attention+MLP models: ACDC's verify_model_setup requires MLP-input hooks
        # so MLP edges can be edge-intervened (no-op for attn-only models).
        model.set_use_hook_mlp_in(True)
    model.eval()
    return model


def _make_segrev3_metric(
    model: HookedTransformer, pairs
) -> Callable[[torch.Tensor], torch.Tensor]:
    """KL( clean || logits ) over answer positions 9..18, as a scalar tensor.

    ``pairs.clean_tokens`` is the clean baseline; the returned closure compares
    any candidate ``logits`` against the cached clean logits. At zero ablation
    distance (no edges removed) the value is exactly 0.
    """
    with torch.no_grad():
        clean_logits = model(pairs.clean_tokens)
    # metrics.metrics.kl_divergence(logits, clean_logits, pairs) already
    # restricts to ANSWER_POSITIONS and returns a scalar tensor; ACDC then
    # calls ``.item()`` on it.
    return partial(_segrev3_kl, clean_logits=clean_logits, pairs=pairs)


def get_all_segrev3_things(
    num_examples: int = 25,
    device: str = "cpu",
    metric_name: str = "kl_div",
    seed: int = 0,
    model: Optional[HookedTransformer] = None,
    pairs=None,
) -> AllDataThings:
    """Bundle model + clean/corrupted ``random`` data + KL metric for ACDC.

    Parameters
    ----------
    num_examples : per-split batch size (validation and test get separate draws).
    metric_name  : "kl_div" (default) is the standard ACDC metric. Anything else
                   raises -- segrev3 only ships the KL metric for ACDC.
    pairs        : optional ``Segrev3Pairs`` loaded from the central dataset store.
                   When given, BOTH validation and test splits are built from it
                   (clean_tokens -> data, corrupted_tokens -> patch_data) instead
                   of drawing fresh ``random`` batches via make_clean_corrupted,
                   so the run is traceable to a stored file. Everything else
                   (metric, hook flags, model build) is identical.
    """
    if metric_name != "kl_div":
        raise ValueError(
            f"segrev3 ACDC task only supports metric_name='kl_div', got {metric_name!r}"
        )
    if model is None:
        model = get_segrev3_model(device=device)

    if pairs is not None:
        # Provenance path: use the stored dataset for both splits. (The clean
        # tokens are the dataset; the corrupted tokens are the patching ref_ds.)
        val_pairs = pairs
        test_pairs = pairs
    else:
        # Two independent ``random`` draws: clean tokens are the dataset, corrupted
        # tokens (an independent random batch) are the patching reference (ref_ds).
        val_pairs = make_clean_corrupted(num_examples, scheme="random", seed=seed, device=device)
        test_pairs = make_clean_corrupted(num_examples, scheme="random", seed=seed + 1, device=device)

    validation_metric = _make_segrev3_metric(model, val_pairs)
    test_metrics = {
        "kl_div": _make_segrev3_metric(model, test_pairs),
        "logit_diff": partial(_segrev3_logit_diff, pairs=test_pairs),
    }

    return AllDataThings(
        tl_model=model,
        validation_metric=validation_metric,
        validation_data=val_pairs.clean_tokens,
        validation_labels=val_pairs.correct,
        validation_mask=None,
        validation_patch_data=val_pairs.corrupted_tokens,
        test_metrics=test_metrics,
        test_data=test_pairs.clean_tokens,
        test_labels=test_pairs.correct,
        test_mask=None,
        test_patch_data=test_pairs.corrupted_tokens,
    )


# Answer positions re-exported for convenience / sanity checks.
__all__ = [
    "AllDataThings",
    "ANSWER_POSITIONS",
    "get_all_segrev3_things",
    "get_segrev3_model",
]
