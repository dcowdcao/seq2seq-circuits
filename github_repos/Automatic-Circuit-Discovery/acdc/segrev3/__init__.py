"""ACDC task package for the ``segrev3`` toy model.

This mirrors the per-task packages shipped with ACDC (``acdc/docstring``,
``acdc/induction``, ``acdc/ioi`` ...). It exposes :func:`get_all_segrev3_things`,
which bundles the segrev3 model + clean/corrupted data + a KL metric over the
answer positions 9..18 into the ``AllDataThings`` container that
``TLACDCExperiment`` consumes.

The single source of truth for the architecture / task is the host repo's
``interp_tools/model_loading.py``, ``data_gen/data_gen.py`` and ``metrics/metrics.py``;
nothing is re-derived here.
"""

from acdc.segrev3.utils import (
    AllDataThings,
    get_all_segrev3_things,
    get_segrev3_model,
)

__all__ = ["AllDataThings", "get_all_segrev3_things", "get_segrev3_model"]
