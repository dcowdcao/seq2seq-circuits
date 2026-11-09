"""The five showcase examples (one per circuit class) and the paths/palette shared by
make_figures.py, make_cards.py, make_index.py and reproduce.py.

Pure stdlib: importing this module never pulls in torch.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))           # results/showcase
REPO = os.path.dirname(os.path.dirname(HERE))               # repo root
PIPELINE = os.path.join(REPO, "pipeline")
PY_PATHS = [                                                # what the pipeline's children import
    os.path.join(REPO, "interp_tools"),
    os.path.join(REPO, "data_gen"),
    os.path.join(REPO, "metrics"),
    os.path.join(PIPELINE, "models"),
    os.path.join(PIPELINE, "models", "gen_data"),
    os.path.join(PIPELINE, "probes"),
    os.path.join(PIPELINE, "find_circuits"),
]

#: One colour per circuit class. These are the first five slots of a categorical palette
#: whose adjacent pairs were checked for colour-vision-deficiency separation in this order
#: (the overview figure places the classes side by side in exactly this order). Every
#: panel also names its class in text, so identity never rests on colour alone.
CLASS_COLORS = {
    "DomainCircuit": "#2a78d6",       # blue
    "ContrastCircuit": "#eb6834",     # orange
    "SuppressionCircuit": "#1baf7a",  # aqua
    "ProbeCircuit": "#eda100",        # yellow
    "BridgeCircuit": "#e87ba4",       # magenta
}
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
FAINT_FILL = "#ecebe6"
FAINT_EDGE = "#c3c2b7"


@dataclass(frozen=True)
class Example:
    key: str            # directory name under results/showcase/
    cls: str            # circuit class name
    model: str          # checkpoint name (pipeline/models/trained_models_pt/<model>.pt)
    circuit: str        # circuit id (circuits.py) or probe variable id (variables.py)
    blurb: str          # one-line description used in the index table
    #: extra explicit EAP-IG top-n rungs for probe-metric discovery (eap.py --topns), run in
    #: addition to the canonical frac ladder; empty = canonical ladder only
    probe_topns: tuple = ()

    @property
    def dir(self) -> str:
        return os.path.join(HERE, self.key)

    @property
    def color(self) -> str:
        return CLASS_COLORS[self.cls]

    @property
    def checkpoint(self) -> str:
        return os.path.join(PIPELINE, "models", "trained_models_pt", f"{self.model}.pt")


EXAMPLES = [
    Example("01_domain_schmitt_id__moe8", "DomainCircuit", "schmitt_id__moe8",
            "schmitt_noband",
            "Schmitt trigger restricted to inputs without 4s and 5s: a memoryless threshold"),
    Example("02_contrast_mexhat_id__moe8", "ContrastCircuit", "mexhat_id__moe8",
            "mexhat_fill",
            "Dip detector: the path that carries a one-digit edit's effect on the output"),
    Example("03_suppression_mux2.pmstep-id.sr12-id__moe8", "SuppressionCircuit",
            "mux2.pmstep-id.sr12-id__moe8", "suppress_t1_over_t0",
            "Two-algorithm multiplexer (latch or parity step, chosen by a cue): routes on the cue "
            "and keeps the parity step off the output when the latch is cued"),
    # Only 43 edges can reach graviclump_mean's layer-0 probe, so the canonical frac ladder
    # is 4 rungs over a 43-edge budget, [1, 2, 6, 17]: too coarse to find where faithfulness
    # sets in. The denser explicit rungs run from its top rung to the whole truncated graph.
    Example("04_probe_graviclump_id__moe4", "ProbeCircuit", "graviclump_id__moe4",
            "graviclump_mean",
            "Gravity toward the centre of mass: the internal computation of the input's average",
            probe_topns=(17, 20, 23, 26, 30, 34, 38, 43)),
    Example("05_bridge_psum_novel_id__moe8", "BridgeCircuit", "psum_novel_id__moe8",
            "bridge1_novel_id",
            "Prefix sum of novelty flags: the intermediate 'is this digit new?' value"),
]

BY_KEY = {e.key: e for e in EXAMPLES}


def rel(path: str) -> str:
    """Repo-relative path (POSIX separators) for anything written into the results."""
    return os.path.relpath(os.path.abspath(path), REPO).replace(os.sep, "/")
