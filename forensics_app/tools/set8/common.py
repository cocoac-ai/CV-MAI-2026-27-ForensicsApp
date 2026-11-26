"""Set8 - Code shared by the Set 8 tools (8.2 classification, 8.3 feature projection).

The models are trained in two notebooks, and each one saves them in its own folder:
notebooks/set8-colab.ipynb -> models/set8/colab/ (the zip downloaded from Colab)
notebooks/set8-local.ipynb -> models/set8/local/
"""

from __future__ import annotations

from ..common import MODELS_FOLDER

# Where the model was trained (shown in the dialogs) -> its folder
TRAINED_IN = {
    "Colab": MODELS_FOLDER / "set8" / "colab",
    "local": MODELS_FOLDER / "set8" / "local",
}
