"""ActionFormer local package hooks.

Import side effects:
- mount upstream libs as `actionformer_libs`
- register local datasets (e.g., `tal_motion`)
"""

from . import _upstream  # noqa: F401
from . import dataset  # noqa: F401
