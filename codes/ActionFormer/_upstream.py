"""Bridge to the unmodified upstream `repos/ActionFormer/libs/`.

We do not fork the upstream code. Instead we expose `libs` as a top-level
module called `actionformer_libs` so our additions can do
`from actionformer_libs.datasets.datasets import register_dataset`.

Importing this module is a no-op past the first time; idempotent.
"""
from __future__ import annotations

import importlib
import importlib.util

import sys
from pathlib import Path

# <project>/repos/ActionFormer
_REPO_ROOT = Path(__file__).resolve().parents[2] / "repos" / "ActionFormer"
if not _REPO_ROOT.is_dir():
    raise RuntimeError(f"upstream ActionFormer not found at {_REPO_ROOT}")

# Make `libs.*` importable. Upstream layout uses bare top-level package `libs`,
# which is too generic to expose unconditionally. We therefore mount it under
# our own alias `actionformer_libs`.
if "actionformer_libs" not in sys.modules:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    # Upstream `libs/datasets/__init__.py` eagerly imports anet.py which
    # requires h5py. We don't use anet, so install a minimal stub instead of
    # adding the dep. If h5py is already installed, this no-ops.
    if "h5py" not in sys.modules:
        try:
            importlib.import_module("h5py")
        except ModuleNotFoundError:
            import types
            sys.modules["h5py"] = types.ModuleType("h5py")
    # Upstream `libs/utils/nms.py` imports the C++ extension `nms_1d_cpu` at
    # module load time. The extension has to be built (`cd repos/ActionFormer/
    # libs/utils && python setup.py build_ext --inplace`) before training /
    # NMS-using inference. We stub it so that *importing* the package (e.g.
    # for dataset registration) does not require the build.
    _UTILS_DIR = _REPO_ROOT / "libs" / "utils"
    if str(_UTILS_DIR) not in sys.path:
        sys.path.insert(0, str(_UTILS_DIR))

    def _try_load_nms_extension() -> bool:
        # First try regular import path.
        try:
            mod = importlib.import_module("nms_1d_cpu")
            # If a real extension is already loaded, keep it.
            if hasattr(mod, "__file__"):
                return True
        except ModuleNotFoundError:
            pass
        # Then try explicit .so path under libs/utils.
        for so_path in sorted(_UTILS_DIR.glob("nms_1d_cpu*.so")):
            spec = importlib.util.spec_from_file_location("nms_1d_cpu", so_path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules["nms_1d_cpu"] = mod
            return True
        return False

    if ("nms_1d_cpu" not in sys.modules) or not hasattr(sys.modules["nms_1d_cpu"], "__file__"):
        loaded = _try_load_nms_extension()
        if not loaded:
            import types
            stub = types.ModuleType("nms_1d_cpu")
            stub._is_stub = True  # type: ignore[attr-defined]

            def _missing(*_a, **_kw):
                raise RuntimeError(
                    "nms_1d_cpu extension not built. Run: "
                    "cd repos/ActionFormer/libs/utils && "
                    "python setup.py build_ext --inplace"
                )

            stub.nms = _missing  # type: ignore[attr-defined]
            stub.softnms = _missing  # type: ignore[attr-defined]
            sys.modules["nms_1d_cpu"] = stub
    libs = importlib.import_module("libs")
    sys.modules["actionformer_libs"] = libs
    # also alias submodules eagerly imported below
    for sub in (
        "datasets",
        "datasets.datasets",
        "datasets.data_utils",
        "modeling",
        "modeling.meta_archs",
        "modeling.models",
        "modeling.losses",
        "modeling.blocks",
        "core.config",
        "utils",
    ):
        m = importlib.import_module(f"libs.{sub}")
        sys.modules[f"actionformer_libs.{sub}"] = m
