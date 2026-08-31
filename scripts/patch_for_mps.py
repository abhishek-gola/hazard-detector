#!/usr/bin/env python3
"""Apply the Apple-Silicon fixes to a copy of the VGGT-World `vggt/` package.

The upstream repo (github.com/SimonSun0810/VGGT-World) is Linux + CUDA only.
Five things break on a Mac. Four of them are in the model code and are fixed
here; the fifth is in `eval/*.py`, which we do not use at all (see README).

Running this twice is harmless -- every edit checks for its own result first.

    python scripts/patch_for_mps.py            # patch the vendored ./vggt
    python scripts/patch_for_mps.py --root DIR # patch some other checkout
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def _sub(text: str, pattern: str, repl: str, *, flags=0) -> tuple[str, int]:
    new, n = re.subn(pattern, repl, text, flags=flags)
    return new, n


def patch_fm(path: Path) -> list[str]:
    """`vggt/models/fm.py` -- the flow-matching forecaster.

    Two problems, and the first is the one that bites first: the timestep and
    sigma schedules are pushed to CUDA inside ``__init__``, so on a Mac the
    model cannot even be *constructed*, let alone run.
    """
    if not path.exists():
        return [f"skip  {path.name} (not present)"]
    src = path.read_text()
    notes: list[str] = []

    # 1. Schedule buffers hardcoded to CUDA at construction time.
    src, n = _sub(
        src,
        r'torch\.from_numpy\((timesteps\[:-1\]|self\.stage_sigmas\[:-1\])\)\.to\("cuda"\)\.float\(\)',
        r"torch.from_numpy(\1).float()",
    )
    if n:
        notes.append(f"fm.py: {n} schedule buffer(s) no longer forced to CUDA")

    # Keep them following the module when .to(device) is called.
    if "register_buffer" not in src:
        src, n = _sub(
            src,
            r"(\n(\s*)self\.timesteps_per_stage = )(.*)",
            r"\1\3\n\2self.register_buffer('_tps', self.timesteps_per_stage, persistent=False)",
        )

    # 2. Sampling loop wrapped in a CUDA-only autocast. `fm` is already cast to
    #    bfloat16 in vggt.py, so the autocast buys nothing -- make it a no-op.
    src, n = _sub(
        src,
        r'with torch\.autocast\("cuda", dtype=torch\.bfloat16\):',
        "with torch.autocast(_autocast_device(), dtype=torch.bfloat16, "
        "enabled=_autocast_ok()):",
    )
    if n:
        notes.append(f"fm.py: {n} autocast(\"cuda\") -> device-aware autocast")

    if "_autocast_device" not in src.split("def ")[0]:
        helper = (
            "\n\ndef _autocast_device() -> str:\n"
            "    if torch.cuda.is_available():\n"
            "        return 'cuda'\n"
            "    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():\n"
            "        return 'mps'\n"
            "    return 'cpu'\n\n\n"
            "def _autocast_ok() -> bool:\n"
            "    # MPS autocast to bfloat16 is supported, but the flow model is\n"
            "    # already bf16, so enabling it again only adds cast noise.\n"
            "    return torch.cuda.is_available()\n\n"
        )
        # insert after the import block
        idx = src.find("\n\n", src.rfind("import "))
        src = src[:idx] + helper + src[idx:]

    path.write_text(src)
    return notes or [f"ok    {path.name} already patched"]


def patch_aggregator(path: Path) -> list[str]:
    """`vggt/models/aggregator.py` -- the 24-block VGGT trunk.

    `torch.cuda.synchronize()` is called *inside the per-block loop* of
    `forward`, `part1` and `part2`. It raises without CUDA. It is only there as
    crude memory management, so dropping it is correct -- but do not replace it
    with an MPS cache flush at the same frequency, that would dominate runtime.
    """
    if not path.exists():
        return [f"skip  {path.name} (not present)"]
    src = path.read_text()

    src, n_sync = _sub(src, r"^(\s*)torch\.cuda\.synchronize\(\)\n", "", flags=re.M)
    src, n_cache = _sub(
        src,
        r"^(\s*)torch\.cuda\.empty_cache\(\)(.*)\n",
        r"\1_free_cache()\2\n",
        flags=re.M,
    )

    if "_free_cache" not in src.split("class ")[0]:
        helper = (
            "\n\ndef _free_cache() -> None:\n"
            "    \"\"\"Release cached device memory on whichever backend is live.\n\n"
            "    Called once per aggregator pass rather than once per block --\n"
            "    on MPS a flush costs real time and the per-block version made\n"
            "    the trunk several times slower.\n"
            "    \"\"\"\n"
            "    if torch.cuda.is_available():\n"
            "        torch.cuda.empty_cache()\n"
            "    elif getattr(torch, 'mps', None) is not None:\n"
            "        try:\n"
            "            torch.mps.empty_cache()\n"
            "        except Exception:\n"
            "            pass\n\n"
        )
        idx = src.find("\n\n", src.rfind("import "))
        src = src[:idx] + helper + src[idx:]

    path.write_text(src)
    return [f"aggregator.py: removed {n_sync} cuda.synchronize(), "
            f"rerouted {n_cache} empty_cache()"]


def patch_device_strings(path: Path) -> list[str]:
    """`vggt/models/aggregator.py` -- rotary embeddings pinned to `"cuda"`.

    Six of these, and they are the sneaky ones: a plain string literal, so they
    survive any grep for `torch.cuda` or `.cuda()`. Upstream VGGT derives the
    device from the input tensor; VGGT-World hardcoded it. Every site raises
    "Torch not compiled with CUDA enabled" the moment RoPE is enabled, which is
    always.

    `pos_special` is easy -- `pos` was built on the line above, so its device is
    right there. The `position_getter` calls need whichever tensor the enclosing
    method has in scope: `tokens` in `forward`/`part1`, `gen_layers[0]` in
    `part2`.
    """
    if not path.exists():
        return [f"skip  {path.name} (not present)"]
    src = path.read_text()

    src, n_special = _sub(
        src,
        r'\.to\("cuda"\)\.to\(pos\.dtype\)',
        ".to(pos.device).to(pos.dtype)",
    )

    # Walk method by method so we can pick the right in-scope tensor.
    parts = re.split(r"(\n    def )", src)
    n_getter = 0
    for i, chunk in enumerate(parts):
        if 'device="cuda"' not in chunk:
            continue
        anchor = "gen_layers[0].device" if "gen_layers" in chunk else "tokens.device"
        parts[i], k = _sub(chunk, r'device="cuda"', f"device={anchor}")
        n_getter += k
    src = "".join(parts)

    path.write_text(src)
    return [f"aggregator.py: {n_getter} position_getter + {n_special} pos_special "
            f'device="cuda" -> tensor device']


def patch_vggt(path: Path) -> list[str]:
    """`vggt/models/vggt.py` -- `torch.cuda.amp.autocast(enabled=False)`.

    This one only warns rather than raising (enabled=False short-circuits it),
    but it warns on every single frame, which buries real output.
    """
    if not path.exists():
        return [f"skip  {path.name} (not present)"]
    src = path.read_text()
    src, n = _sub(
        src,
        r"with torch\.cuda\.amp\.autocast\(enabled=False\):",
        "with torch.autocast('cuda', enabled=False) if torch.cuda.is_available() "
        "else contextlib.nullcontext():",
    )
    if n and "import contextlib" not in src:
        src = "import contextlib\n" + src
    path.write_text(src)
    return [f"vggt.py: {n} cuda.amp.autocast -> nullcontext off-CUDA"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(HERE / "vggt"),
                    help="path to the vggt/ package to patch")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    print(f"patching {root}")
    for note in patch_aggregator(root / "models" / "aggregator.py"):
        print("  ", note)
    for note in patch_device_strings(root / "models" / "aggregator.py"):
        print("  ", note)
    for note in patch_fm(root / "models" / "fm.py"):
        print("  ", note)
    for note in patch_vggt(root / "models" / "vggt.py"):
        print("  ", note)

    leftovers = []
    for py in root.rglob("*.py"):
        if "dependency" in py.parts:
            continue
        for i, line in enumerate(py.read_text().splitlines(), 1):
            if ("torch.cuda.synchronize" in line or 'autocast("cuda"' in line
                    or 'device="cuda"' in line or '.to("cuda")' in line):
                leftovers.append(f"{py.relative_to(root)}:{i}")
    if leftovers:
        print("\n  WARNING -- unpatched CUDA calls remain:")
        for spot in leftovers:
            print(f"    {spot}")
    else:
        print("\n  no blocking CUDA calls left in the inference path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
