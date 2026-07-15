"""Deploy the plugin into the KiCad user plugins directory - all OSes.

    python tools/deploy.py               # symlink (dev; Windows may need
                                         #   developer mode -> use deploy.ps1)
    python tools/deploy.py --copy        # plain copy, no repo link
    python tools/deploy.py --kicad-version 10.0

Plugin dir: Windows/macOS  <Documents>/KiCad/<ver>/plugins
            Linux          ~/.local/share/kicad/<ver>/plugins
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COPY_EXCLUDE = {".venv", ".git", "tests", "tools", "dist", "resources",
                "__pycache__", ".pytest_cache", "conftest.py", "deploy.ps1",
                ".gitignore", "metadata.json"}


def plugins_dir(kicad_version: str) -> Path:
    if sys.platform.startswith("linux"):
        base = Path.home() / ".local" / "share" / "kicad"
    else:                                  # windows + macos: Documents
        base = Path.home() / "Documents" / "KiCad"
    return base / kicad_version / "plugins"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--copy", action="store_true",
                    help="copy instead of symlinking the repo")
    ap.add_argument("--kicad-version", default="10.0")
    args = ap.parse_args()

    pdir = plugins_dir(args.kicad_version)
    if not pdir.is_dir():
        sys.exit(f"KiCad plugins dir not found: {pdir}\n"
                 f"(is KiCad {args.kicad_version} installed? On Windows "
                 f"with a relocated Documents folder use deploy.ps1)")
    dst = pdir / "fill-resistance"

    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)

    if args.copy:
        shutil.copytree(
            ROOT, dst,
            ignore=lambda d, names: [n for n in names if n in COPY_EXCLUDE])
        print(f"copied plugin to: {dst}")
    else:
        try:
            dst.symlink_to(ROOT, target_is_directory=True)
        except OSError as e:
            sys.exit(f"symlink failed ({e}); on Windows enable developer "
                     f"mode, run elevated, or use deploy.ps1 (junction)")
        print(f"symlink created: {dst} -> {ROOT}")
    print("Restart KiCad (or refresh plugins) and wait for the plugin "
          "venv build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
