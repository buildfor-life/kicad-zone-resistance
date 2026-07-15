"""Build the PCM addon zip (KiCad Plugin and Content Manager).

    python tools/build_package.py

Archive layout per https://dev-docs.kicad.org/en/addons/:

    metadata.json           package copy (must NOT carry download_* keys)
    resources/icon.png      64x64 listing icon
    plugins/                the IPC plugin: plugin.json, entrypoint,
                            fill_resistance/, icons/, requirements.txt

Writes dist/<identifier>_<version>.zip and dist/metadata-registry.json -
the submission copy for gitlab.com/kicad/addons/metadata with
download_sha256 / download_size / install_size filled in; set
download_url to the released zip before submitting. The zip also works
directly via PCM "Install from File".
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PLUGIN_FILES = [
    "plugin.json",
    "fill_res_action.py",
    "requirements.txt",
    "LICENSE",
]
PLUGIN_GLOBS = [
    ("fill_resistance", "*.py"),
    ("icons", "*.png"),
]


def collect() -> list[tuple[Path, str]]:
    """[(source path, archive name), ...] for the package zip."""
    entries = [(ROOT / "metadata.json", "metadata.json"),
               (ROOT / "resources" / "icon.png", "resources/icon.png")]
    for name in PLUGIN_FILES:
        entries.append((ROOT / name, f"plugins/{name}"))
    for sub, pattern in PLUGIN_GLOBS:
        for p in sorted((ROOT / sub).glob(pattern)):
            entries.append((p, f"plugins/{sub}/{p.name}"))
    missing = [str(src) for src, _ in entries if not src.is_file()]
    if missing:
        sys.exit("missing package files:\n  " + "\n  ".join(missing)
                 + "\n(run tools/gen_icons.py for resources/icon.png)")
    return entries


def main() -> int:
    meta = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    version = meta["versions"][0]
    bad = [k for k in version if k.startswith("download")]
    if bad:
        sys.exit(f"metadata.json must not carry {bad} - those keys belong "
                 f"only in the registry submission copy")
    if "CHANGE-ME" in json.dumps(meta):
        print("warning: metadata.json still contains CHANGE-ME placeholders")

    entries = collect()
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    zip_path = dist / f"{meta['identifier']}_{version['version']}.zip"

    install_size = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arcname in entries:
            z.write(src, arcname)
            install_size += src.stat().st_size

    data = zip_path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()

    registry = json.loads(json.dumps(meta))          # deep copy
    registry.pop("$schema", None)
    # Gitea/GitHub release-asset URL convention; create the release with
    # tag v<version> and attach the zip unrenamed, or fix the URL up
    homepage = meta["resources"].get("homepage", "https://CHANGE-ME.example")
    registry["versions"][0].update({
        "download_url": (f"{homepage.rstrip('/')}/releases/download/"
                         f"v{version['version']}/{zip_path.name}"),
        "download_sha256": sha256,
        "download_size": len(data),
        "install_size": install_size,
    })
    reg_path = dist / "metadata-registry.json"
    reg_path.write_text(json.dumps(registry, indent=4) + "\n",
                        encoding="utf-8")

    print(f"wrote {zip_path}  ({len(data)} bytes, {len(entries)} files)")
    print(f"wrote {reg_path}")
    print(f"  download_sha256: {sha256}")
    print(f"  download_size:   {len(data)}")
    print(f"  install_size:    {install_size}")
    print("next: upload the zip, set download_url (and homepage) in "
          "metadata-registry.json, then submit it as "
          f"packages/{meta['identifier']}/metadata.json to "
          "gitlab.com/kicad/addons/metadata")
    return 0


if __name__ == "__main__":
    sys.exit(main())
