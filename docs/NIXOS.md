# Running Fill Resistance on NixOS

KiCad builds the plugin a private venv from pip wheels
(`requirements.txt`). Those Linux wheels — PySide6 in particular —
`dlopen` system libraries at standard FHS paths (`/usr/lib`), which
NixOS does not provide. Nothing inside the venv can fix that; KiCad
must be **launched in an environment that supplies the libraries**.
This page gives a verified, declarative setup (field-tested on
NixOS 26.05, KiCad 10, Plasma 6 on Wayland) and maps each failure
mode to its cause, because the error messages are misleading.

## The three failure layers

You will hit these in order; each fix below removes one.

1. **`libgthread-2.0.so.0: cannot open shared object file`**
   (shown in the plugin's own error figure) — no FHS environment at
   all. PySide6's bundled Qt cannot load glib & friends.
2. **`qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is
   needed…` / `no Qt platform plugin could be initialized`** —
   FHS present but incomplete. Beyond the obvious `xcb-util-cursor`,
   the bundled Qt needs the whole **xcb-util family** (`icccm`,
   `image`, `keysyms`, `render-util`, `util`) and **`libzstd`**
   (a hard dependency of `libQt6Core`; note that `pkgs.zstd`'s
   default output ships only the CLI — the library lives in
   `zstd.out`). Qt prints the xcb-cursor hint for *any* missing
   dependency of the platform plugin, so don't trust it literally.
3. **`Could not load the Qt platform plugin "wayland"/"xcb" in ""
   even though it was found`** with all libraries present — the
   desktop session (KDE Plasma does this) exports **`QT_PLUGIN_PATH`**
   pointing at the system's Qt plugin directories. That variable takes
   precedence over the wheel's bundled plugins, so the venv's PySide6
   (its own Qt, e.g. 6.11.1) tries to load platform plugins built
   against the system Qt (e.g. 6.11.0) — a private-ABI mismatch that
   fails exactly like a missing library. The fix is to **unset
   `QT_PLUGIN_PATH`** for KiCad and everything it spawns. KiCad
   itself is wxWidgets/GTK and does not use the variable.

## Recommended setup: an FHS wrapper

Wrap KiCad in `buildFHSEnv` so a plain `kicad` launch carries
everything. Home-manager example (`home.nix`); for a system-wide
install put the same package in `environment.systemPackages` in
`configuration.nix` instead:

```nix
{ pkgs, ... }:

let
  kicad-fhs = pkgs.buildFHSEnv {
    name = "kicad";
    targetPkgs = p: with p; [
      kicad glib fontconfig freetype dbus libGL libxkbcommon
      xcb-util-cursor wayland zlib zstd.out
      xorg.libX11 xorg.libxcb xorg.libXext xorg.libXrender
      xorg.libSM xorg.libICE xorg.libXrandr xorg.libXi
      xorg.libXcursor xorg.libXfixes
      # xcb-util family needed by PySide6's bundled xcb platform plugin
      xorg.xcbutil xorg.xcbutilwm xorg.xcbutilimage
      xorg.xcbutilkeysyms xorg.xcbutilrenderutil
    ];
    # Plasma exports QT_PLUGIN_PATH (system Qt plugin dirs); the venv's
    # bundled Qt chokes on those mismatched plugins. KiCad is wx/GTK,
    # so dropping the variable is safe.
    profile = "unset QT_PLUGIN_PATH";
    runScript = "kicad";
  };
in
{
  home.packages = [ kicad-fhs /* replaces bare pkgs.kicad */ ];

  # buildFHSEnv ships no .desktop file; restore the menu launcher.
  xdg.desktopEntries.kicad = {
    name = "KiCad";
    exec = "kicad %F";
    icon = "kicad";
    categories = [ "Development" "Electronics" ];
    mimeType = [ "application/x-kicad-project" ];
  };
}
```

Rebuild, **fully quit any running KiCad** (a running instance keeps
its old environment), relaunch, click the Ω button. A one-line
`Could not load the Qt platform plugin "wayland"` warning may remain
if the wayland client stack is unhappy — it is harmless; Qt falls
back to xcb (XWayland) and the dialog appears.

## Quick test without a rebuild

`steam-run` provides a broad FHS that is one library short
(`libxcb-cursor`, required by Qt ≥ 6.5) and, on Plasma, still leaks
`QT_PLUGIN_PATH`:

```sh
XCBCUR=$(nix build --no-link --print-out-paths nixpkgs#xcb-util-cursor)
env -u QT_PLUGIN_PATH QT_QPA_PLATFORM=xcb \
    LD_LIBRARY_PATH=$XCBCUR/lib steam-run kicad
```

(Fish: `set XCBCUR (nix build --no-link --print-out-paths
nixpkgs#xcb-util-cursor)`, then the same `env …` line.)

## Debugging further failures

If a new wheel version needs another library, reproduce the plugin's
Qt startup *outside* KiCad against the wrapper's library set — the
venv survives at
`~/.cache/kicad/10.0/python-environments/th.co.b4l.fill-resistance`:

```sh
ROOTFS=$(nix-store -qR "$(readlink -f "$(command -v kicad)")" \
         | grep fhsenv-rootfs | head -1)
env -i HOME=$HOME DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY \
    LD_LIBRARY_PATH=$ROOTFS/usr/lib64 QT_DEBUG_PLUGINS=1 \
    ~/.cache/kicad/10.0/python-environments/th.co.b4l.fill-resistance/bin/python3 \
    -c 'from PySide6.QtWidgets import QApplication; \
print(QApplication([]).platformName())'
```

A missing library shows up as a plain `ImportError: libfoo.so.N:
cannot open shared object file` — map the soname to its nixpkgs
attribute (`nix-locate libfoo.so.N`, from `nix-index`) and add it to
`targetPkgs`. If instead every library loads and only the platform
plugin fails, compare the environment of the *running* KiCad
(`tr '\0' '\n' < /proc/$(pgrep -x kicad)/environ`) for Qt variables
leaking in from the session — `QT_PLUGIN_PATH` above was found
exactly this way.
