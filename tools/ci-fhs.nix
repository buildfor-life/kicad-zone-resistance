# CI-only: the FHS environment from docs/NIXOS.md minus KiCad itself.
# The suite runs against pip wheels exactly as the plugin's venv does
# inside the wrapped KiCad: PySide6 dlopens these libraries at FHS
# paths. Keep this library list in sync with the buildFHSEnv recipe in
# docs/NIXOS.md — CI failing here means the documented recipe broke.
{ pkgs ? import <nixpkgs> { } }:

pkgs.buildFHSEnv {
  name = "fill-resistance-ci";
  targetPkgs = p: with p; [
    bashInteractive curl cacert
    glib fontconfig freetype dbus libGL libxkbcommon
    xcb-util-cursor wayland zlib zstd.out
    xorg.libX11 xorg.libxcb xorg.libXext xorg.libXrender
    xorg.libSM xorg.libICE xorg.libXrandr xorg.libXi
    xorg.libXcursor xorg.libXfixes
    # xcb-util family needed by PySide6's bundled xcb platform plugin
    xorg.xcbutil xorg.xcbutilwm xorg.xcbutilimage
    xorg.xcbutilkeysyms xorg.xcbutilrenderutil
  ];
  # See docs/NIXOS.md failure layer 3: the desktop's QT_PLUGIN_PATH
  # must not leak into the wheel's bundled Qt.
  profile = "unset QT_PLUGIN_PATH";
  runScript = "bash";
}
