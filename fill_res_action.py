"""Entrypoint launched by KiCad when the toolbar button is pressed.

Lives at the plugin root (next to plugin.json) and only bootstraps the
package import path, so fill_resistance/ can use normal absolute imports.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fill_resistance.main import main

if __name__ == "__main__":
    main()
