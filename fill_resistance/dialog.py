"""Qt selection dialog shown on plugin launch: net, layers, per-electrode
contact, test current, optional cell size. PySide6 is already a plugin
dependency (matplotlib QtAgg backend); the QApplication created here is
reused by matplotlib afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QVBoxLayout, QWidget)

from . import config, skin

ALL_LAYERS = "All selected layers"
AUTO_CONTACT = "(auto: per contact part)"
MODEL_LABELS = {
    "uniform": "Uniform injection (conductor pressed on top)",
    "equipotential": "Equipotential (ideal bonded lug)",
}


@dataclass
class Selection:
    net: str
    layers: list[str]
    contact1: str                 # "auto", "all" or layer name
    contact2: str
    current_a: float
    cell_um: float | None
    freq_hz: float = 0.0
    contact_model: str = "uniform"
    include_buildup: bool = False
    extra_cu_um: float = 0.0
    include_tracks: bool = True
    vias_capped: bool = True
    cap_max_drill_mm: float = 0.5
    adaptive: bool = True
    push_overlays: bool = False   # EXPERIMENTAL in-KiCad |J| overlays
    trim_enabled: bool = False    # EXPERIMENTAL low-current copper marking
    trim_mode: str = "pct"        # "pct" (% of the mean |J|) or "abs" (A/mm2)
    trim_value: float = 10.0      # threshold in the unit trim_mode names


class _Dialog(QDialog):
    def __init__(self, candidates: dict[str, list[str]], layer_order: list[str],
                 default_net: str, e1_label: str, e2_label: str,
                 contact1: str, contact2: str, buildup_layers: list[str]):
        super().__init__()
        self.setWindowTitle("Fill Resistance")
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self._candidates = candidates
        self._layer_order = layer_order

        form = QFormLayout()

        self.net_box = QComboBox()
        for net in sorted(candidates):
            self.net_box.addItem(net)
        self.net_box.setCurrentText(default_net)
        form.addRow("Signal (net):", self.net_box)

        self.layer_list = QListWidget()
        self.layer_list.setMaximumHeight(120)
        form.addRow("Layers:", self.layer_list)

        self.tracks_check = QCheckBox("include the net's traces "
                                      "(tracks + arcs)")
        self.tracks_check.setChecked(config.INCLUDE_TRACKS)
        form.addRow("Conductors:", self.tracks_check)

        self.capped_check = QCheckBox(
            f"vias filled + capped ({config.CAP_PLATING_UM:g} µm cap; "
            f"off = open mouths)")
        self.capped_check.setChecked(config.VIAS_CAPPED)
        form.addRow("Vias:", self.capped_check)

        self.cap_drill_edit = QLineEdit(f"{config.CAP_MAX_DRILL_MM:g}")
        self.cap_drill_edit.setEnabled(config.VIAS_CAPPED)
        self.capped_check.toggled.connect(self.cap_drill_edit.setEnabled)
        form.addRow("Capped up to drill [mm]:", self.cap_drill_edit)

        self.adaptive_check = QCheckBox(
            "adaptive cells (coarsen plane interiors; faster on large "
            "boards, corrected to ≲0.03 % of the uniform grid)")
        self.adaptive_check.setChecked(config.ADAPTIVE_CELLS)
        form.addRow("Grid:", self.adaptive_check)

        self.contact1_box = QComboBox()
        self.contact2_box = QComboBox()
        form.addRow(f"V+ ({e1_label}):", self.contact1_box)
        form.addRow(f"V− ({e2_label}):", self.contact2_box)

        self.model_box = QComboBox()
        for key in ("uniform", "equipotential"):
            self.model_box.addItem(MODEL_LABELS[key], key)
        default_index = 0 if config.CONTACT_MODEL == "uniform" else 1
        self.model_box.setCurrentIndex(default_index)
        form.addRow("Contact model:", self.model_box)

        self.current_edit = QLineEdit(f"{config.TEST_CURRENT_A:g}")
        form.addRow("Test current [A]:", self.current_edit)

        self.freq_edit = QLineEdit("")
        self.freq_edit.setPlaceholderText("0 = DC   (e.g. 142k, 1.5M)")
        form.addRow("Frequency [Hz]:", self.freq_edit)

        self.cell_edit = QLineEdit("")
        self.cell_edit.setPlaceholderText("auto")
        form.addRow("Cell size [µm]:", self.cell_edit)

        self.buildup_check = QCheckBox(
            f"{config.SOLDER_THICKNESS_UM:g} µm solder on mask openings"
            + (f" ({', '.join(buildup_layers)})" if buildup_layers
               else " (none found)"))
        self.buildup_check.setChecked(bool(buildup_layers)
                                      and config.INCLUDE_MASK_BUILDUP)
        self.buildup_check.setEnabled(bool(buildup_layers))
        form.addRow("Buildup:", self.buildup_check)

        self.extracu_edit = QLineEdit(f"{config.BUILDUP_EXTRA_CU_UM:g}")
        self.extracu_edit.setEnabled(bool(buildup_layers))
        form.addRow("Extra Cu in openings [µm]:", self.extracu_edit)

        first, last = config.OVERLAY_LAYERS[0], config.OVERLAY_LAYERS[-1]
        self.overlay_check = QCheckBox(
            f"experimental: push per-layer |J| heatmaps into the board as "
            f"reference images on {first}..{last} (replaces images there; "
            f"layers must be enabled in Board Setup)")
        self.overlay_check.setChecked(config.PUSH_OVERLAYS)
        form.addRow("Overlays:", self.overlay_check)

        tfirst, tlast = config.TRIM_LAYERS[0], config.TRIM_LAYERS[-1]
        self.trim_check = QCheckBox(
            f"experimental: mark copper below the threshold as polygons "
            f"on {tfirst}..{tlast} (replaces polygons there; a suggestion "
            f"only - removing copper shifts current elsewhere)")
        self.trim_check.setChecked(config.TRIM_ENABLED)
        form.addRow("Low-current copper:", self.trim_check)

        self.trim_mode_box = QComboBox()
        self.trim_mode_box.addItem("% of mean |J|", "pct")
        self.trim_mode_box.addItem("A/mm²", "abs")
        self.trim_mode_box.setCurrentIndex(1 if config.TRIM_MODE == "abs"
                                           else 0)
        self.trim_edit = QLineEdit(self._trim_default())
        for w in (self.trim_edit, self.trim_mode_box):
            w.setEnabled(config.TRIM_ENABLED)
            self.trim_check.toggled.connect(w.setEnabled)
        self.trim_mode_box.currentIndexChanged.connect(
            self._trim_mode_changed)
        trim_row = QWidget()
        trim_lay = QHBoxLayout(trim_row)
        trim_lay.setContentsMargins(0, 0, 0, 0)
        trim_lay.addWidget(self.trim_edit, 1)
        trim_lay.addWidget(self.trim_mode_box)
        form.addRow("Threshold:", trim_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._try_accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        note = QLabel("Multiple layers are coupled through the net's "
                      "via/through-pad barrels. f > 0 applies only the "
                      "foil-thickness skin effect (a lower bound on the "
                      "resistance rise) - not an AC impedance simulation: "
                      "proximity and inductance are not modeled.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        lay.addWidget(note)
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b02a2a;")
        self.error_label.setVisible(False)
        lay.addWidget(self.error_label)
        lay.addWidget(buttons)

        self._selection: Selection | None = None

        self._desired1, self._desired2 = contact1, contact2
        self.net_box.currentTextChanged.connect(self._refresh)
        self._refresh()

    def _trim_default(self, mode: str | None = None) -> str:
        mode = mode or self.trim_mode_box.currentData()
        return (f"{config.TRIM_THRESHOLD_A_MM2:g}" if mode == "abs"
                else f"{config.TRIM_THRESHOLD_PCT:g}")

    def _trim_mode_changed(self):
        # swap in the new unit's default, but never clobber a number the
        # user typed themselves
        mode = self.trim_mode_box.currentData()
        other = "pct" if mode == "abs" else "abs"
        if self.trim_edit.text().strip() in ("", self._trim_default(other)):
            self.trim_edit.setText(self._trim_default(mode))

    def _refresh(self):
        self.error_label.setVisible(False)
        net = self.net_box.currentText()
        layers = [n for n in self._layer_order
                  if n in self._candidates.get(net, [])]
        self.layer_list.clear()
        for name in layers:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.layer_list.addItem(item)
        for box, desired in ((self.contact1_box, self._desired1),
                             (self.contact2_box, self._desired2)):
            box.clear()
            box.addItem(AUTO_CONTACT)
            box.addItem(ALL_LAYERS)
            box.addItems(layers)
            if desired == "all":
                box.setCurrentText(ALL_LAYERS)
            elif desired in layers:
                box.setCurrentText(desired)

    def checked_layers(self) -> list[str]:
        out = []
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            if item.checkState() == Qt.Checked:
                out.append(item.text())
        return out

    def _build_selection(self) -> Selection:
        """Parse and validate every field; raises ValueError with a
        user-readable message instead of silently substituting defaults
        (a typo silently becoming 1 A / DC would mislabel the result)."""
        layers = self.checked_layers()
        if not layers:
            raise ValueError("Check at least one layer.")

        def number(edit: QLineEdit, name: str) -> float:
            text = edit.text().strip()
            try:
                return float(skin.normalize_decimal(text))
            except ValueError as exc:
                if "separator" in str(exc):
                    raise ValueError(f"{name}: {exc}")
                raise ValueError(f"{name}: '{text}' is not a number.")

        current = number(self.current_edit, "Test current")
        if current <= 0:
            raise ValueError("Test current must be > 0 A.")
        cell = None
        if self.cell_edit.text().strip():
            cell = number(self.cell_edit, "Cell size")
            if cell <= 0:
                raise ValueError("Cell size must be > 0 µm.")
        try:
            freq = skin.parse_frequency(self.freq_edit.text())
        except ValueError as exc:
            # as in number(): keep parse_frequency's own explanation for
            # the inputs it rejects deliberately, not just "unparseable"
            if any(k in str(exc) for k in ("separator", "negative")):
                raise ValueError(f"Frequency: {exc}")
            raise ValueError(
                f"Frequency: cannot parse '{self.freq_edit.text()}' "
                f"(examples: 0, 142k, 1.5M).")
        extra_cu = 0.0
        if self.extracu_edit.isEnabled():
            extra_cu = number(self.extracu_edit, "Extra Cu")
            if extra_cu < 0:
                raise ValueError("Extra Cu must be ≥ 0 µm.")
        trim_mode = self.trim_mode_box.currentData()
        trim_value = float(self._trim_default(trim_mode))
        if self.trim_check.isChecked():
            trim_value = number(self.trim_edit, "Trim threshold")
            if trim_mode == "pct" and not 0 < trim_value < 100:
                raise ValueError("Trim threshold must be between 0 and "
                                 "100 (% of the mean |J|).")
            if trim_mode == "abs" and trim_value <= 0:
                raise ValueError("Trim threshold must be > 0 A/mm².")
        cap_max_drill = config.CAP_MAX_DRILL_MM
        if self.capped_check.isChecked():
            cap_max_drill = number(self.cap_drill_edit, "Capped up to drill")
            if cap_max_drill <= 0:
                raise ValueError("Capped-up-to drill must be > 0 mm.")

        def contact(box: QComboBox) -> str:
            t = box.currentText()
            if t == AUTO_CONTACT:
                return "auto"
            return "all" if t == ALL_LAYERS else t

        return Selection(net=self.net_box.currentText(), layers=layers,
                         contact1=contact(self.contact1_box),
                         contact2=contact(self.contact2_box),
                         current_a=current, cell_um=cell,
                         freq_hz=freq,
                         contact_model=self.model_box.currentData(),
                         include_buildup=self.buildup_check.isChecked(),
                         extra_cu_um=extra_cu,
                         include_tracks=self.tracks_check.isChecked(),
                         vias_capped=self.capped_check.isChecked(),
                         cap_max_drill_mm=cap_max_drill,
                         adaptive=self.adaptive_check.isChecked(),
                         push_overlays=self.overlay_check.isChecked(),
                         trim_enabled=self.trim_check.isChecked(),
                         trim_mode=trim_mode, trim_value=trim_value)

    def _try_accept(self) -> None:
        try:
            self._selection = self._build_selection()
        except ValueError as e:
            self.error_label.setText(str(e))
            self.error_label.setVisible(True)
            return
        self.accept()


def ask(candidates: dict[str, list[str]], layer_order: list[str],
        default_net: str, e1_label: str, e2_label: str,
        contact1: str, contact2: str,
        buildup_layers: list[str] | None = None) -> Selection | None:
    """Show the dialog; returns None on cancel."""
    app = QApplication.instance() or QApplication([])
    dlg = _Dialog(candidates, layer_order, default_net, e1_label, e2_label,
                  contact1, contact2, buildup_layers or [])
    dlg.raise_()
    dlg.activateWindow()
    if dlg.exec() != QDialog.Accepted:
        return None
    return dlg._selection
