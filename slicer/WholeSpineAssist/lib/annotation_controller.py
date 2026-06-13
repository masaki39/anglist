"""
WholeSpineAnnotation のメインコントローラー。
UI シグナルとデータ I/O を接続する。
"""

from __future__ import annotations

import logging
import os

import qt
import slicer

import dataset_io
from dataset_io import (
    LANDMARK_GROUPS,
    active_keys_for_variant,
    count_annotated,
    create_markup_node,
    create_volume_node,
    discover_cases,
    get_placed_keys,
    load_json,
    load_landmarks_into_node,
    save_json,
    save_landmarks_from_node,
)
from train.landmark_scheme import ALL_LANDMARK_KEYS

log = logging.getLogger(__name__)

_DEFAULT_DATASET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "train", "dataset", "phase2")
)


class AnnotationController:
    def __init__(self, dataset_ui, annotate_ui, save_ui):
        self.dataset_ui = dataset_ui
        self.annotate_ui = annotate_ui
        self.save_ui = save_ui

        self._dataset_dir: str | None = None
        self._case_ids: list[str] = []
        self._current_idx: int = -1

        self._volume_node = None
        self._markup_node = None
        self._meta: dict = {}

        self._pending_label: str | None = None
        self._markup_observer_tags: list = []
        self._shortcuts: list = []
        self._modified: bool = False
        self._loading_variant: bool = False

        self._connect_signals()
        if os.path.isdir(_DEFAULT_DATASET_DIR):
            self._load_dataset(_DEFAULT_DATASET_DIR)

    # ------------------------------------------------------------------ #
    #  シグナル接続
    # ------------------------------------------------------------------ #

    def _connect_signals(self):
        ui = self.dataset_ui
        ui.browseButton.connect("clicked()", self._on_browse)
        ui.prevButton.connect("clicked()", self._on_prev)
        ui.nextButton.connect("clicked()", self._on_next)
        ui.caseCombo.connect("currentIndexChanged(int)", self._on_case_changed)

        self.annotate_ui.placeNextButton.connect("clicked()", self._on_place_next)
        self.annotate_ui.lumbarCombo.connect("currentIndexChanged(int)", self._on_lumbar_changed)
        self.annotate_ui.hasT12Check.connect("toggled(bool)", self._on_t12_changed)
        self.save_ui.saveButton.connect("clicked()", self._on_save)

        # 各 Place ボタン
        for key, (place_btn, _) in self.annotate_ui.rows.items():
            place_btn.connect("clicked()", lambda k=key: self._on_place(k))

    def setup_shortcuts(self):
        mw = slicer.util.mainWindow()
        bindings = [
            (qt.QKeySequence(qt.Qt.Key_Space), self._on_place_next),
            (qt.QKeySequence("Ctrl+S"),         self._on_save),
            (qt.QKeySequence(","),             self._on_prev),
            (qt.QKeySequence("."),             self._on_next),
        ]
        for seq, slot in bindings:
            sc = qt.QShortcut(seq, mw)
            sc.setContext(qt.Qt.ApplicationShortcut)
            sc.connect("activated()", slot)
            self._shortcuts.append(sc)

    def teardown_shortcuts(self):
        for sc in self._shortcuts:
            sc.setEnabled(False)
            sc.deleteLater()
        self._shortcuts = []

    # ------------------------------------------------------------------ #
    #  データセット読み込み
    # ------------------------------------------------------------------ #

    def _on_browse(self):
        chosen = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(),
            "Phase2 データセットディレクトリを選択",
            self._dataset_dir or "",
        )
        if not chosen:
            return
        self._load_dataset(chosen)

    def _load_dataset(self, directory: str):
        self._dataset_dir = directory
        self.dataset_ui.dirEdit.setText(directory)

        self._case_ids = discover_cases(directory)
        if not self._case_ids:
            slicer.util.warningDisplay(f"npy ファイルが見つかりません: {directory}")
            return

        self.dataset_ui.set_cases(self._case_ids)
        self._update_dataset_progress()
        self._load_case(0)

    # ------------------------------------------------------------------ #
    #  ケースナビゲーション
    # ------------------------------------------------------------------ #

    def _on_prev(self):
        if self._current_idx > 0:
            self._navigate_to(self._current_idx - 1)

    def _on_next(self):
        if self._current_idx < len(self._case_ids) - 1:
            self._navigate_to(self._current_idx + 1)

    def _on_case_changed(self, idx: int):
        if idx != self._current_idx and idx >= 0:
            self._navigate_to(idx)

    def _navigate_to(self, idx: int):
        if self.save_ui.auto_save and self._modified:
            self._save_current()
        self._load_case(idx)

    def _load_case(self, idx: int):
        if not self._case_ids or idx < 0 or idx >= len(self._case_ids):
            return

        self._detach_observers()

        if self._markup_node:
            slicer.mrmlScene.RemoveNode(self._markup_node)
            self._markup_node = None
        if self._volume_node:
            slicer.mrmlScene.RemoveNode(self._volume_node)
            self._volume_node = None

        self._current_idx = idx
        case_id = self._case_ids[idx]

        self.dataset_ui.set_current_case(idx)
        self._meta = load_json(self._dataset_dir, case_id)

        # ボリュームロード
        self._volume_node = create_volume_node(case_id, self._dataset_dir, self._meta)

        # Sliceビューに表示
        slicer.util.setSliceViewerLayers(background=self._volume_node)
        lm = slicer.app.layoutManager()
        for name in lm.sliceViewNames():
            lm.sliceWidget(name).sliceLogic().FitSliceToAll()

        # マークアップノード
        self._markup_node = create_markup_node(case_id)
        load_landmarks_into_node(self._meta, self._markup_node, self._volume_node)
        self._attach_observers()

        self._pending_label = None
        self.annotate_ui.set_active_landmark(None)
        self.annotate_ui.tabWidget.setCurrentIndex(0)
        self._modified = False
        self._load_variant_into_ui(self._meta.get("lumbar_variant", "normal"))
        self._refresh_ui()
        self.save_ui.set_status(f"Loaded {case_id}")

    # ------------------------------------------------------------------ #
    #  ランドマーク配置
    # ------------------------------------------------------------------ #

    def _on_lumbar_changed(self, index: int):
        if not self._loading_variant:
            self._refresh_ui()

    def _on_t12_changed(self, checked: bool):
        if not self._loading_variant:
            self._refresh_ui()

    def _get_variant(self) -> str:
        """UIの選択からバリアント文字列を返す。"""
        lumbar = self.annotate_ui.lumbarCombo.currentIndex  # 0=L5, 1=L4, 2=L6
        has_t12 = self.annotate_ui.hasT12Check.isChecked()
        if lumbar == 2:
            return "lumbarization"
        if lumbar == 1:
            return "sacralization"
        if not has_t12:
            return "t12_missing"
        return "normal"

    def _load_variant_into_ui(self, variant: str):
        """保存済みバリアントをUIコントロールに反映する。"""
        self._loading_variant = True
        ui = self.annotate_ui
        ui.hasT12Check.setChecked(True)
        if variant == "lumbarization":
            ui.lumbarCombo.setCurrentIndex(2)
        elif variant == "sacralization":
            ui.lumbarCombo.setCurrentIndex(1)
        elif variant == "t12_missing":
            ui.lumbarCombo.setCurrentIndex(0)
            ui.hasT12Check.setChecked(False)
        else:
            ui.lumbarCombo.setCurrentIndex(0)
        self._loading_variant = False

    def _on_place_next(self):
        """現在タブ優先で次の未設定ランドマークに配置モードを切り替える。"""
        if self._markup_node is None:
            self.save_ui.set_status("ケースを読み込んでください", ok=False)
            return
        try:
            placed = get_placed_keys(self._markup_node)
            active = active_keys_for_variant(self._get_variant())
            active_set = set(active)

            current_tab = self.annotate_ui.tabWidget.currentIndex
            if 0 <= current_tab < len(LANDMARK_GROUPS):
                group = LANDMARK_GROUPS[current_tab]
                tab_keys = list(group["keys"]) + list(group.get("optional_keys", []))
                for key in tab_keys:
                    if key in active_set and key not in placed:
                        self._activate_placement(key)
                        return

            for key in active:
                if key not in placed:
                    self._activate_placement(key)
                    return

            self.annotate_ui.set_active_landmark(None)
            self.save_ui.set_status("全ランドマーク設定済みです", ok=True)
        except Exception as e:
            log.exception("_on_place_next failed")
            self.save_ui.set_status(f"Error: {e}", ok=False)

    def _on_place(self, key: str):
        """特定ランドマークの配置モードを起動する。"""
        if self._markup_node is None:
            return
        self._activate_placement(key)

    def _activate_placement(self, key: str):
        self._pending_label = key
        self.annotate_ui.set_active_landmark(key)

        selNode = slicer.app.applicationLogic().GetSelectionNode()
        selNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selNode.SetActivePlaceNodeID(self._markup_node.GetID())
        slicer.modules.markups.logic().StartPlaceMode(0)

    def _on_point_added(self, caller, event):
        """新しいコントロールポイントが追加されたときにラベルを付ける。"""
        if self._pending_label is None:
            return
        node = caller
        n = node.GetNumberOfControlPoints()
        if n == 0:
            return

        new_idx = n - 1
        key = self._pending_label
        self._pending_label = None

        # 既存の同名ポイントを削除（上書き再配置）
        for i in range(n - 1):
            if node.GetNthControlPointLabel(i) == key:
                node.RemoveNthControlPoint(i)
                new_idx = node.GetNumberOfControlPoints() - 1
                break

        node.SetNthControlPointLabel(new_idx, key)
        self._modified = True
        self._refresh_ui()
        self.annotate_ui.set_active_landmark(None)

    def _on_point_modified(self, caller, event):
        self._modified = True

    def _on_point_removed(self, caller, event):
        self._modified = True
        self._refresh_ui()

    # ------------------------------------------------------------------ #
    #  保存
    # ------------------------------------------------------------------ #

    def _on_save(self):
        self._save_current()

    def _save_current(self):
        if self._markup_node is None or self._volume_node is None:
            return
        if not self._case_ids or self._current_idx < 0:
            return

        case_id = self._case_ids[self._current_idx]
        variant = self._get_variant()
        try:
            updated = save_landmarks_from_node(
                self._meta, self._markup_node, self._volume_node, variant
            )
            save_json(updated, self._dataset_dir, case_id)
            self._meta = updated
            self._modified = False
            self.save_ui.set_status(f"Saved {case_id}", ok=True)
            self._update_dataset_progress()
        except Exception as e:
            log.exception("Save failed")
            self.save_ui.set_status(f"Error: {e}", ok=False)

    # ------------------------------------------------------------------ #
    #  UI 更新
    # ------------------------------------------------------------------ #

    def _refresh_ui(self):
        if self._markup_node is None:
            return
        placed = get_placed_keys(self._markup_node)
        active = active_keys_for_variant(self._get_variant())
        self.annotate_ui.update_status(placed, active)
        self.dataset_ui.update_case_progress(len(placed & set(active)), len(active))

    def _update_dataset_progress(self):
        if not self._dataset_dir or not self._case_ids:
            return
        fully_annotated = 0
        for cid in self._case_ids:
            placed, total = count_annotated(self._dataset_dir, cid)
            if placed > 0:  # 1点以上で「作業中」としてカウント; 全完了は placed==total
                fully_annotated += (1 if placed == total else 0)
        self.dataset_ui.update_dataset_progress(fully_annotated, len(self._case_ids))

    # ------------------------------------------------------------------ #
    #  オブザーバー管理
    # ------------------------------------------------------------------ #

    def _attach_observers(self):
        if self._markup_node is None:
            return
        node = self._markup_node
        self._markup_observer_tags = [
            node.AddObserver(slicer.vtkMRMLMarkupsNode.PointAddedEvent, self._on_point_added),
            node.AddObserver(slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._on_point_modified),
            node.AddObserver(slicer.vtkMRMLMarkupsNode.PointRemovedEvent, self._on_point_removed),
        ]

    def _detach_observers(self):
        if self._markup_node is None:
            return
        for tag in self._markup_observer_tags:
            self._markup_node.RemoveObserver(tag)
        self._markup_observer_tags = []

    # ------------------------------------------------------------------ #
    #  クリーンアップ
    # ------------------------------------------------------------------ #

    def cleanup(self):
        self._detach_observers()
        self.teardown_shortcuts()
