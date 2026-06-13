"""
アノテーションパネル。
96点ランドマークをグループ別タブで表示し、配置ボタンと状態アイコンを提供する。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import ctk
import qt

from dataset_io import LANDMARK_GROUPS

# 状態アイコン
_ICON_DONE = "✓"
_ICON_TODO = "○"


class AnnotateUI:
    def __init__(self, parent_layout):
        self.button = ctk.ctkCollapsibleButton()
        self.button.text = "Annotation"
        parent_layout.addWidget(self.button)

        vbox = qt.QVBoxLayout(self.button)
        vbox.setSpacing(4)

        # ---- 上部コントロール ----
        top_row = qt.QHBoxLayout()
        self.placeNextButton = qt.QPushButton("▶ Place Next Unset")
        self.placeNextButton.setToolTip("次の未設定ランドマークに配置モードを切り替える  [Space]")
        self.placeNextButton.setStyleSheet("QPushButton { font-weight: bold; padding: 4px 8px; }")
        self.progressLabel = qt.QLabel("0 / 96")
        self.progressLabel.setStyleSheet("font-weight: bold; font-size: 13px;")
        top_row.addWidget(self.placeNextButton)
        top_row.addStretch()
        top_row.addWidget(self.progressLabel)
        vbox.addLayout(top_row)

        # ---- バリアント設定 ----
        variant_row = qt.QHBoxLayout()
        variant_row.addWidget(qt.QLabel("腰椎数:"))
        self.lumbarCombo = qt.QComboBox()
        self.lumbarCombo.addItems(["L5 (通常)", "L4 (sacralization)", "L6 (lumbarization)"])
        self.lumbarCombo.setFixedWidth(160)
        variant_row.addWidget(self.lumbarCombo)
        variant_row.addSpacing(12)
        variant_row.addWidget(qt.QLabel("T12:"))
        self.hasT12Check = qt.QCheckBox("あり")
        self.hasT12Check.setChecked(True)
        variant_row.addWidget(self.hasT12Check)
        variant_row.addStretch()
        vbox.addLayout(variant_row)

        # ---- 現在配置中のランドマーク表示 ----
        self.activeLandmarkLabel = qt.QLabel("— 待機中 —")
        self.activeLandmarkLabel.setAlignment(qt.Qt.AlignCenter)
        self.activeLandmarkLabel.setStyleSheet(
            "background: #1a3a5c; color: #7dd4fc; "
            "padding: 3px 8px; border-radius: 3px; font-size: 12px;"
        )
        vbox.addWidget(self.activeLandmarkLabel)

        # ---- タブウィジェット ----
        self.tabWidget = qt.QTabWidget()
        vbox.addWidget(self.tabWidget)

        # {key: (place_btn, status_lbl, row_widget)}
        self.rows: Dict[str, Tuple] = {}

        self._build_tabs()

    def _build_tabs(self):
        for group in LANDMARK_GROUPS:
            tab = qt.QWidget()
            tab_vbox = qt.QVBoxLayout(tab)
            tab_vbox.setSpacing(0)
            tab_vbox.setContentsMargins(2, 2, 2, 2)

            scroll = qt.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(qt.QFrame.NoFrame)

            inner = qt.QWidget()
            grid = qt.QGridLayout(inner)
            grid.setSpacing(2)
            grid.setContentsMargins(4, 4, 4, 4)

            all_keys = list(group["keys"]) + list(group.get("optional_keys", []))
            for row_idx, key in enumerate(all_keys):
                # キー名ラベル
                key_lbl = qt.QLabel(key)
                key_lbl.setFixedWidth(90)
                key_lbl.setStyleSheet("font-family: monospace; font-size: 11px;")

                # Placeボタン
                place_btn = qt.QPushButton("Place")
                place_btn.setFixedWidth(56)
                place_btn.setFixedHeight(22)
                place_btn.setProperty("landmark_key", key)

                # オプションラベル
                optional_keys = group.get("optional_keys", [])
                if key in optional_keys:
                    opt_lbl = qt.QLabel("(optional)")
                    opt_lbl.setStyleSheet("color: gray; font-size: 10px;")
                else:
                    opt_lbl = qt.QLabel("")

                # 状態ラベル
                status_lbl = qt.QLabel(_ICON_TODO)
                status_lbl.setFixedWidth(24)
                status_lbl.setAlignment(qt.Qt.AlignCenter)
                status_lbl.setStyleSheet("color: gray; font-size: 14px;")

                grid.addWidget(key_lbl, row_idx, 0)
                grid.addWidget(place_btn, row_idx, 1)
                grid.addWidget(status_lbl, row_idx, 2)
                grid.addWidget(opt_lbl, row_idx, 3)

                self.rows[key] = (place_btn, status_lbl)

            grid.setColumnStretch(3, 1)

            scroll.setWidget(inner)
            tab_vbox.addWidget(scroll)

            r, g, b = group["color"]
            hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
            label = f"{group['tab_label']} (0/{len(all_keys)})"
            self.tabWidget.addTab(tab, label)

        # タブラベルの色は後から _update_tab_label() で更新

    # ---- 外部から呼ばれる更新メソッド ----

    def update_status(self, placed_keys: set, active_keys: List[str]) -> None:
        """✓/○ アイコンを更新し、進捗ラベルとタブラベルを更新する。"""
        placed = 0
        group_counts: Dict[str, Tuple[int, int]] = {}

        for group in LANDMARK_GROUPS:
            all_keys = list(group["keys"]) + list(group.get("optional_keys", []))
            g_placed = 0
            g_active = 0
            for key in all_keys:
                if key not in self.rows:
                    continue
                place_btn, status_lbl = self.rows[key]
                is_active = key in active_keys
                is_placed = key in placed_keys

                if is_placed:
                    status_lbl.setText(_ICON_DONE)
                    status_lbl.setStyleSheet("color: #22c55e; font-size: 14px;")  # green
                    place_btn.setStyleSheet("")
                    g_placed += 1
                    if is_active:
                        placed += 1
                else:
                    status_lbl.setText(_ICON_TODO)
                    if is_active:
                        status_lbl.setStyleSheet("color: #f59e0b; font-size: 14px;")  # amber
                    else:
                        status_lbl.setStyleSheet("color: gray; font-size: 14px;")

                # 非アクティブキー（バリアント除外）はグレーアウト
                place_btn.setEnabled(is_active)
                if not is_active:
                    place_btn.setStyleSheet("color: gray;")
                    status_lbl.setStyleSheet("color: #444; font-size: 14px;")

                if is_active:
                    g_active += 1

            group_counts[group["name"]] = (g_placed, g_active)

        # タブラベル更新
        for tab_idx, group in enumerate(LANDMARK_GROUPS):
            g_placed, g_total = group_counts[group["name"]]
            self.tabWidget.setTabText(tab_idx, f"{group['tab_label']} ({g_placed}/{g_total})")

        total_active = len(active_keys)
        self.progressLabel.setText(f"{placed} / {total_active}")
        color = "green" if placed == total_active else ("orange" if placed > 0 else "white")
        self.progressLabel.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 13px;")

    def set_active_landmark(self, key: Optional[str]) -> None:
        """現在配置中のランドマーク名を表示し、対応タブに切り替える。"""
        if key is None:
            self.activeLandmarkLabel.setText("— 待機中 —")
            self.activeLandmarkLabel.setStyleSheet(
                "background: #1a3a5c; color: #7dd4fc; "
                "padding: 3px 8px; border-radius: 3px; font-size: 12px;"
            )
            return

        self.activeLandmarkLabel.setText(f"配置中: {key}")
        self.activeLandmarkLabel.setStyleSheet(
            "background: #7c2d12; color: #fed7aa; "
            "padding: 3px 8px; border-radius: 3px; font-weight: bold; font-size: 12px;"
        )

        # 該当タブに切り替え
        for tab_idx, group in enumerate(LANDMARK_GROUPS):
            all_keys = list(group["keys"]) + list(group.get("optional_keys", []))
            if key in all_keys:
                self.tabWidget.setCurrentIndex(tab_idx)
                # 行をスクロールして見えるようにする
                if key in self.rows:
                    place_btn, _ = self.rows[key]
                    # スクロールエリア内の対応ウィジェットを表示
                    scroll_area = self.tabWidget.widget(tab_idx).layout().itemAt(0).widget()
                    scroll_area.ensureWidgetVisible(place_btn)
                break

    def scroll_to_key(self, key: str) -> None:
        if key in self.rows:
            place_btn, _ = self.rows[key]
            for tab_idx, group in enumerate(LANDMARK_GROUPS):
                all_keys = list(group["keys"]) + list(group.get("optional_keys", []))
                if key in all_keys:
                    scroll_area = self.tabWidget.widget(tab_idx).layout().itemAt(0).widget()
                    scroll_area.ensureWidgetVisible(place_btn)
                    break
