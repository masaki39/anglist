"""
データセットナビゲーションパネル。
ディレクトリ選択・ケース前後送り・腰椎バリアント。
"""

from __future__ import annotations

import ctk
import qt
import slicer


class DatasetUI:
    def __init__(self, parent_layout):
        self.button = ctk.ctkCollapsibleButton()
        self.button.text = "Dataset"
        parent_layout.addWidget(self.button)

        form = qt.QFormLayout(self.button)
        form.setSpacing(4)

        # ---- データセットディレクトリ ----
        dir_row = qt.QHBoxLayout()
        self.dirEdit = qt.QLineEdit()
        self.dirEdit.setPlaceholderText("train/dataset/phase2 のパス")
        self.dirEdit.setReadOnly(True)
        self.browseButton = qt.QPushButton("Browse…")
        self.browseButton.setFixedWidth(80)
        dir_row.addWidget(self.dirEdit)
        dir_row.addWidget(self.browseButton)
        form.addRow("Dataset dir:", dir_row)

        # ---- ケースナビゲーション ----
        nav_row = qt.QHBoxLayout()
        self.prevButton = qt.QPushButton("◀")
        self.prevButton.setFixedWidth(36)
        self.caseCombo = qt.QComboBox()
        self.caseCombo.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        self.nextButton = qt.QPushButton("▶")
        self.nextButton.setFixedWidth(36)
        nav_row.addWidget(self.prevButton)
        nav_row.addWidget(self.caseCombo)
        nav_row.addWidget(self.nextButton)
        form.addRow("Case:", nav_row)

        # ---- 全体進捗 ----
        self.datasetProgressLabel = qt.QLabel("—")
        form.addRow("Dataset progress:", self.datasetProgressLabel)

        # ---- ケース内進捗 ----
        self.caseProgressLabel = qt.QLabel("— / —")
        form.addRow("Case progress:", self.caseProgressLabel)


    # ---- ヘルパー ----

    def current_case_id(self) -> str | None:
        text = self.caseCombo.currentText
        return text if text else None

    def set_cases(self, case_ids: list[str]) -> None:
        self.caseCombo.blockSignals(True)
        self.caseCombo.clear()
        for c in case_ids:
            self.caseCombo.addItem(c)
        self.caseCombo.blockSignals(False)

    def set_current_case(self, idx: int) -> None:
        self.caseCombo.blockSignals(True)
        self.caseCombo.setCurrentIndex(idx)
        self.caseCombo.blockSignals(False)

    def update_case_progress(self, placed: int, total: int) -> None:
        self.caseProgressLabel.setText(f"{placed} / {total}")
        color = "green" if placed == total else ("orange" if placed > 0 else "gray")
        self.caseProgressLabel.setStyleSheet(f"color: {color}; font-weight: bold;")

    def update_dataset_progress(self, annotated_cases: int, total_cases: int) -> None:
        self.datasetProgressLabel.setText(f"{annotated_cases} / {total_cases} cases fully annotated")
