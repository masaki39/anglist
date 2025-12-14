import ctk
import qt


class ExportUI:
    """
    Builds the export panel (output dir, IDs, overwrite, trigger button).
    """

    def __init__(self, parentLayout, initial_preview="case001"):
        self.button = ctk.ctkCollapsibleButton()
        self.button.text = "エクスポート（学習データ）"
        parentLayout.addWidget(self.button)

        form = qt.QFormLayout(self.button)

        self.outputDirEdit = qt.QLineEdit()
        self.outputDirEdit.placeholderText = "出力先フォルダ（例: /path/to/dataset）"
        self.browseButton = qt.QPushButton("参照...")
        dirLayout = qt.QHBoxLayout()
        dirLayout.addWidget(self.outputDirEdit, 1)
        dirLayout.addWidget(self.browseButton)
        form.addRow("出力先:", dirLayout)

        self.caseIdEdit = qt.QLineEdit()
        self.caseIdEdit.placeholderText = "手入力する場合はこちら（例: case001）"
        form.addRow("ケースID:", self.caseIdEdit)

        self.prefixEdit = qt.QLineEdit("case")
        self.overwriteCheck = qt.QCheckBox("既存があれば上書きする")
        self.overwriteCheck.checked = False
        self.nextIdLabel = qt.QLabel(initial_preview)
        autoLayout = qt.QHBoxLayout()
        autoLayout.addWidget(qt.QLabel("プレフィックス:"))
        autoLayout.addWidget(self.prefixEdit)
        autoLayout.addWidget(qt.QLabel("次番号:"))
        autoLayout.addWidget(self.nextIdLabel)
        form.addRow("自動採番:", autoLayout)
        form.addRow("", self.overwriteCheck)

        self.exportButton = qt.QPushButton("エクスポート")
        self.exportButton.toolTip = ".npy/.nrrd とランドマークJSON(角度付き)を書き出します。"
        form.addRow(self.exportButton)

        self.exportStatusLabel = qt.QLabel("")
        self.exportStatusLabel.wordWrap = True
        form.addRow(self.exportStatusLabel)
