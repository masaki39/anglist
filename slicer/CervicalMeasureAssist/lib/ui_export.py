import ctk
import qt


class ExportUI:
    """
    Builds the export panel (output dir, IDs, overwrite, trigger button).
    """

    def __init__(self, parentLayout):
        self.button = ctk.ctkCollapsibleButton()
        self.button.text = "Export"
        parentLayout.addWidget(self.button)

        form = qt.QFormLayout(self.button)

        self.outputDirEdit = qt.QLineEdit()
        self.outputDirEdit.placeholderText = "Output folder (e.g. /path/to/dataset)"
        self.browseButton = qt.QPushButton("Browse...")
        dirLayout = qt.QHBoxLayout()
        dirLayout.addWidget(self.outputDirEdit, 1)
        dirLayout.addWidget(self.browseButton)
        form.addRow("Output dir:", dirLayout)

        self.caseIdEdit = qt.QLineEdit()
        self.caseIdEdit.placeholderText = "Auto-filled or enter manually"
        self.caseIdEdit.setToolTip(
            "エクスポートで使用する Case ID。手動編集も可能。\n"
            "'ID source' ドロップダウンに従い、ボリューム読み込み時に自動入力されます。"
        )
        form.addRow("Case ID:", self.caseIdEdit)

        self.caseIdSourceCombo = qt.QComboBox()
        self.caseIdSourceCombo.addItem("DICOM Patient ID")
        self.caseIdSourceCombo.addItem("Filename")
        self.caseIdSourceCombo.addItem("Auto-numbering")
        self.caseIdSourceCombo.currentIndex = 1
        self.caseIdSourceCombo.setToolTip(
            "ボリュームロード時に Case ID を何から取得するかを選択します。\n"
            "DICOM Patient ID: タグ 0010,0020 を読み取ります。\n"
            "Filename: DICOM ファイルのベース名（拡張子なし）を使用します。\n"
            "Auto-numbering: Case ID を空にして、エクスポート時に prefix+連番を使用します。"
        )
        form.addRow("ID source:", self.caseIdSourceCombo)

        self.prefixEdit = qt.QLineEdit("case")
        self.prefixEdit.setToolTip(
            "Prefix for auto-generated IDs (e.g. 'case' → case001, case002...).\n"
            "The counter increments after each successful export."
        )
        self.overwriteCheck = qt.QCheckBox("Overwrite existing")
        self.overwriteCheck.checked = True
        self.nextIdLabel = qt.QLabel("case001")
        autoLayout = qt.QHBoxLayout()
        autoLayout.addWidget(qt.QLabel("Prefix:"))
        autoLayout.addWidget(self.prefixEdit)
        autoLayout.addWidget(qt.QLabel("Next ID:"))
        autoLayout.addWidget(self.nextIdLabel)
        form.addRow("Auto-numbering:", autoLayout)
        form.addRow("", self.overwriteCheck)

        self.exportTrainingDataCheck = qt.QCheckBox("Also export training data (.npy / .nrrd / .json)")
        self.exportTrainingDataCheck.setChecked(False)
        form.addRow("", self.exportTrainingDataCheck)

        self.autoExportCheck = qt.QCheckBox("Auto-export CSV on volume/case change  [s]")
        self.autoExportCheck.setChecked(False)
        self.autoExportCheck.setToolTip(
            "ボリューム切り替え（DICOMタブ）またはケース切り替え（データセットタブ）時に\n"
            "自動で CSV エクスポートを実行します。"
        )
        form.addRow("", self.autoExportCheck)

        self.csvButton = qt.QPushButton("Export CSV  [s]")
        self.csvButton.toolTip = (
            "Export the current case's angles to angles.csv in the output directory.\n"
            "Creates the file if it does not exist; adds a header row on first write.\n"
            "With 'Overwrite existing' checked, replaces the row for this case ID.\n"
            "With 'Also export training data' checked, also saves .npy / .nrrd / .json."
        )
        form.addRow(self.csvButton)

        self.exportStatusLabel = qt.QLabel("")
        self.exportStatusLabel.wordWrap = True
        form.addRow(self.exportStatusLabel)

    def set_next_id_preview(self, text):
        self.nextIdLabel.setText(text)
