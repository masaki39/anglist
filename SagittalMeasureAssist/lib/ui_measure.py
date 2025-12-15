import ctk
import qt
import slicer


class MeasureUI:
    """
    Builds the measurement panel (volume/markups selection, controls, results table).
    Intended to keep UI wiring separate from the main entrypoint.
    """

    def __init__(self, parentLayout):
        self.button = ctk.ctkCollapsibleButton()
        self.button.text = "計測"
        parentLayout.addWidget(self.button)

        form = qt.QFormLayout(self.button)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = True
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.showHidden = False
        self.volumeSelector.showChildNodeTypes = False
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.volumeSelector.setToolTip("対象の側面X線Volumeを選択します（計測のみなら未選択でも可）。")
        form.addRow("Volume:", self.volumeSelector)

        self.markupSelector = slicer.qMRMLNodeComboBox()
        self.markupSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.markupSelector.selectNodeUponCreation = True
        self.markupSelector.addEnabled = False
        self.markupSelector.removeEnabled = False
        self.markupSelector.noneEnabled = True
        self.markupSelector.showHidden = False
        self.markupSelector.showChildNodeTypes = False
        self.markupSelector.setMRMLScene(slicer.mrmlScene)
        self.markupSelector.setToolTip("5ランドマーク用Markups Fiducialを選択/作成します。")
        form.addRow("Markups:", self.markupSelector)

        self.createMarkupButton = qt.QPushButton("新規作成 / 1点追加")
        self.createMarkupButton.toolTip = "Markupsを作成し、1点だけ配置モードに入ります。次の点もこのボタンで追加してください。"
        form.addRow("Markupsを自動作成:", self.createMarkupButton)

        self.clearMarkupButton = qt.QPushButton("配置点をクリア")
        self.clearMarkupButton.toolTip = "選択中のMarkups内の既存ポイントをすべて削除します。"
        form.addRow("補助操作:", self.clearMarkupButton)

        self.flipXAxisCheckBox = qt.QCheckBox("左右反転を補正（x軸反転）")
        self.flipXAxisCheckBox.toolTip = "画像が左右逆（前方が左向き）で表示されている場合にチェック。計算時にx座標の符号を反転します。"
        form.addRow("左右反転補正:", self.flipXAxisCheckBox)

        self.instructionsLabel = qt.QLabel(
            "5つのランドマークを順番に配置してください:\n"
            "1) L1_ant (L1頭側終板 前縁)\n"
            "2) L1_post (L1頭側終板 後縁)\n"
            "3) S1_ant (S1頭側終板 前縁)\n"
            "4) S1_post (S1頭側終板 後縁)\n"
            "5) FH (両側大腿骨頭の中心)"
        )
        self.instructionsLabel.wordWrap = True
        form.addRow("ランドマーク手順:", self.instructionsLabel)

        self.updateButton = qt.QPushButton("計測を更新")
        self.updateButton.toolTip = "Compute PI, PT, SS, and LL from the current 5 landmarks."
        self.updateButton.enabled = True
        form.addRow(self.updateButton)

        self.resultsTable = qt.QTableWidget()
        self.resultsTable.setRowCount(4)
        self.resultsTable.setColumnCount(2)
        self.resultsTable.setHorizontalHeaderLabels(["Parameter", "Value (deg)"])
        self.resultsTable.verticalHeader().hide()
        self.resultsTable.horizontalHeader().setStretchLastSection(True)
        params = ["PI", "PT", "SS", "LL"]
        for i, name in enumerate(params):
            nameItem = qt.QTableWidgetItem(name)
            nameItem.setFlags(qt.Qt.ItemIsEnabled)
            self.resultsTable.setItem(i, 0, nameItem)
            valueItem = qt.QTableWidgetItem("--")
            valueItem.setFlags(qt.Qt.ItemIsEnabled)
            self.resultsTable.setItem(i, 1, valueItem)
        form.addRow("計測結果:", self.resultsTable)

        self.statusLabel = qt.QLabel("")
        self.statusLabel.wordWrap = True
        form.addRow(self.statusLabel)
