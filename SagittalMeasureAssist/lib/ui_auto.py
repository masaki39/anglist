import ctk
import qt


class AutoUI:
    """ONNX推論用の簡易UIセクション."""

    def __init__(self, parentLayout):
        self.button = ctk.ctkCollapsibleButton()
        self.button.text = "自動推論 (ONNX)"
        parentLayout.addWidget(self.button)

        form = qt.QFormLayout(self.button)

        self.modelPathEdit = qt.QLineEdit()
        self.modelBrowseButton = qt.QPushButton("参照...")
        modelLayout = qt.QHBoxLayout()
        modelLayout.addWidget(self.modelPathEdit, 1)
        modelLayout.addWidget(self.modelBrowseButton)
        form.addRow("ONNXモデル:", modelLayout)

        sizeLayout = qt.QHBoxLayout()
        self.heightSpin = qt.QSpinBox()
        self.heightSpin.setRange(64, 2048)
        self.heightSpin.setValue(512)
        self.widthSpin = qt.QSpinBox()
        self.widthSpin.setRange(64, 2048)
        self.widthSpin.setValue(512)
        sizeLayout.addWidget(qt.QLabel("H:"))
        sizeLayout.addWidget(self.heightSpin)
        sizeLayout.addWidget(qt.QLabel("W:"))
        sizeLayout.addWidget(self.widthSpin)
        form.addRow("入力サイズ:", sizeLayout)

        self.runButton = qt.QPushButton("推論してMarkupsに配置")
        form.addRow(self.runButton)

        self.statusLabel = qt.QLabel("")
        self.statusLabel.wordWrap = True
        form.addRow(self.statusLabel)
