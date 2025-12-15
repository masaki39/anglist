import logging
import math
import os

import qt
import slicer

from logic_export import ExportLogic, REQUIRED_LABELS_ORDERED
from logic_inference import OnnxInferenceLogic


class AssistController:
    """
    Glue code between MeasureUI / ExportUI and the underlying logic.
    Keeps the entrypoint thin and encapsulates event handling.
    """

    def __init__(self, measure_ui, export_ui, auto_ui, logic):
        self.measure_ui = measure_ui
        self.export_ui = export_ui
        self.auto_ui = auto_ui
        self.logic = logic
        self.counter = 1
        self.infer = OnnxInferenceLogic()
        self._connect_signals()
        self._update_counter_preview()

    # --- Signal wiring ---
    def _connect_signals(self):
        self.measure_ui.createMarkupButton.connect("clicked()", self.onCreateMarkup)
        self.measure_ui.clearMarkupButton.connect("clicked()", self.onClearMarkups)
        self.measure_ui.updateButton.connect("clicked()", self.onUpdateMeasurements)

        self.export_ui.exportButton.connect("clicked()", self.onExport)
        self.export_ui.browseButton.connect("clicked()", self.onBrowse)
        self.export_ui.prefixEdit.textChanged.connect(lambda *_: self._update_counter_preview())
        self.auto_ui.modelBrowseButton.connect("clicked()", self.onBrowseModel)
        self.auto_ui.runButton.connect("clicked()", self.onRunInference)

    # --- Handlers ---
    def onCreateMarkup(self):
        fiducialNode = self._ensureMarkupNodeExists()
        self.measure_ui.markupSelector.setCurrentNode(fiducialNode)
        slicer.modules.markups.logic().StartPlaceMode(0)  # place a single point then exit place mode
        self.measure_ui.statusLabel.text = "1点だけ配置モードです。点を置いたら、次の点も同じボタンで再度配置してください。"

    def onClearMarkups(self):
        markupNode = self.measure_ui.markupSelector.currentNode()
        if markupNode is None:
            self.measure_ui.statusLabel.text = "クリアできるMarkupsが選択されていません。"
            return
        markupNode.RemoveAllControlPoints()
        self.measure_ui.statusLabel.text = "既存のポイントをクリアしました。"

    def onUpdateMeasurements(self):
        markupNode = self.measure_ui.markupSelector.currentNode()
        if markupNode is None:
            self.measure_ui.statusLabel.text = "エラー: Markupsが選択されていません。"
            return
        self._assignLandmarkLabels(markupNode)
        if markupNode.GetNumberOfFiducials() != len(REQUIRED_LABELS_ORDERED):
            self.measure_ui.statusLabel.text = "エラー: マークアップ点が5個ではありません。指定の順番で5点を配置してください。"
            return

        points = {}
        coordsRAS = [0.0, 0.0, 0.0]
        for idx, label in enumerate(REQUIRED_LABELS_ORDERED):
            markupNode.GetNthFiducialPosition(idx, coordsRAS)
            x = coordsRAS[0]
            y = coordsRAS[1]
            if self.measure_ui.flipXAxisCheckBox.isChecked():
                x = -x
            points[label] = (x, y)

        try:
            angles = self.logic.compute_angles_from_points(points)
        except ValueError as exc:
            self.measure_ui.statusLabel.text = f"エラー: {exc}"
            return

        self._updateResultsTable(angles)
        self.measure_ui.statusLabel.text = "計測を更新しました。"

    def onBrowseModel(self):
        file_path = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(), "ONNXモデルを選択", "", "ONNX (*.onnx)"
        )
        if file_path:
            self.auto_ui.modelPathEdit.setText(file_path)

    def onRunInference(self):
        volumeNode = self.measure_ui.volumeSelector.currentNode()
        if volumeNode is None:
            self.auto_ui.statusLabel.setText("エラー: Volumeが選択されていません。")
            return

        markupNode = self.measure_ui.markupSelector.currentNode()
        if markupNode is None:
            markupNode = self._ensureMarkupNodeExists()
            self.measure_ui.markupSelector.setCurrentNode(markupNode)

        model_path = self.auto_ui.modelPathEdit.text.strip()
        if not model_path:
            self.auto_ui.statusLabel.setText("エラー: ONNXモデルパスを指定してください。")
            return

        target_h = int(self.auto_ui.heightSpin.value)
        target_w = int(self.auto_ui.widthSpin.value)

        try:
            # 再ロードは毎回（単純性優先）。必要ならパスが同じ場合はスキップも可。
            self.infer.load_model(model_path, (target_h, target_w))
            coords_ij = self.infer.predict_and_place(volumeNode, markupNode)
        except Exception as exc:
            logging.exception("Inference failed")
            self.auto_ui.statusLabel.setText(f"エラー: 推論に失敗しました ({exc})")
            return

        self.auto_ui.statusLabel.setText("推論完了: Markupsに自動配置しました。")
        # 計測も更新しておく
        self.onUpdateMeasurements()

    def onBrowse(self):
        directory = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(), "出力先フォルダを選択"
        )
        if directory:
            self.export_ui.outputDirEdit.text = directory

    def onExport(self):
        volumeNode = self.measure_ui.volumeSelector.currentNode()
        markupNode = self.measure_ui.markupSelector.currentNode()
        outputDir = self.export_ui.outputDirEdit.text.strip()
        manualCaseId = self.export_ui.caseIdEdit.text.strip()

        if volumeNode is None:
            self.export_ui.exportStatusLabel.text = "エラー: Volumeが選択されていません。"
            return
        if markupNode is None:
            self.export_ui.exportStatusLabel.text = "エラー: Markupsが選択されていません。"
            return
        if not outputDir:
            self.export_ui.exportStatusLabel.text = "エラー: 出力先フォルダを指定してください。"
            return

        caseId = manualCaseId if manualCaseId else self._find_next_case_id(outputDir)
        if caseId is None:
            self.export_ui.exportStatusLabel.text = "エラー: 利用可能なケースIDを見つけられませんでした。"
            return

        try:
            exporter = ExportLogic(flip_x_axis=self.measure_ui.flipXAxisCheckBox.isChecked())
            result = exporter.export_training_sample(
                volumeNode=volumeNode,
                markupNode=markupNode,
                outputDir=outputDir,
                caseId=caseId,
                overwrite=self.export_ui.overwriteCheck.isChecked(),
            )
        except ValueError as exc:
            self.export_ui.exportStatusLabel.text = f"エラー: {exc}"
            return
        except Exception:
            logging.exception("Export failed")
            self.export_ui.exportStatusLabel.text = "エラー: エクスポートに失敗しました。詳細はPython Consoleを確認してください。"
            return

        self.export_ui.exportStatusLabel.text = (
            "エクスポート完了: "
            f"{os.path.basename(result['npy'])}, "
            f"{os.path.basename(result['json'])}"
        )
        if not manualCaseId:
            self.counter += 1
            self._update_counter_preview()

    # --- Helpers ---
    def _ensureMarkupNodeExists(self):
        current = self.measure_ui.markupSelector.currentNode()
        if current and current.IsA("vtkMRMLMarkupsFiducialNode"):
            return current
        fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        displayNode = fiducialNode.GetDisplayNode()
        if displayNode:
            displayNode.SetSelectedColor(1.0, 0.4, 0.0)
            displayNode.SetGlyphScale(1.5)
        return fiducialNode

    def _assignLandmarkLabels(self, markupNode):
        count = min(markupNode.GetNumberOfFiducials(), len(REQUIRED_LABELS_ORDERED))
        for i in range(count):
            markupNode.SetNthControlPointLabel(i, REQUIRED_LABELS_ORDERED[i])

    def _updateResultsTable(self, anglesDict):
        params = ["PI", "PT", "SS", "LL"]
        for i, name in enumerate(params):
            value = anglesDict.get(name, float("nan"))
            if math.isnan(value):
                text = "--"
            else:
                text = f"{value:.1f}°"
            self.measure_ui.resultsTable.item(i, 1).setText(text)

    def _format_counter_preview(self):
        prefix = self.export_ui.prefixEdit.text.strip() or "case"
        return f"{prefix}{self.counter:03d}"

    def _update_counter_preview(self):
        self.export_ui.set_next_id_preview(self._format_counter_preview())

    def _find_next_case_id(self, outputDir):
        prefix = self.export_ui.prefixEdit.text.strip() or "case"
        for idx in range(self.counter, 10000):
            candidate = f"{prefix}{idx:03d}"
            npy = os.path.join(outputDir, f"{candidate}_image.npy")
            json_path = os.path.join(outputDir, f"{candidate}_landmarks.json")
            nrrd_path = os.path.join(outputDir, f"{candidate}_volume.nrrd")
            if self.export_ui.overwriteCheck.isChecked():
                return candidate
            if not (os.path.exists(npy) or os.path.exists(json_path) or os.path.exists(nrrd_path)):
                return candidate
        return None
