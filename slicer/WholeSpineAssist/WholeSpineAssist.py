"""
WholeSpineAnnotation — Phase 2 全脊椎 96点ランドマークアノテーションモジュール。

3D Slicer ScriptedLoadableModule として動作する。
"""

import os
import sys

import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

# lib/ と プロジェクトルートを sys.path に追加
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_MODULE_DIR))
for _p in [os.path.join(_MODULE_DIR, "lib"), _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class WholeSpineAssist(ScriptedLoadableModule):
    def __init__(self, parent=None):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Whole Spine Assist"
        self.parent.categories = ["Spine"]
        self.parent.dependencies = []
        self.parent.contributors = ["masaki39"]
        self.parent.helpText = (
            "Phase 2 全脊椎 96点ランドマークアノテーション支援ツール。\n"
            "train/dataset/phase2/ のデータセットを読み込み、\n"
            "EAC〜大腿骨遠位の全ランドマークを順番に配置し JSON に保存する。\n\n"
            "ショートカット:\n"
            "  Space → 次の未設定ランドマークを配置\n"
            "  , → 前のケース  . → 次のケース\n"
            "  S → 保存"
        )
        self.parent.acknowledgementText = ""


class WholeSpineAssistWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        from annotation_controller import AnnotationController
        from ui_annotate import AnnotateUI
        from ui_dataset import DatasetUI
        from ui_save import SaveUI

        self.dataset_ui = DatasetUI(self.layout)
        self.annotate_ui = AnnotateUI(self.layout)
        self.save_ui = SaveUI(self.layout)

        self.controller = AnnotationController(
            self.dataset_ui,
            self.annotate_ui,
            self.save_ui,
        )

        self.layout.addStretch()

    def enter(self):
        if hasattr(self, "controller"):
            self.controller.setup_shortcuts()

    def exit(self):
        if hasattr(self, "controller"):
            self.controller.teardown_shortcuts()

    def cleanup(self):
        if hasattr(self, "controller"):
            self.controller.cleanup()


class WholeSpineAssistLogic(ScriptedLoadableModuleLogic):
    pass


class WholeSpineAssistTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_import()

    def test_import(self):
        """モジュールが正常にインポートできることを確認する。"""
        import dataset_io
        import ui_annotate
        import ui_dataset
        import ui_save
        self.assertIsNotNone(dataset_io.LANDMARK_GROUPS)
        self.delayDisplay("Import test passed")
