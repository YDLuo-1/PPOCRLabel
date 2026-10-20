import ast
import copy
import json
import os
import platform
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5.QtCore import QThread, QTimer
from PyQt5.QtWidgets import QApplication, QDialog, QLineEdit, QWidget

from libs.autoDialog import AutoDialog
from libs.autoLabelSettings import default_config
from libs.autoLabelSettingsDialog import AutoLabelSettingsDialog
from libs.llmAnnotator import LLMAnnotator
from test_llm_annotator import completion, vision_server


class ClosingAutoDialog(AutoDialog):
    def popUp(self):
        self.thread_1.finished.connect(self.validate)
        return super().popUp()


def main_methods():
    source = Path(__file__).parents[1] / "PPOCRLabel.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    main = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MainWindow"
    )
    names = {
        "autoRecognition",
        "applyAutoLabelResult",
        "saveLabels",
        "validFilestate",
        "getImglabelidx",
    }
    functions = [
        n for n in main.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    namespace = {
        "os": os,
        "platform": platform,
        "QMessageBox": Mock(),
        "AutoDialog": ClosingAutoDialog,
        "LLMAnnotator": LLMAnnotator,
    }
    exec(
        compile(ast.Module(functions, type_ignores=[]), str(source), "exec"), namespace
    )
    return namespace, names


METHODS, METHOD_NAMES = main_methods()


class LabelWindow(QWidget):
    def __init__(self, directory):
        super().__init__()
        self.lang = "en"
        self.kie_mode = False
        self.model_config = default_config()
        self._model_config_error = None
        self.stringBundle = Mock()
        self.stringBundle.getString.return_value = "Automatic labeling"
        self.errorMessage = Mock()
        self.ensurePaddleModel = lambda: True
        self.saveCacheLabel = Mock()
        self.init_key_list = Mock()
        self.loadFile = Mock()
        self.mImgList = []
        for index in range(3):
            path = str(Path(directory) / ("%s.png" % index))
            cv2.imencode(".png", np.full((80, 120, 3), index, np.uint8))[1].tofile(path)
            self.mImgList.append(path)
        self.currIndex = 0
        self.auto_recognition_num = 3
        self.filePath = self.mImgList[0]
        self.result_dic = ["existing UI state"]
        self.PPlabel = {}
        for path in self.mImgList:
            self.PPlabel[self.getImglabelidx(path)] = [
                {"transcription": "original", "points": []}
            ]
        self.Cachelabel = copy.deepcopy(self.PPlabel)
        self.fileStatedict = {self.getImglabelidx(self.mImgList[0]): 1}
        self.ocr = Mock()
        self.ocr.predict.return_value = [
            {
                "rec_polys": [[[2, 3], [110, 3], [110, 25], [2, 25]]],
                "rec_texts": ["A0514+"],
                "rec_scores": [0.99],
            }
        ]
        self.llm_api_key = "test-session-key"
        self.llm_service = {}


for _name in METHOD_NAMES:
    setattr(LabelWindow, _name, METHODS[_name])


class AutoLabelingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_both_backends_skip_confirmed_and_save_on_gui_thread(self):
        for backend in ("paddle", "llm"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as root:
                window = LabelWindow(root)
                original = copy.deepcopy(window.PPlabel)
                old_state = window.fileStatedict.copy()
                ui_state = window.result_dic
                saved_threads = []
                save = window.applyAutoLabelResult

                def record(path, result):
                    saved_threads.append(QThread.currentThread())
                    return save(path, result)

                window.applyAutoLabelResult = record
                with vision_server() as (url, server):
                    window.model_config["backend"] = backend
                    window.llm_service = {"base_url": url, "model": "test-vision"}
                    window.autoRecognition()
                    self.assertEqual(len(server.received), 2 if backend == "llm" else 0)
                first = window.getImglabelidx(window.mImgList[0])
                self.assertEqual(window.PPlabel[first], original[first])
                self.assertEqual(window.Cachelabel[first], original[first])
                self.assertEqual(window.fileStatedict, old_state)
                self.assertIs(window.result_dic, ui_state)
                self.assertTrue(
                    all(thread is self.app.thread() for thread in saved_threads)
                )
                self.assertEqual(len(saved_threads), 2)
                for path in window.mImgList[1:]:
                    self.assertEqual(
                        window.PPlabel[window.getImglabelidx(path)][0]["transcription"],
                        "A0514+",
                    )
                window.close()

    def test_current_range_only(self):
        with tempfile.TemporaryDirectory() as root:
            window = LabelWindow(root)
            window.currIndex, window.auto_recognition_num = 2, 1
            original = copy.deepcopy(window.PPlabel)
            window.autoRecognition()
            self.assertEqual(window.ocr.predict.call_count, 1)
            key = window.getImglabelidx(window.mImgList[1])
            self.assertEqual(window.PPlabel[key], original[key])
            window.close()

    def test_invalid_image_preserves_labels_and_empty_result_clears_pending_only(self):
        responses = iter(
            [
                {"choices": [{"message": {"content": "invalid JSON"}}]},
                {"choices": [{"message": {"content": "invalid JSON"}}]},
                completion({"annotations": []}),
            ]
        )
        with tempfile.TemporaryDirectory() as root, vision_server(
            lambda body: (200, next(responses))
        ) as (url, _):
            window = LabelWindow(root)
            before = copy.deepcopy(window.PPlabel)
            window.model_config["backend"] = "llm"
            window.llm_service = {"base_url": url, "model": "test-vision"}
            window.autoRecognition()
            first, second = (window.getImglabelidx(p) for p in window.mImgList[1:])
            self.assertEqual(window.PPlabel[first], before[first])
            self.assertEqual(window.Cachelabel[first], before[first])
            self.assertEqual(window.PPlabel[second], [])
            self.assertEqual(window.Cachelabel[second], [])
            self.assertNotIn(second, window.fileStatedict)
            window.close()

    def test_authentication_error_stops_batch(self):
        for status, error in (
            (401, {}),
            (400, {"error": {"message": "Model Not Exist", "param": "model"}}),
        ):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as root:
                with vision_server(lambda body: (status, error)) as (url, server):
                    window = LabelWindow(root)
                    before = copy.deepcopy(window.PPlabel)
                    window.model_config["backend"] = "llm"
                    window.llm_service = {"base_url": url, "model": "test-vision"}
                    window.autoRecognition()
                    self.assertEqual(len(server.received), 1)
                    self.assertEqual(window.PPlabel, before)
                    if status == 400:
                        log = "\n".join(
                            window.autoDialog.listWidget.item(i).text()
                            for i in range(window.autoDialog.listWidget.count())
                        )
                        self.assertIn("Model Not Exist", log)
                        self.assertIn("param: model", log)
                    window.close()

    def test_cancellation_ignores_late_response_and_next_image(self):
        entered, release = threading.Event(), threading.Event()

        def delayed(body):
            entered.set()
            release.wait(3)
            return 200, completion()

        with tempfile.TemporaryDirectory() as root, vision_server(delayed) as (
            url,
            server,
        ):
            window = LabelWindow(root)
            before = copy.deepcopy(window.PPlabel)
            dialog = AutoDialog(
                parent=window,
                ocr=LLMAnnotator(url, "vision"),
                image_list=window.mImgList[1:],
                len_bar=2,
                model="llm",
            )
            timer = QTimer()
            timer.setInterval(10)
            timer.timeout.connect(lambda: dialog.reject() if entered.is_set() else None)
            timer.start()
            started = time.monotonic()
            try:
                dialog.popUp()
                self.assertLess(time.monotonic() - started, 1.5)
                self.assertEqual(len(server.received), 1)
                self.assertFalse(dialog.thread_1.isRunning())
                self.assertEqual(window.PPlabel, before)
            finally:
                timer.stop()
                release.set()
            self.app.processEvents()
            self.assertEqual(window.PPlabel, before)
            window.close()

    def test_commit_rechecks_confirmation_and_keeps_pending_ui_results(self):
        with tempfile.TemporaryDirectory() as root:
            window = LabelWindow(root)
            before = copy.deepcopy(window.PPlabel)
            self.assertFalse(window.applyAutoLabelResult(window.mImgList[0], []))
            self.assertEqual(window.PPlabel, before)
            state = window.result_dic
            window.kie_mode = True
            window.applyAutoLabelResult(
                window.mImgList[1],
                [[[[1, 1], [20, 1], [20, 10], [1, 10]], ("###", None)]],
            )
            self.assertIs(window.result_dic, state)
            key = window.getImglabelidx(window.mImgList[1])
            self.assertEqual(window.PPlabel[key][0]["key_cls"], "None")
            window.close()

    def test_settings_dialog_retains_paddle_selection_and_never_exports_key(self):
        config = {**default_config(), "preset": "PP-OCRv6_medium"}
        dialog = AutoLabelSettingsDialog(
            config,
            service={"base_url": "http://localhost/v1", "model": "vision"},
            api_key="session-key",
        )
        dialog.source.setCurrentIndex(dialog.source.findData("llm"))
        dialog.instructions.setPlainText("Read serial numbers only.")
        self.assertEqual(dialog.key_edit.echoMode(), QLineEdit.Password)
        saved = dialog.config()
        self.assertEqual(saved["backend"], "llm")
        self.assertEqual(saved["preset"], "PP-OCRv6_medium")
        self.assertEqual(saved["instructions"], "Read serial numbers only.")
        self.assertNotIn("session-key", json.dumps(saved))
        self.assertEqual(dialog.service()["model"], "vision")
        dialog.close()


if __name__ == "__main__":
    unittest.main()
