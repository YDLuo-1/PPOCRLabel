import datetime
import logging
import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox as BB,
    QProgressBar,
    QVBoxLayout,
    QListWidget,
)

from libs.llmAnnotator import (
    AnnotationCancelled,
    InvalidAnnotationError,
    LLMServiceError,
)
from libs.utils import newIcon

logger = logging.getLogger("PPOCRLabel")


class Worker(QThread):
    progressBarValue = pyqtSignal(int)
    listValue = pyqtSignal(str)
    resultReady = pyqtSignal(str, object)
    end_signal = pyqtSignal(int, int, bool)

    def __init__(self, ocr, img_list, main_thread=None, model="paddle", parent=None):
        super().__init__(parent)
        self.ocr = ocr
        self.img_list = img_list or []
        self.model = model
        self.setStackSize(1024 * 1024)

    def run(self):
        completed, failed, stopped = 0, 0, False
        for index, img_path in enumerate(self.img_list, 1):
            if self.isInterruptionRequested():
                break
            fatal = False
            self.listValue.emit(img_path)
            try:
                image = cv2.imdecode(
                    np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if image is None:
                    raise InvalidAnnotationError("Could not decode image.")
                if self.model == "llm":
                    results = self.ocr.annotate(image, self.isInterruptionRequested)
                else:
                    results = []
                    if min(image.shape[:2]) > 32:
                        result = self.ocr.predict(image)[0]
                        for poly, text, score in zip(
                            result["rec_polys"],
                            result["rec_texts"],
                            result["rec_scores"],
                        ):
                            points = poly.tolist() if hasattr(poly, "tolist") else poly
                            results.append([points, (text, float(score))])
                if self.isInterruptionRequested():
                    break
                self.resultReady.emit(img_path, results)
                completed += 1
                self.listValue.emit("Text regions: %s" % len(results))
            except AnnotationCancelled:
                stopped = True
                break
            except Exception as error:
                failed += 1
                fatal = isinstance(error, LLMServiceError)
                self.listValue.emit("%s: %s" % (type(error).__name__, error))
                logger.warning("Automatic labeling failed for %s: %s", img_path, error)
            self.progressBarValue.emit(index)
            if fatal:
                stopped = True
                break
        self.end_signal.emit(
            completed, failed, stopped or self.isInterruptionRequested()
        )


class AutoDialog(QDialog):
    def __init__(
        self,
        text="Enter object label",
        parent=None,
        ocr=None,
        image_list=None,
        len_bar=0,
        model="paddle",
    ):
        super().__init__(parent)
        self.parent = parent
        self.len_bar = len_bar
        self._cancelled = False
        self._save_errors = 0
        self.setMinimumSize(750, 400)
        self.resize(950, 480)
        self.pb = QProgressBar(self)
        self.pb.setRange(0, len_bar)
        self.pb.setValue(0)
        self.listWidget = QListWidget(self)
        self.buttonBox = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)
        self.buttonBox.button(BB.Ok).setIcon(newIcon("done"))
        self.buttonBox.button(BB.Cancel).setIcon(newIcon("undo"))
        self.buttonBox.button(BB.Ok).setEnabled(False)
        self.buttonBox.accepted.connect(self.validate)
        self.buttonBox.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.pb)
        layout.addWidget(self.listWidget)
        layout.addWidget(self.buttonBox)
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle(self.text("Automatic labeling", "自动标注"))
        self.thread_1 = Worker(ocr, image_list, model=model, parent=self)
        self.thread_1.progressBarValue.connect(self.handleProgressBarSingal)
        self.thread_1.listValue.connect(self.handleListWidgetSingal)
        # Acknowledge each result on the GUI thread before completing the batch.
        self.thread_1.resultReady.connect(
            self.handleResult, Qt.BlockingQueuedConnection
        )
        self.thread_1.end_signal.connect(self.handleEndsignalSignal)
        self.thread_1.finished.connect(self.workerFinished)
        self.time_start = time.monotonic()

    def text(self, english, chinese):
        return chinese if getattr(self.parent, "lang", "en") == "ch" else english

    def handleProgressBarSingal(self, index):
        self.pb.setValue(index)
        if index:
            remaining = (
                (time.monotonic() - self.time_start) / index * (self.len_bar - index)
            )
            left = str(datetime.timedelta(seconds=int(remaining)))
            self.setWindowTitle(self.text("Time left: ", "预计剩余：") + left)

    def handleListWidgetSingal(self, value):
        self.listWidget.addItem(value)
        self.listWidget.scrollToBottom()

    def handleResult(self, path, results):
        if self._cancelled:
            return
        try:
            self.parent.applyAutoLabelResult(path, results)
        except Exception as error:
            self._save_errors += 1
            self.handleListWidgetSingal(
                self.text("Could not save result: ", "无法保存结果：") + str(error)
            )
            self.thread_1.requestInterruption()

    def handleEndsignalSignal(self, completed, failed, stopped):
        self.setWindowTitle(
            self.text("Stopped", "已停止") if stopped else self.text("Completed", "已完成")
        )
        self.handleListWidgetSingal(
            self.text("Completed: %s; failed: %s", "已处理：%s；失败：%s")
            % (max(0, completed - self._save_errors), failed + self._save_errors)
        )
        self.buttonBox.button(BB.Ok).setEnabled(True)
        self.buttonBox.button(BB.Cancel).setEnabled(False)

    def workerFinished(self):
        if self._cancelled:
            self.accept()

    def reject(self):
        self._cancelled = True
        if self.thread_1.isRunning():
            self.thread_1.requestInterruption()
            self.buttonBox.button(BB.Cancel).setEnabled(False)
            self.buttonBox.button(BB.Ok).setEnabled(False)
            self.setWindowTitle(self.text("Stopping…", "正在停止…"))
        else:
            self.accept()

    def validate(self):
        if self.thread_1.isRunning():
            self.reject()
        else:
            self.accept()

    def popUp(self):
        if not self.thread_1.img_list:
            return 1
        self.thread_1.start()
        return 1 if self.exec_() else None

    def closeEvent(self, event):
        if self.thread_1.isRunning():
            event.ignore()
            self.reject()
        else:
            event.accept()
