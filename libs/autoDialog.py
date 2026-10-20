import datetime
import logging
import os
import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, QTimer, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox as BB,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QListWidget,
)

from libs.llmAnnotator import (
    AnnotationCancelled,
    InvalidAnnotationError,
    LLMServiceError,
    TOTAL_TIMEOUT_SECONDS,
)
from libs.utils import newIcon

logger = logging.getLogger("PPOCRLabel")


class Worker(QThread):
    progressBarValue = pyqtSignal(int)
    imageStarted = pyqtSignal(int, str)
    statusValue = pyqtSignal(str)
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
            self.imageStarted.emit(index, img_path)
            self.statusValue.emit("reading")
            self.listValue.emit(img_path)
            try:
                image = cv2.imdecode(
                    np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if image is None:
                    raise InvalidAnnotationError("Could not decode image.")
                if self.model == "llm":
                    results = self.ocr.annotate(
                        image,
                        self.isInterruptionRequested,
                        status=self.statusValue.emit,
                    )
                else:
                    self.statusValue.emit("paddle")
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
                self.statusValue.emit("saving")
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
        self._image_started = None
        self._current_index = 0
        self._current_name = ""
        self._stage = "reading"
        self.setMinimumSize(750, 400)
        self.resize(950, 480)
        self.pb = QProgressBar(self)
        self.pb.setRange(0, len_bar)
        self.pb.setValue(0)
        self.pb.setFormat(self.text("Processed %v/%m images (%p%)", "已处理 %v/%m 张（%p%）"))
        self.statusLabel = QLabel(
            self.text("Preparing automatic labeling…", "准备自动标注…"), self
        )
        self.statusLabel.setTextFormat(Qt.PlainText)
        self.statusLabel.setWordWrap(True)
        self.listWidget = QListWidget(self)
        self.buttonBox = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)
        self.buttonBox.button(BB.Ok).setIcon(newIcon("done"))
        self.buttonBox.button(BB.Cancel).setIcon(newIcon("undo"))
        self.buttonBox.button(BB.Ok).setEnabled(False)
        self.buttonBox.accepted.connect(self.validate)
        self.buttonBox.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.pb)
        layout.addWidget(self.statusLabel)
        layout.addWidget(self.listWidget)
        layout.addWidget(self.buttonBox)
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle(self.text("Automatic labeling", "自动标注"))
        self.thread_1 = Worker(ocr, image_list, model=model, parent=self)
        self.thread_1.progressBarValue.connect(self.handleProgressBarSingal)
        self.thread_1.imageStarted.connect(self.handleImageStarted)
        self.thread_1.statusValue.connect(self.handleStatus)
        self.thread_1.listValue.connect(self.handleListWidgetSingal)
        # Acknowledge each result on the GUI thread before completing the batch.
        self.thread_1.resultReady.connect(
            self.handleResult, Qt.BlockingQueuedConnection
        )
        self.thread_1.end_signal.connect(self.handleEndsignalSignal)
        self.thread_1.finished.connect(self.workerFinished)
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(1000)
        self.status_timer.timeout.connect(self.updateStatus)
        self.time_start = time.monotonic()

    def text(self, english, chinese):
        return chinese if getattr(self.parent, "lang", "en") == "ch" else english

    def handleImageStarted(self, index, path):
        self._current_index = index
        self._current_name = os.path.basename(path)
        self.statusLabel.setToolTip(path)
        self._image_started = time.monotonic()
        self._stage = "reading"
        self.updateStatus()

    def handleStatus(self, stage):
        self._stage = stage
        if stage == "correcting":
            self.handleListWidgetSingal(
                self.text(
                    "Invalid annotation received; requesting one correction.",
                    "返回的标注无效，正在请求纠正一次。",
                )
            )
        self.updateStatus()

    def updateStatus(self):
        if self._image_started is None:
            return
        stages = {
            "reading": self.text("Reading image", "正在读取图片"),
            "paddle": self.text("Running Paddle recognition", "正在运行 Paddle 识别"),
            "requesting": self.text(
                "Requesting vision service; waiting for the full result",
                "正在请求视觉服务，等待完整结果",
            ),
            "correcting": self.text(
                "Waiting for the single correction result", "正在等待纠正结果（仅重试一次）"
            ),
            "validating": self.text("Validating annotations", "正在校验标注"),
            "saving": self.text("Saving annotations", "正在保存标注"),
        }
        stage = (
            self.text("Stopping; late replies will be ignored", "正在停止，迟到的结果不会写入")
            if self._cancelled
            else stages[self._stage]
        )
        elapsed = int(time.monotonic() - self._image_started)
        name = self.statusLabel.fontMetrics().elidedText(
            self._current_name, Qt.ElideMiddle, max(80, self.width() - 260)
        )
        text = self.text(
            "Image %s/%s: %s\n%s · Elapsed: %s s", "正在处理第 %s/%s 张：%s\n%s；本图已用时：%s 秒"
        ) % (self._current_index, self.len_bar, name, stage, elapsed)
        if self.thread_1.model == "llm":
            text += (
                self.text(
                    "\nTotal wait limit: %s s per image, including correction. You can cancel.",
                    "\n单图总等待上限 %s 秒（含纠正重试），可随时取消。",
                )
                % TOTAL_TIMEOUT_SECONDS
            )
        self.statusLabel.setText(text)

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
        self.status_timer.stop()
        self.setWindowTitle(
            self.text("Stopped", "已停止") if stopped else self.text("Completed", "已完成")
        )
        summary = self.text("Completed: %s; failed: %s", "已处理：%s；失败：%s") % (
            max(0, completed - self._save_errors),
            failed + self._save_errors,
        )
        self.handleListWidgetSingal(summary)
        self.statusLabel.setText(
            summary
            + self.text(" · Total: %s s", "；总用时：%s 秒")
            % int(time.monotonic() - self.time_start)
        )
        self.buttonBox.button(BB.Ok).setEnabled(True)
        self.buttonBox.button(BB.Cancel).setEnabled(False)

    def workerFinished(self):
        self.status_timer.stop()
        if self._cancelled:
            self.accept()

    def reject(self):
        self._cancelled = True
        if self.thread_1.isRunning():
            self.thread_1.requestInterruption()
            self.buttonBox.button(BB.Cancel).setEnabled(False)
            self.buttonBox.button(BB.Ok).setEnabled(False)
            self.setWindowTitle(self.text("Stopping…", "正在停止…"))
            self.updateStatus()
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
        self.time_start = time.monotonic()
        self.status_timer.start()
        self.thread_1.start()
        return 1 if self.exec_() else None

    def closeEvent(self, event):
        if self.thread_1.isRunning():
            event.ignore()
            self.reject()
        else:
            event.accept()
