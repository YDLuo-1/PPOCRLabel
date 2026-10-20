from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from libs.autoLabelSettings import LANGUAGES, MODEL_PRESETS


class AutoLabelSettingsDialog(QDialog):
    def __init__(
        self,
        config,
        parent=None,
        language="ch",
        cli_overrides=False,
        service=None,
        api_key="",
    ):
        super().__init__(parent)
        self.language = language
        self.previous_config = config.copy()
        service = service or {}
        self.setWindowTitle(self.text("Automatic labeling settings", "自动标注设置"))
        self.setMinimumWidth(620)
        self.source = QComboBox()
        self.source.setObjectName("modelSource")
        self.source.addItem(
            self.text("Official Paddle model", "官方 Paddle 模型"), "official"
        )
        self.source.addItem(self.text("Local Paddle model", "本地 Paddle 模型"), "local")
        self.source.addItem(self.text("LLM vision model", "LLM 视觉模型"), "llm")
        selected_source = (
            "llm" if config["backend"] == "llm" else config["model_source"]
        )
        self.source.setCurrentIndex(self.source.findData(selected_source))
        self.preset = QComboBox()
        self.preset.setObjectName("modelPreset")
        self.preset.addItems(MODEL_PRESETS)
        self.preset.setCurrentText(config["preset"])
        self.model_language = QComboBox()
        for key, label in LANGUAGES.items():
            self.model_language.addItem(label, key)
        self.model_language.setCurrentIndex(
            self.model_language.findData(config["language"])
        )
        self.det_directory = QLineEdit(config["det_model_dir"])
        self.det_directory.setObjectName("detModelDirectory")
        self.rec_directory = QLineEdit(config["rec_model_dir"])
        self.rec_directory.setObjectName("recModelDirectory")
        self.official_widget = QWidget()
        official = QFormLayout(self.official_widget)
        official.setContentsMargins(0, 0, 0, 0)
        official.addRow(self.text("Model", "模型"), self.preset)
        official.addRow(
            self.text("v5 mobile language", "v5 mobile 语言"), self.model_language
        )
        self.local_widget = QWidget()
        local = QFormLayout(self.local_widget)
        local.setContentsMargins(0, 0, 0, 0)
        local.addRow(
            self.text("Detection model", "检测模型"), self.directory_row(self.det_directory)
        )
        local.addRow(
            self.text("Recognition model", "识别模型"),
            self.directory_row(self.rec_directory),
        )
        note = QLabel(
            self.text(
                "Select exported Paddle inference directories containing inference.yml, "
                "inference.json (or .pdmodel) and inference.pdiparams.",
                "请选择导出的 Paddle 推理目录，包含 inference.yml、"
                "inference.json（或 .pdmodel）和 inference.pdiparams。",
            )
        )
        note.setWordWrap(True)
        local.addRow(note)
        self.llm_widget = QWidget()
        llm = QFormLayout(self.llm_widget)
        llm.setContentsMargins(0, 0, 0, 0)
        self.base_url = QLineEdit(service.get("base_url", ""))
        self.base_url.setObjectName("llmBaseUrl")
        self.base_url.setPlaceholderText("https://your-service.example/v1")
        self.model_name = QLineEdit(service.get("model", ""))
        self.model_name.setObjectName("llmModelName")
        model_hint = self.text(
            "API model ID, e.g. deepseek-flash; not the API key's name",
            "填写模型 ID，例如 deepseek-flash；不要填 API Key 的备注名",
        )
        self.model_name.setPlaceholderText(model_hint)
        self.model_name.setToolTip(model_hint)
        self.key_edit = QLineEdit(api_key)
        self.key_edit.setObjectName("llmApiKey")
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText(
            self.text(
                "Session only; leave blank to use PPOCRLABEL_API_KEY",
                "仅本次使用；留空使用环境变量 PPOCRLABEL_API_KEY",
            )
        )
        self.instructions = QPlainTextEdit(config["instructions"])
        self.instructions.setObjectName("annotationInstructions")
        self.instructions.setFixedHeight(90)
        llm.addRow(self.text("API base URL", "API 服务地址"), self.base_url)
        llm.addRow(self.text("Vision model name", "视觉模型名称"), self.model_name)
        llm.addRow("API Key", self.key_edit)
        llm.addRow(self.text("Annotation instructions", "附加标注说明"), self.instructions)
        llm_note = QLabel(
            self.text(
                "Whole images are sent to this service when automatic labeling starts. "
                "The model must support image input and return text polygons. Check the results before confirming.",
                "点击自动标注后会将整图发送至此服务。模型需支持图片输入和文字定位，" "返回的框与文字直接用于标注，请人工检查后确认。",
            )
        )
        llm_note.setWordWrap(True)
        llm.addRow(llm_note)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow(self.text("Model source", "模型来源"), self.source)
        layout.addLayout(form)
        layout.addWidget(self.official_widget)
        layout.addWidget(self.local_widget)
        layout.addWidget(self.llm_widget)
        if cli_overrides:
            notice = QLabel(
                self.text(
                    "Explicit command-line model arguments take priority for this session.",
                    "本次启动显式指定的命令行模型参数优先于目录配置。",
                )
            )
            notice.setWordWrap(True)
            layout.addWidget(notice)
        hint = QLabel(
            self.text(
                "Settings are saved for the current image directory. Confirmed images are skipped.",
                "设置按当前图片目录保存。批量自动标注会跳过已确认（√）图片。",
            )
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.source.currentIndexChanged.connect(self.update_fields)
        self.preset.currentIndexChanged.connect(self.update_fields)
        self.update_fields()

    def text(self, english, chinese):
        return chinese if self.language == "ch" else english

    def directory_row(self, edit):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit)
        browse = QPushButton(self.text("Browse…", "浏览…"))
        browse.clicked.connect(lambda: self.browse(edit))
        layout.addWidget(browse)
        return row

    def browse(self, edit):
        directory = QFileDialog.getExistingDirectory(
            self, self.windowTitle(), edit.text()
        )
        if directory:
            edit.setText(directory)

    def update_fields(self):
        source = self.source.currentData()
        official = source == "official"
        self.official_widget.setVisible(official)
        self.local_widget.setVisible(source == "local")
        self.llm_widget.setVisible(source == "llm")
        self.model_language.setEnabled(self.preset.currentText() == "PP-OCRv5_mobile")

    def config(self):
        source = self.source.currentData()
        return {
            "backend": "llm" if source == "llm" else "paddle",
            "model_source": self.previous_config["model_source"]
            if source == "llm"
            else source,
            "preset": self.preset.currentText(),
            "language": self.model_language.currentData(),
            "det_model_dir": self.det_directory.text().strip(),
            "rec_model_dir": self.rec_directory.text().strip(),
            "instructions": self.instructions.toPlainText(),
        }

    def service(self):
        return {
            "base_url": self.base_url.text().strip(),
            "model": self.model_name.text().strip(),
        }
