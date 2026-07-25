"""Qt panel for embedded Cursor SDK agent sessions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, QSettings, QThread, Signal, Slot
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from retro_harness.editor.cursor_agent import (
    CursorAgentSession,
    EditorAgentContext,
    build_agent_prompt,
    cursor_sdk_available,
    default_api_key,
    format_editor_context,
    list_model_ids,
    validate_api_key,
)

ContextProviderFn = Callable[[], EditorAgentContext | None]


@dataclass(frozen=True)
class CursorAgentPanelConfig:
    """Static configuration for an embedded Cursor agent panel."""

    workspace_cwd: Path
    settings_org: str
    settings_app: str
    session_name: str
    instructions: tuple[str, ...] = ()
    api_key_setting: str = "cursor/apiKey"
    model_setting: str = "cursor/model"
    default_model: str = "composer-2.5"


class _CursorAgentWorker(QObject):
    line_appended = Signal(str)
    status_changed = Signal(str)
    session_ready = Signal(str)
    run_finished = Signal(str)
    error_occurred = Signal(str)

    reset_session = Signal()
    shutdown_requested = Signal()

    def __init__(
        self,
        *,
        workspace_cwd: str,
        session_name: str,
        instructions: tuple[str, ...],
    ) -> None:
        super().__init__()
        self._workspace_cwd = workspace_cwd
        self._session_name = session_name
        self._instructions = instructions
        self._session: CursorAgentSession | None = None
        self._context_provider: ContextProviderFn | None = None
        self._published_context = ""
        self._api_key = ""
        self._model = ""

    def set_context_provider(self, provider: ContextProviderFn | None) -> None:
        self._context_provider = provider

    def set_published_context(self, text: str) -> None:
        self._published_context = text

    @Slot()
    def reset_session_slot(self) -> None:
        self._close_session()

    @Slot(str, str)
    def start_session(self, api_key: str, model: str) -> None:
        self._close_session()
        self._api_key = api_key.strip()
        self._model = model.strip()
        self.status_changed.emit("idle")

    @Slot(str)
    def send_prompt(self, user_message: str) -> None:
        if not self._ensure_session():
            return
        session = self._session
        if session is None:
            return
        context = self._context_provider() if self._context_provider else None
        try:
            for line in session.send(
                user_message,
                context=context,
                published_context=self._published_context or None,
            ):
                self.line_appended.emit(line)
            self.run_finished.emit("finished")
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.run_finished.emit("error")

    @Slot()
    def cancel_run(self) -> None:
        session = self._session
        if session is None:
            return
        try:
            session.cancel()
            self.status_changed.emit("cancelled")
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    @Slot()
    def shutdown_slot(self) -> None:
        self._close_session()

    def _ensure_session(self) -> bool:
        if self._session is not None:
            return True
        if not self._api_key or not self._model:
            self.error_occurred.emit("Start an agent session first.")
            return False
        try:
            self._session = CursorAgentSession(
                api_key=self._api_key,
                workspace_cwd=self._workspace_cwd,
                model=self._model,
                name=self._session_name,
                instructions=self._instructions,
            )
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            self.status_changed.emit("error")
            return False
        self.session_ready.emit(self._session.agent_id)
        self.status_changed.emit("ready")
        return True

    def _close_session(self) -> None:
        session = self._session
        self._session = None
        if session is None:
            return
        try:
            session.close()
        except Exception:
            pass


class CursorAgentPanel(QWidget):
    """Terminal-style Cursor agent panel for game editors."""

    def __init__(
        self,
        config: CursorAgentPanelConfig,
        *,
        context_provider: ContextProviderFn | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._context_provider = context_provider
        self._published_context = ""
        self._settings = QSettings(config.settings_org, config.settings_app)
        self._sdk_available = cursor_sdk_available()

        self.status_label = QLabel("SDK unavailable" if not self._sdk_available else "idle")
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("CURSOR_API_KEY")
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.context_preview = QPlainTextEdit()
        self.context_preview.setReadOnly(True)
        self.context_preview.setMaximumHeight(120)
        self.context_preview.setPlaceholderText(
            "Published editor context appears here. Use Publish Context while playing."
        )
        self.terminal = QPlainTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        terminal_font = QFont("Monospace")
        terminal_font.setStyleHint(QFont.StyleHint.Monospace)
        self.terminal.setFont(terminal_font)
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText(
            "Ask the agent about extraction, WRAM, map layers, or modding..."
        )
        self.prompt_edit.setMaximumHeight(90)

        self.publish_button = QPushButton("Publish Context")
        self.send_button = QPushButton("Send")
        self.cancel_button = QPushButton("Cancel")
        self.new_session_button = QPushButton("New Session")
        self.save_key_button = QPushButton("Save Key")

        self.publish_button.clicked.connect(self.publish_context)
        self.send_button.clicked.connect(self.send_prompt)
        self.cancel_button.clicked.connect(self.cancel_run)
        self.new_session_button.clicked.connect(self.start_session)
        self.save_key_button.clicked.connect(self.save_api_key)
        self.prompt_edit.installEventFilter(self)

        self._build_layout()
        self._restore_settings()
        self._setup_worker()
        self._set_controls_enabled(self._sdk_available)

    def set_context_provider(self, provider: ContextProviderFn | None) -> None:
        self._context_provider = provider
        if self._worker is not None:
            self._worker.set_context_provider(provider)

    def publish_context(self) -> None:
        if self._context_provider is None:
            QMessageBox.information(
                self,
                "No Context Provider",
                "This editor has not wired a live context provider yet.",
            )
            return
        context = self._context_provider()
        if context is None:
            QMessageBox.information(
                self,
                "No Context",
                "There is no live editor context to publish yet.",
            )
            return
        self._published_context = format_editor_context(context)
        self.context_preview.setPlainText(self._published_context)
        if self._worker is not None:
            self._worker.set_published_context(self._published_context)
        self._append_terminal("[context published]")
        preview = build_agent_prompt(
            "(preview)",
            instructions=self._config.instructions,
            context=context,
            published_context=self._published_context,
        )
        self.status_label.setText(f"context ready ({len(preview)} chars)")

    def send_prompt(self) -> None:
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            return
        if not self._sdk_available:
            self._show_sdk_install_help()
            return
        api_key = self.api_key_edit.text().strip() or default_api_key()
        if not api_key:
            QMessageBox.warning(
                self,
                "Missing API Key",
                "Set CURSOR_API_KEY in the environment or paste a key and click Save Key.",
            )
            return
        model = self.model_combo.currentText().strip() or self._config.default_model
        self._worker.start_session.emit(api_key, model)
        self.prompt_edit.clear()
        self._append_terminal(f"\n> {prompt}")
        self.status_label.setText("running")
        self._worker.send_prompt.emit(prompt)

    def cancel_run(self) -> None:
        if self._worker is not None:
            self._worker.cancel_run.emit()

    def start_session(self) -> None:
        if not self._sdk_available:
            self._show_sdk_install_help()
            return
        api_key = self.api_key_edit.text().strip() or default_api_key()
        if not api_key:
            QMessageBox.warning(
                self,
                "Missing API Key",
                "Set CURSOR_API_KEY in the environment or paste a key and click Save Key.",
            )
            return
        model = self.model_combo.currentText().strip() or self._config.default_model
        self._settings.setValue(self._config.model_setting, model)
        self._worker.reset_session.emit()
        self._append_terminal("[starting session]")
        self.status_label.setText("starting")
        self._worker.start_session.emit(api_key, model)

    def save_api_key(self) -> None:
        api_key = self.api_key_edit.text().strip()
        if not api_key:
            self._settings.remove(self._config.api_key_setting)
            self.status_label.setText("api key cleared")
            return
        if not self._sdk_available:
            self._settings.setValue(self._config.api_key_setting, api_key)
            self.status_label.setText("api key saved")
            return
        try:
            validate_api_key(api_key)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid API Key", str(exc))
            return
        self._settings.setValue(self._config.api_key_setting, api_key)
        self._populate_models(api_key)
        self.status_label.setText("api key saved")

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.shutdown_requested.emit()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)

    def eventFilter(self, watched: object, event: object) -> bool:
        prompt_edit = getattr(self, "prompt_edit", None)
        if watched is prompt_edit and isinstance(event, QKeyEvent):
            if event.type() == QEvent.Type.KeyPress and event.key() in {
                Qt.Key.Key_Return,
                Qt.Key.Key_Enter,
            }:
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self.send_prompt()
                    return True
        return super().eventFilter(watched, event)

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("API Key"))
        header.addWidget(self.api_key_edit, stretch=1)
        header.addWidget(self.save_key_button)
        header.addWidget(QLabel("Model"))
        header.addWidget(self.model_combo, stretch=1)
        header.addWidget(self.new_session_button)
        root.addLayout(header)

        root.addWidget(QLabel("Published Context"))
        root.addWidget(self.context_preview)
        root.addWidget(QLabel("Agent Terminal"))
        root.addWidget(self.terminal, stretch=1)

        prompt_row = QHBoxLayout()
        prompt_row.addWidget(self.prompt_edit, stretch=1)
        prompt_row.addWidget(self.publish_button)
        prompt_row.addWidget(self.send_button)
        prompt_row.addWidget(self.cancel_button)
        root.addLayout(prompt_row)
        root.addWidget(self.status_label)

    def _setup_worker(self) -> None:
        self._thread = QThread(self)
        self._worker = _CursorAgentWorker(
            workspace_cwd=str(self._config.workspace_cwd),
            session_name=self._config.session_name,
            instructions=self._config.instructions,
        )
        self._worker.moveToThread(self._thread)
        self._worker.set_context_provider(self._context_provider)
        self._worker.reset_session.connect(self._worker.reset_session_slot)
        self._worker.shutdown_requested.connect(self._worker.shutdown_slot)
        self._worker.line_appended.connect(self._append_terminal)
        self._worker.status_changed.connect(self.status_label.setText)
        self._worker.session_ready.connect(self._on_session_ready)
        self._worker.run_finished.connect(self._on_run_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._thread.start()

    def _restore_settings(self) -> None:
        saved_key = str(self._settings.value(self._config.api_key_setting, "") or "")
        if saved_key:
            self.api_key_edit.setText(saved_key)
        elif default_api_key():
            self.api_key_edit.setText(default_api_key())
        saved_model = str(
            self._settings.value(self._config.model_setting, self._config.default_model)
            or self._config.default_model
        )
        self.model_combo.addItem(saved_model)
        self.model_combo.setCurrentText(saved_model)
        if self._sdk_available and self.api_key_edit.text().strip():
            self._populate_models(self.api_key_edit.text().strip())

    def _populate_models(self, api_key: str) -> None:
        current = self.model_combo.currentText().strip()
        self.model_combo.clear()
        for model_id in list_model_ids(api_key):
            self.model_combo.addItem(model_id)
        if current:
            self.model_combo.setCurrentText(current)
        elif self.model_combo.count():
            self.model_combo.setCurrentIndex(0)

    def _append_terminal(self, text: str) -> None:
        if not text:
            return
        cursor = self.terminal.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        if self.terminal.toPlainText() and not text.startswith("\n"):
            cursor.insertText("\n")
        cursor.insertText(text)
        self.terminal.setTextCursor(cursor)
        self.terminal.ensureCursorVisible()

    def _on_session_ready(self, agent_id: str) -> None:
        self._append_terminal(f"[session ready] {agent_id}")
        self.status_label.setText("ready")

    def _on_run_finished(self, status: str) -> None:
        self.status_label.setText(status)

    def _on_error(self, message: str) -> None:
        self._append_terminal(f"[error] {message}")
        self.status_label.setText("error")

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.api_key_edit,
            self.save_key_button,
            self.model_combo,
            self.publish_button,
            self.send_button,
            self.cancel_button,
            self.new_session_button,
            self.prompt_edit,
        ):
            widget.setEnabled(enabled)

    def _show_sdk_install_help(self) -> None:
        QMessageBox.information(
            self,
            "Cursor SDK Not Installed",
            "Install the optional Cursor SDK extra with:\n\n"
            "uv sync --extra cursor",
        )


def attach_cursor_agent_dock(
    window: QMainWindow,
    *,
    config: CursorAgentPanelConfig,
    context_provider: ContextProviderFn | None = None,
    area: int | None = None,
) -> tuple[QDockWidget, CursorAgentPanel]:
    """Add a Cursor agent dock to a game editor main window."""

    from PySide6.QtCore import Qt

    dock = QDockWidget("Agent", window)
    dock.setObjectName("CursorAgentDock")
    panel = CursorAgentPanel(config, context_provider=context_provider)
    dock.setWidget(panel)
    dock.setMinimumWidth(360)
    dock_area = area if area is not None else Qt.DockWidgetArea.BottomDockWidgetArea
    window.addDockWidget(dock_area, dock)
    toggle = dock.toggleViewAction()
    toggle.setText("Agent Panel")
    menu_bar = window.menuBar()
    view_menu = None
    for action in menu_bar.actions():
        if action.text() == "View":
            view_menu = action.menu()
            break
    if view_menu is not None:
        view_menu.addAction(toggle)
    return dock, panel
