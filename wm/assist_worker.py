"""Run explicit assisted actions off the desktop event loop."""
from PySide6.QtCore import QThread, Signal
from . import assisted
from .selection_edit import apply_selection


class AssistWorker(QThread):
    completed = Signal(object)

    def __init__(self, action, result, argument, parent=None, **options):
        super().__init__(parent)
        self.action, self.result, self.argument, self.options = action, result, argument, options

    def run(self):
        try:
            if self.action == 'selection':
                value = apply_selection(self.result, self.argument)
            elif self.action == 'find':
                value = assisted.find_strokes(self.result, self.argument, **self.options)
            else:
                value = assisted.apply_proposal(self.result, self.argument)
            self.completed.emit(dict(action=self.action, value=value, error=None))
        except Exception as exc:
            self.completed.emit(dict(action=self.action, value=None, error=str(exc)))
