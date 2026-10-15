"""QObject receivers ensure background completion handlers run on the GUI thread."""
from PySide6.QtCore import QObject, Slot, Qt

class GuiTaskCallbacks(QObject):
    def __init__(self, parent, thread, worker, finished, failed, cleanup):
        super().__init__(parent)
        self._thread = thread
        self._worker = worker
        self._finished = finished
        self._failed = failed
        self._cleanup = cleanup
        worker.finished.connect(self.finished, Qt.QueuedConnection)
        worker.error.connect(self.failed, Qt.QueuedConnection)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(self.cleanup, Qt.QueuedConnection)

    @Slot(object)
    def finished(self, result):
        try:
            self._finished(result)
        except Exception:
            import traceback
            self._failed(traceback.format_exc())
        finally:
            self._thread.quit()

    @Slot(str)
    def failed(self, error):
        try:
            self._failed(error)
        finally:
            self._thread.quit()

    @Slot()
    def cleanup(self):
        self._cleanup()
        self.deleteLater()
