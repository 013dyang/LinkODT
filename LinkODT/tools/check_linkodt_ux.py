"""Focused regression checks for LinkODT identity, queued callbacks and compatibility."""
from pathlib import Path
import os,sys,json,ast,threading,time
os.environ['QT_API']='pyside6'
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from unittest.mock import patch
from PySide6.QtGui import QCloseEvent
from PySide6.QtCore import QObject,QThread,Signal,Slot,QEventLoop,QTimer
from PySide6.QtWidgets import QApplication,QMessageBox
from ui_theme import configure_application
from gui_tasks import GuiTaskCallbacks
app=QApplication.instance() or QApplication([]); configure_application(app)
results={'application_name':app.applicationName(),'font':app.font().family(),'numpy_version':np.__version__}
assert app.applicationName()=='LinkODT' and app.font().family()=='Segoe UI'

class Worker(QObject):
    finished=Signal(object); error=Signal(str)
    @Slot()
    def run(self): self.finished.emit({'thread':threading.get_ident()})

gui_thread=threading.get_ident(); received=[]; cleaned=[]
thread=QThread(); worker=Worker(); worker.moveToThread(thread)
def completed(result): received.append((threading.get_ident(),result['thread']))
callbacks=GuiTaskCallbacks(app,thread,worker,completed,lambda err: (_ for _ in ()).throw(AssertionError(err)),lambda: cleaned.append(True))
thread.started.connect(worker.run); thread.start()
deadline=time.time()+5
while not cleaned and time.time()<deadline: app.processEvents(); time.sleep(.01)
assert received and received[0][0]==gui_thread and received[0][1]!=gui_thread
assert cleaned and not thread.isRunning()
results['queued_worker_callback_on_gui_thread']=True

import LinkNex as core
network=core._load_hierarchical_network_module()
assert abs(float(network._trapezoid(np.array([0.,1.,0.]),dx=1.))-1.)<1e-9
results['embedded_network_integral_compatible']=True
window=core.LinkNexMainWindow()
assert window.windowTitle().startswith('LinkODT')
assert not window.windowIcon().isNull()
assert window.semantic.brush_mode_btn.toolTip()
results['main_window_and_all_five_modules_constructed']=True
window.semantic.dirty=True
with patch.object(QMessageBox,'question',return_value=QMessageBox.Cancel):
    close=QCloseEvent();window.closeEvent(close)
    assert not close.isAccepted() and window.semantic.dirty
window.semantic.dirty=False
busy=QThread(window);busy.start()
deadline=time.time()+2
while not busy.isRunning() and time.time()<deadline:app.processEvents()
with patch.object(QMessageBox,'information',return_value=QMessageBox.Ok):
    close=QCloseEvent();window.closeEvent(close);assert not close.isAccepted()
busy.quit();busy.wait(3000)
results['unsaved_cancel_and_running_task_close_protected']=True
window._open_semantic();app.processEvents()
bounds=window.screen().availableGeometry()
assert window.width()<=bounds.width() and window.height()<=bounds.height()
results['workspace_fits_available_screen']=True
results['local_guide']=str(ROOT.parent/'使用说明文档/index.html')
window.close(); app.processEvents()
for p in ROOT.rglob('*.py'):
    if 'dinov2-main' in p.parts: continue
    ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
results['python_sources_parse']=True
(ROOT/'review/ux_regression.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
inventory=[]
for p in ROOT.rglob('*.py'):
    if 'dinov2-main' in p.parts:continue
    import hashlib
    inventory.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'lines':len(p.read_text(encoding='utf-8-sig').splitlines())})
(ROOT.parent/'使用说明文档/evidence/source_inventory_linkodt31.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(results,ensure_ascii=False))
