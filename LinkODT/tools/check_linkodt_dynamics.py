"""Regress generation -> mask view -> frame refresh through the real Qt worker."""
from pathlib import Path
import sys,os,time,json,shutil,threading
os.environ['QT_API']='pyside6';sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6.QtCore import qInstallMessageHandler
from PySide6.QtWidgets import QApplication
from ui_theme import configure_application
from punctate_dynamics_module import PunctateDynamicsWidget
app=QApplication.instance() or QApplication([]);configure_application(app)
data=ROOT/'review/demo_data';data.mkdir(exist_ok=True)
shutil.copy2(ROOT.parent/'使用说明文档/demo_data/DEMO_time.tif',data/'DEMO_time.tif')
messages=[]
previous=qInstallMessageHandler(lambda kind,context,message: messages.append(message))
widget=PunctateDynamicsWidget();widget.resize(1440,860);widget.show()
widget._add_dataset(data/'DEMO_time.tif');widget.dataset_list.setCurrentRow(0)
widget.generate_masks()
deadline=time.time()+90
while widget._threads and time.time()<deadline: app.processEvents();time.sleep(.02)
assert not widget._threads,'Mask generation did not finish'
ds=widget.current_dataset(); assert len(ds.frames)==8
assert len(list(ds.processed_dir.glob('*_mask.tif')))==8
assert widget.view_mode=='mask' and widget.current_mask is not None
widget.frame_slider.setValue(7)
deadline=time.time()+.3
while time.time()<deadline: app.processEvents();time.sleep(.01)
assert widget.current_frame_index==7
assert widget.current_mask is not None and widget.current_mask.shape==(192,192)
bad=[m for m in messages if any(s in m for s in ['Timers cannot','Cannot create children','different thread','QObject::'])]
assert not bad,bad
result={'generated_masks':8,'mask_view_after_generation':True,'frame_navigation_after_generation':True,'qt_thread_warnings':bad}
(ROOT/'review/dynamics_regression.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
widget.close();app.processEvents();qInstallMessageHandler(previous)
