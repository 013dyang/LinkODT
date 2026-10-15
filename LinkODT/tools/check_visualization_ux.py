"""Check the standalone visualization entry point against the shared LinkODT theme."""
from pathlib import Path
import sys,os,json
os.environ['QT_API']='pyside6';sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'visualization'),str(ROOT)]
from PySide6.QtWidgets import QApplication
from ui_theme import configure_application
from visualization_app import VisualizationWindow,APP_NAME
app=QApplication.instance() or QApplication([]);configure_application(app)
window=VisualizationWindow();assert window.windowTitle()==APP_NAME
assert APP_NAME=='LinkODT Visualization Studio'
assert not app.windowIcon().isNull()
result={'standalone_visualization_constructed':True,'window_title':window.windowTitle(),'font':app.font().family(),'icon_loaded':True}
(ROOT/'review/visualization_regression.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
window.close();app.processEvents()
