"""Exercise the real network backend and dynamics generation with copied demo data."""
from pathlib import Path
import sys,os,json,shutil,time,threading
sys.stdout.reconfigure(encoding='utf-8'); os.environ['QT_API']='pyside6'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import LinkNex as core
import tifffile,numpy as np
from PySide6.QtWidgets import QApplication,QMessageBox
from PySide6.QtCore import qInstallMessageHandler
app=QApplication.instance() or QApplication([])
data=ROOT/'review/demo_data';data.mkdir(exist_ok=True)
source=ROOT.parent/'使用说明文档/demo_data'
shutil.copytree(source/'processed_output',data/'processed_output',dirs_exist_ok=True)
shutil.copy2(source/'DEMO_time.tif',data/'DEMO_time.tif')
network=core._load_hierarchical_network_module()
metrics=network.process_one_image(stem='DEMO_static',input_path=data/'processed_output',output_path=data/'processed_output',coordinate_scale_xyz=(1.,1.,1.),skeleton_neighbor_radius=1.8,social_distance_threshold=5.,whole_physical_distance_threshold=5.,contact_region_radius=3.,omega_values=(.01,.03,.1,.3,1.),maximum_efficiency_nodes=250,maximum_resistance_nodes=400,maximum_transport_nodes=300,minimum_cross_nodes=6,low_local_efficiency_threshold=.25,hub_fraction=.1,size_tail_fraction=.25,whole_physical_centrality_sample_nodes=300,whole_physical_spectral_entropy_max_nodes=800,whole_physical_weighted_efficiency_max_nodes=800,workers=1,random_seed=42)
assert (data/'processed_output/DEMO_static_hierarchical_network.pkl').exists()
assert (data/'processed_output/DEMO_static_hierarchical_network_metrics.csv').exists()
result={'numpy_version':np.__version__,'complete_static_network_demo':True,'network_metric_fields':len(metrics)}
(ROOT/'review/data_regression.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
