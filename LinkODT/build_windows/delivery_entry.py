"""One executable for the main application, 3D viewer and delivery verification."""
import sys,os,json,traceback
from pathlib import Path
os.environ['QT_API']='pyside6'
if '--delivery-check' in sys.argv:
    report_path=Path(sys.argv[sys.argv.index('--delivery-check')+1])
    report={}
    try:
        import torch,numpy,open3d,scipy,pandas,tifffile,pyvista,PySide6
        from PySide6.QtWidgets import QApplication
        from ui_theme import configure_application
        import LinkNex as core
        app=QApplication([]);configure_application(app)
        window=core.LinkNexMainWindow();window.show();app.processEvents()
        report['frozen']=bool(getattr(sys,'frozen',False))
        report['runtime_modules']={m.__name__:str(m.__file__) for m in (torch,numpy,open3d,scipy,pandas,tifffile,pyvista,PySide6)}
        report['torch_version']=torch.__version__;report['cuda']=torch.cuda.is_available()
        report['modules_constructed']=[]
        for name in ('preprocessing','semantic','instance','network','dynamics'):
            getattr(window,'_open_'+name)();app.processEvents();report['modules_constructed'].append(name)
        from semantic_models import build_semantic_model
        import torch.nn as nn,torch.nn.functional as F
        torch.set_num_threads(2)
        model,_=build_semantic_model('unet',torch,nn,F);model.eval()
        with torch.no_grad():result=model(torch.zeros(1,1,64,64))
        report['unet_output']=list(result.shape)
        repo=Path(sys._MEIPASS)/'dinov2-main'
        encoder=torch.hub.load(str(repo),'dinov2_vits14',source='local',pretrained=False,verbose=False)
        report['dino_local_encoder']=type(encoder).__name__
        # Exercise dynamically embedded geometry/network backends, not just imports.
        test_dir=report_path.parent/'delivery_backend_check';test_dir.mkdir(exist_ok=True)
        zz,yy,xx=numpy.indices((12,48,48))
        foreground=(((zz-6)**2+(yy-17)**2<10)&(xx>6)&(xx<40)) | (((zz-6)**2+(xx-25)**2<9)&(yy>8)&(yy<33)) | (((zz-6)**2+(yy-37)**2<6)&(xx>12)&(xx<29))
        raw=test_dir/'test.tif';mask=test_dir/'test_mask.tif'
        tifffile.imwrite(raw,(foreground*1500).astype(numpy.uint16))
        tifffile.imwrite(mask,(foreground*255).astype(numpy.uint8))
        core.run_complete_pipeline(mask)
        processed=test_dir/'processed_output'
        assert list(processed.glob('*_instance_morphology_physical.csv'))
        network=core._load_hierarchical_network_module()
        metrics=network.process_one_image(stem='test',input_path=processed,output_path=processed,coordinate_scale_xyz=(1.,1.,1.),skeleton_neighbor_radius=1.8,social_distance_threshold=5.,whole_physical_distance_threshold=5.,contact_region_radius=3.,omega_values=(.01,.03,.1,.3,1.),maximum_efficiency_nodes=250,maximum_resistance_nodes=400,maximum_transport_nodes=300,minimum_cross_nodes=6,low_local_efficiency_threshold=.25,hub_fraction=.1,size_tail_fraction=.25,whole_physical_centrality_sample_nodes=300,whole_physical_spectral_entropy_max_nodes=800,whole_physical_weighted_efficiency_max_nodes=800,workers=1,random_seed=42)
        report['instance_backend_completed']=True;report['network_metric_fields']=len(metrics)
        assert (processed/'test_hierarchical_network.pkl').exists()
        from visualization_app import VisualizationWindow
        viz=VisualizationWindow();viz.show();app.processEvents()
        sample=Path(report_path.parent)/'delivery_check_surface.obj'
        pyvista.Sphere().save(sample)
        viz.load_objects([sample]);viz.render_window.Render();app.processEvents()
        report['visualization_objects']=len(viz.objects)
        report['guide_exists']=(Path(sys.executable).parent/'UserGuide/index.html').exists()
        report['icon_exists']=(Path(sys._MEIPASS)/'assets/linkodt_app.ico').exists()
        viz.close();window.close();report['passed']=True
    except BaseException:
        report['passed']=False;report['error']=traceback.format_exc()
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    sys.exit(0 if report['passed'] else 1)
elif '--visualization' in sys.argv:
    from visualization_app import main
    main()
else:
    from LinkNex import main
    main()
