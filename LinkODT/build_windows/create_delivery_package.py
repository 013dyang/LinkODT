"""Assemble the runnable Windows package and read-only bilingual HTML guide."""
from pathlib import Path
import shutil,json,sys,zipfile,hashlib
import importlib.metadata
from packaging.requirements import Requirement
from urllib.parse import urlsplit,unquote
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'LinkODT';GUIDE=ROOT/'使用说明文档'
OUTPUT=ROOT.parent/'LinkODT_3.2_Windows'
sys.path.insert(0,str(APP/'review/delivery_addons'))
if sys.argv[1]=='assemble':
    if OUTPUT.exists():
        assert hashlib.sha256((OUTPUT/'LinkODT.exe').read_bytes()).digest()==hashlib.sha256((APP/'review/delivery_dist/LinkODT/LinkODT.exe').read_bytes()).digest(),'Existing folder belongs to a different build'
    else:shutil.copytree(APP/'review/delivery_dist/LinkODT',OUTPUT)
    for p in Path(importlib.metadata.distribution('mypy').locate_file('')).glob('*__mypyc*.pyd'):
        shutil.copy2(p,OUTPUT/'_internal'/p.name)
    def copy_current(source,destination):
        source=Path(source);destination=Path(destination)
        if destination.exists() and source.stat().st_size==destination.stat().st_size and source.stat().st_mtime_ns==destination.stat().st_mtime_ns:return str(destination)
        return shutil.copy2(source,destination)
    # Open3D imports its web-view helpers at startup. Keep their pure-Python
    # packages intact because Plotly uses dynamic module discovery.
    queue=['plotly','dash','umap-learn','imageio-ffmpeg','submitit','iopath','fvcore','omegaconf','threadpoolctl'];bundled=set()
    while queue:
        name=queue.pop()
        key=name.lower().replace('_','-')
        if key in bundled:continue
        distribution=importlib.metadata.distribution(name);bundled.add(key)
        for requirement in distribution.requires or []:
            parsed=Requirement(requirement)
            if parsed.marker is None or parsed.marker.evaluate({'extra':''}):queue.append(parsed.name)
        if key in ('numpy','scipy','scikit-learn'):continue
        if key=='pywin32':
            native=Path(distribution.locate_file('win32'))
            for p in native.glob('*.pyd'):copy_current(p,OUTPUT/'_internal'/p.name)
            for p in (native/'lib').glob('*.py'):copy_current(p,OUTPUT/'_internal'/p.name)
            continue
        roots={f.parts[0] for f in distribution.files or [] if f.parts and f.parts[0] not in ('.','..','__pycache__') and ':' not in f.parts[0]}
        for root in roots:
            source=Path(distribution.locate_file(root));destination=OUTPUT/'_internal'/root
            if source.is_dir():
                shutil.copytree(source,destination,dirs_exist_ok=True,copy_function=copy_current,ignore=shutil.ignore_patterns('__pycache__','*.pyc','tests'))
            elif source.is_file():copy_current(source,destination)
        print('Bundled',name,flush=True)
    license_names=['numpy','scipy','pandas','torch','torchvision','PySide6','shiboken6','vtk','open3d','pyvista','pyvistaqt','qtpy','scikit-image','scikit-learn','matplotlib','Pillow','tifffile','opencv-python','networkx','timm','torchmetrics','imageio','imageio-ffmpeg','numba','llvmlite','umap-learn','pynndescent','omegaconf','fvcore','iopath','plotly','dash']
    for name in license_names:
        try:distribution=importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:continue
        for item in distribution.files or []:
            if not item.name.upper().startswith(('LICENSE','COPYING','NOTICE')):continue
            source=Path(distribution.locate_file(item))
            if not source.is_file():continue
            destination=OUTPUT/'_internal/THIRD_PARTY_LICENSES'/name/Path(*[p for p in item.parts if p not in ('.','..')])
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    target=OUTPUT/'UserGuide';target.mkdir(exist_ok=True)
    shutil.copy2(GUIDE/'index.html',target/'index.html')
    # Source views are disabled. Keep only files used by the published HTML pages.
    for source,dest in ((GUIDE/'html',target/'html'),(GUIDE/'en/html',target/'en/html')):
        used=set()
        for p in source.glob('*.html'):
            used.add(p.relative_to(source))
            soup=BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser')
            for tag in soup.find_all(['img','script','link','a']):
                value=tag.get('src') or tag.get('href')
                if value and not value.startswith(('http:','https:','//')):
                    resource=source/value.split('?')[0].split('#')[0]
                    if resource.is_file():used.add(resource.relative_to(source))
        # CSS and JS references include fonts, theme styles and search data.
        for p in (source/'_static').rglob('*'):
            if p.is_file() and 'screenshots' not in p.parts:
                used.add(p.relative_to(source))
        for p in (source/'searchindex.js',source/'objects.inv'):
            if p.exists():used.add(p.relative_to(source))
        for relative in used:
            destination=dest/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/relative,destination)
    (OUTPUT/'README.txt').write_text('''LinkODT 3.2 — Windows review package
14 October 2026

QUICK START
1. Extract the entire ZIP to a normal writable folder.
2. Double-click LinkODT.exe. Python installation is not required.
3. Open Help > User guide, or UserGuide/index.html.
4. Keep _internal and UserGuide beside LinkODT.exe.

Windows 10/11, 64-bit. This package uses CPU processing.
Image and mask inspection, instance/network analysis and 3D visualization
are included. Model prediction requires a compatible trained checkpoint.
Training data and model weights are not included. SegFormer and local
AI Director require additional dependencies/models and are not supplied
in this review package. Use the source edition for GPU training.

中文说明
1. 将 ZIP 完整解压到普通、可写的文件夹。
2. 双击 LinkODT.exe，无需安装 Python。
3. 通过 Help > User guide 或 UserGuide/index.html 打开中英文手册。
4. _internal 和 UserGuide 必须与 LinkODT.exe 保持在同一文件夹。

适用于 64 位 Windows 10/11，本包使用 CPU 处理。
包含图像与掩膜查看、实例/网络分析及三维可视化。
模型预测需要相匹配的训练权重；不附带训练数据和模型权重。
SegFormer、当地 AI Director 所需的附加依赖与模型未包含；
GPU 训练请使用源码环境。

© 2026 Dan LI and Dong YANG · Dayong Jin’s ODT Group.
All rights reserved. Third-party components retain their original licenses.
'''.replace('当地 AI','本地 AI'),encoding='utf-8-sig')
    pages={p.resolve():BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser') for p in target.rglob('*.html')}
    errors=[];links=0
    for p,soup in pages.items():
        for tag in soup.find_all(['a','img','script','link']):
            value=tag.get('href') or tag.get('src')
            if not value:continue
            parsed=urlsplit(value)
            if parsed.scheme or parsed.netloc:continue
            resource=(p.parent/unquote(parsed.path)).resolve() if parsed.path else p
            links+=1
            if not resource.exists():errors.append([str(p),value])
            elif parsed.fragment and resource in pages and not pages[resource].find(id=unquote(parsed.fragment)):
                errors.append([str(p),value])
    assert not errors,errors
    (APP/'review/delivery_guide_check.json').write_text(json.dumps({'html_pages':len(pages),'links_checked':links,'broken_links':errors},indent=2),encoding='utf-8')
    print(OUTPUT)
elif sys.argv[1]=='zip':
    report=json.loads((APP/'review/delivery_check.json').read_text(encoding='utf-8'))
    assert report['passed'] and report['frozen']
    assert all('_internal' in path for path in report['runtime_modules'].values()),report
    archive=OUTPUT.parent/(OUTPUT.name+'.zip');assert not archive.exists()
    files=[p for p in OUTPUT.rglob('*') if p.is_file()]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in files:z.write(p,Path(OUTPUT.name)/p.relative_to(OUTPUT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.infolist())==len(files)
    metadata={'folder':str(OUTPUT),'zip':str(archive),'files':len(files),'folder_bytes':sum(p.stat().st_size for p in files),'zip_bytes':archive.stat().st_size,'archive_test_passed':True}
    (APP/'review/delivery_package_report.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata))
