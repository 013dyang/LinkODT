"""Small complete source update for a collaborator with the existing environment."""
from pathlib import Path
import ast,json,shutil,zipfile,hashlib,os,re
from urllib.parse import urlsplit,unquote
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[2];APP=ROOT/'LinkODT'
DEST=ROOT.parent/'LinkODT_3.2_Update'
if DEST.exists():assert (DEST/'LinkODT/LinkNex.py').read_bytes()==(APP/'LinkNex.py').read_bytes(),'Existing folder belongs to another version'
DEST.mkdir(exist_ok=True);code=DEST/'LinkODT';code.mkdir(exist_ok=True)
main_files=['LinkODT.py','LinkNex.py','gui_tasks.py','ui_theme.py','semantic_models.py','punctate_dynamics_module.py','linknex_module34_directload_patch.py','microscopy_segmentation_augmentation_full_v2.py','paired_tiff_segmentation_augmentation_full.py','evaluate_masks_with_panels_v2.py','RUN_LinkODT.bat','VERSION.txt','requirements_semantic_optional.txt','environment.yml']
for name in main_files:shutil.copy2(APP/name,code/name)
for folder in ('backends','ssl','visualization','dinov2-main'):
    base=APP/folder
    for source in base.rglob('*'):
        if not source.is_file():continue
        rel=source.relative_to(base)
        if any(p in ('__pycache__','.github','build_windows','docs','notebooks','scripts') for p in rel.parts):continue
        if source.suffix!='.py' and not source.name.startswith('LICENSE') and source.name not in ('RUN_Visualization.bat','requirements_visualization.txt','_studio_core.pyc','_studio_io.pyc','_studio_director.pyc','studio_origin.json'):continue
        target=code/folder/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
# Keep DINOv2 runtime configuration files used by source workflows.
for source in (APP/'dinov2-main/dinov2/configs').rglob('*.yaml'):
    target=code/'dinov2-main'/source.relative_to(APP/'dinov2-main');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
(code/'assets').mkdir(exist_ok=True)
for name in ('linkodt_app.ico','linkodt_icon.png','linkodt_wordmark.png','linkodt_wordmark_dark.png','check_white.svg'):
    shutil.copy2(APP/'assets'/name,code/'assets'/name)
(code/'tools').mkdir(exist_ok=True)
for name in ('check_environment.py','download_pretrained_models.py'):
    shutil.copy2(APP/'tools'/name,code/'tools'/name)
requirements=(APP/'requirements_linknex.txt').read_text(encoding='utf-8')
requirements=requirements.replace('# Windows EXE build only\npyinstaller\npyinstaller-hooks-contrib\n','')
(code/'requirements_linknex.txt').write_text(requirements,encoding='utf-8')
shutil.copytree(ROOT.parent/'LinkODT_3.2_Windows/UserGuide',DEST/'UserGuide',dirs_exist_ok=True)
guide=DEST/'UserGuide';shared=guide/'_shared/images';shared.mkdir(parents=True,exist_ok=True)
htmls={p:BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser') for p in guide.rglob('*.html')}
referenced=set()
for p,soup in htmls.items():
    if p.name=='setup-linknex.html':
        standalone=soup.find(id='windows-standalone-package') or soup.find(id='windows')
        if standalone is None:
            standalone=next((s for s in soup.find_all('section') if s.find(['h2','h3']) and 'Windows' in s.find(['h2','h3']).get_text()),None)
        if standalone:
            paragraph=standalone.find('p')
            if paragraph and paragraph.contents:
                first=paragraph.contents[0]
                if isinstance(first,str):
                    prefix='If you are using the standalone EXE package, ' if 'en' in p.parts else '使用独立 EXE 包时，'
                    first.replace_with(prefix+str(first))
    for tag in soup.find_all(['a','img','link']):
        value=tag.get('href') or tag.get('src')
        if not value:continue
        u=urlsplit(value)
        if u.scheme or u.netloc:continue
        target=(p.parent/unquote(u.path)).resolve()
        if target.suffix.lower()=='.png' and target.exists():referenced.add(target)
mapping={};canonical={}
for p in sorted(referenced):
    if p.is_relative_to(shared.resolve()):continue
    key=hashlib.sha256(p.read_bytes()).hexdigest()
    if key not in canonical:
        target=shared/p.name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=key:target=shared/(p.stem+'_'+str(len(canonical))+p.suffix)
        shutil.copy2(p,target);canonical[key]=target.resolve()
    mapping[p]=canonical[key]
for p,soup in htmls.items():
    for tag in soup.find_all(['a','img','link']):
        for attribute in ('src','href'):
            value=tag.get(attribute)
            if not value:continue
            u=urlsplit(value)
            if u.scheme or u.netloc:continue
            original=(p.parent/unquote(u.path)).resolve()
            if original in mapping:tag[attribute]=os.path.relpath(mapping[original],p.parent).replace(os.sep,'/')
    p.write_text(str(soup),encoding='utf-8')
# Screenshots keep their original pixels and full resolution. Both languages
# and the clickable full-size views share a single copy of each image.
for p in list(guide.rglob('*.png')):
    if not p.resolve().is_relative_to(shared.resolve()):p.unlink()
(DEST/'RUN_LinkODT.bat').write_bytes(b'@echo off\r\nsetlocal\r\ncd /d "%~dp0LinkODT"\r\ncall RUN_LinkODT.bat\r\n')
(DEST/'README.txt').write_text('''LinkODT 3.2 — program update / 程序更新包

For the collaborator who already runs the original Python project.
This package contains all current program modules and the bilingual guide.
It reuses your existing Python environment; it is not a standalone EXE package.

1. Extract to a new folder. Keep the old project unchanged.
2. Activate the Python/Conda environment used for the original project.
3. Double-click RUN_LinkODT.bat, or run python LinkODT/LinkODT.py.
4. Open UserGuide/index.html or Help > User guide.
5. Use your existing data and model files. Copy required weights into
   LinkODT/models/, or select your trained checkpoint in the interface.

适用于原 Python 项目的开发者／合作者，沿用原有运行环境。
本包保留全部当前程序模块与中英文手册，不附带大型运行库。
1. 解压到新文件夹，保留旧项目。
2. 激活原程序使用的 Python／Conda 环境。
3. 双击 RUN_LinkODT.bat，或执行 python LinkODT/LinkODT.py。
4. 通过 UserGuide/index.html 或 Help > User guide 阅读手册。
5. 继续使用原有数据和模型；需要的权重放入 LinkODT/models/，
   或在界面内选择已有训练权重。

If needed, set LINKODT_PYTHON to your existing Python executable.
Only missing packages need installation; preserve your working PyTorch/CUDA setup.
如需指定环境，可设置 LINKODT_PYTHON。缺少依赖时按程序提示补装，
保留已有可用的 PyTorch／CUDA 配置。

This is a source update. Replacing an old EXE alone is not supported.
本包通过已有 Python 环境启动，不用于单独替换旧版 EXE。

© 2026 Dan LI and Dong YANG · Dayong Jin’s ODT Group. All rights reserved.
Third-party code retains its original licenses.
''',encoding='utf-8-sig')
for p in code.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
pages={p.resolve():BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser') for p in (DEST/'UserGuide').rglob('*.html')}
links=0
for p,soup in pages.items():
    for tag in soup.find_all(['a','img','script','link']):
        value=tag.get('href') or tag.get('src')
        if not value:continue
        parsed=urlsplit(value)
        if parsed.scheme or parsed.netloc:continue
        links+=1;target=(p.parent/unquote(parsed.path)).resolve() if parsed.path else p
        assert target.exists(),(p,value)
        if parsed.fragment and target in pages:assert pages[target].find(id=unquote(parsed.fragment)),(p,value)
archive=DEST.parent/(DEST.name+'.zip')
files=[p for p in DEST.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in files:z.write(p,Path(DEST.name)/p.relative_to(DEST))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
report={'folder':str(DEST),'archive':str(archive),'zip_bytes':archive.stat().st_size,'folder_bytes':sum(p.stat().st_size for p in files),'files':len(files),'html_pages':len(pages),'links_checked':links,'source_syntax_passed':True,'archive_check_passed':True}
(APP/'review/source_update_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
