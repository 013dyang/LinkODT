from pathlib import Path
import shutil,json
ROOT=Path('C:/Users/sgc-sgc/Desktop/LinkODT_3.2_Windows').resolve()
assert ROOT.name=='LinkODT_3.2_Windows' and (ROOT/'LinkODT.exe').is_file()
targets=[ROOT/'_internal/.idea',ROOT/'_internal/.linkodt_build',ROOT/'_internal/.linkodt_runtime']
removed=[]
for target in targets:
    if target.exists():
        resolved=target.resolve();assert resolved.is_relative_to(ROOT/'_internal') and not target.is_symlink()
        shutil.rmtree(target);removed.append(str(target))
# Unit-test folders are not needed by the review application.
for target in sorted((ROOT/'_internal').rglob('tests'),key=lambda p:len(p.parts),reverse=True):
    if target.is_dir() and target.exists():
        resolved=target.resolve();assert resolved.is_relative_to(ROOT/'_internal') and not target.is_symlink()
        shutil.rmtree(target);removed.append(str(target))
print(json.dumps({'removed_generated_development_and_test_folders':len(removed)}))
