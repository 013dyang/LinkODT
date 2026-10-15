"""Check bilingual guide assets, English localization and navigation after a build."""
from pathlib import Path
import hashlib,json,re
from urllib.parse import urlsplit,unquote
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
GUIDE=ROOT.parent/'使用说明文档'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p in (GUIDE/'source_en').glob('*.rst'):
    assert not re.search(r'[\u3400-\u9fff]',p.read_text(encoding='utf-8')),p.name
manifest=json.loads((GUIDE/'evidence/screenshot_manifest.json').read_text(encoding='utf-8'))
assert len(manifest)==20
for lang in ('source','source_en'):
    used=[]
    for chapter in (GUIDE/lang).glob('*.rst'):
        text=chapter.read_text(encoding='utf-8')
        assert not re.search(r'SHA[- ]?256',text,re.I),chapter
        used.extend(re.findall(r'^\.\. figure:: _static/screenshots/(\S+)',text,re.M))
    assert len(used)==len(manifest) and set(used)=={item['file'] for item in manifest},lang
for item in manifest:
    for lang in ('source','source_en'):
        path=GUIDE/lang/'_static/screenshots'/item['file']
        assert digest(path)==item['sha256'],f'Screenshot differs from current capture: {path}'
    assert 'cursor' in item['capture'].lower()
    for output in (GUIDE/'html',GUIDE/'en/html'):
        assert digest(output/'_static/screenshots'/item['file'])==item['sha256'],item['file']
pages={p.resolve():BeautifulSoup(p.read_text(encoding='utf-8'),'html.parser')
       for base in (GUIDE/'html',GUIDE/'en/html') for p in base.glob('*.html')}
links=0;errors=[];english_pages=0
for p,soup in pages.items():
    footer=soup.select_one('#guide-footer')
    assert footer is not None,p
    assert not any(word in footer.get_text() for word in ('Eastern Institute','宁波','东方理工')),p
    assert 'Dayong Jin’s ODT' in footer.get_text(),p
    assert footer.get_text().count('Dan LI')==1 and footer.get_text().count('Dong YANG')==1,p
    for logo in soup.select('.sidebar-logo'):
        assert logo['src']=='_static/linkodt_icon.png',p
        assert digest(p.parent/logo['src'])==digest(ROOT/'assets/linkodt_icon.png'),p
    if p.parent== (GUIDE/'en/html').resolve():
        english_pages+=1
        assert 'All rights reserved.' in footer.get_text(),p
        article=soup.find('article')
        if article:
            assert not re.search(r'[\u3400-\u9fff]',article.get_text()),p
    for tag in soup.find_all(['a','img','script','link']):
        url=tag.get('href') or tag.get('src')
        if not url:continue
        u=urlsplit(url)
        if u.scheme or u.netloc:continue
        links+=1;target=(p.parent/unquote(u.path)).resolve() if u.path else p
        if not target.exists():errors.append((str(p),url))
        elif u.fragment and target in pages and not pages[target].find(id=unquote(u.fragment)):
            errors.append((str(p),url))
assert not errors,errors
report={'html_pages':len(pages),'english_pages_without_chinese_content':english_pages,
        'local_links_checked':links,'broken_links':errors,'current_application_icon_used':True,
        'school_affiliation_removed':True,'team_attribution_present':True,'current_screenshots_verified':len(manifest)*2}
(GUIDE/'evidence/document_style_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
