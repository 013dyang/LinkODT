project = 'LinkODT 使用说明'
author = 'Dan LI and Dong YANG'
release = '3.2 · 2026-10-14'
language = 'zh_CN'
extensions = []
html_theme = 'furo'
html_title = 'LinkODT 使用说明'
html_static_path = ['_static']
html_css_files = ['manual.css']
from pathlib import Path
templates_path = [str(Path(__file__).resolve().parent / '_templates')]
html_theme_options = {'light_css_variables': {'color-brand-primary':'#0067B1','color-brand-content':'#003B70'}}
exclude_patterns = []
html_show_copyright = False
html_show_sphinx = False
html_show_sourcelink = False

def repair_singlehtml_navigation(app, exception):
    """Make singlehtml's document-qualified fragments work without JS."""
    if exception or app.builder.name != 'singlehtml':
        return
    from pathlib import Path
    from bs4 import BeautifulSoup
    path = Path(app.outdir) / 'index.html'
    soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
    for marker in soup.find_all('span', id=lambda x: x and x.startswith('document-')):
        section = marker.find_next_sibling('section')
        if not section:
            continue
        prefix = marker['id'] + '--'
        mapping = {}
        for tag in [section, *section.find_all(id=True)]:
            old = tag.get('id')
            if old:
                mapping[old] = prefix + old
                tag['id'] = prefix + old
        for link in section.find_all(href=True):
            href = link['href']
            if href.startswith('#') and href[1:] in mapping:
                link['href'] = '#' + mapping[href[1:]]
    for tag in soup.find_all(href=True):
        href = tag['href']
        if href.startswith('#document-') and '#' in href[1:]:
            tag['href'] = '#' + href[1:].replace('#', '--', 1)
        elif href in ('genindex.html', 'search.html'):
            tag['href'] = '../html/' + href
    for form in soup.find_all('form', action='search.html'):
        form['action'] = '../html/search.html'
    path.write_text(str(soup), encoding='utf-8')

def setup(app):
    app.connect('builder-inited', validate_english_content)
    app.connect('build-finished', repair_singlehtml_navigation)
    app.connect('build-finished', install_language_buttons, priority=600)
    app.connect('html-page-context', language_page_context)

def language_page_context(app, pagename, templatename, context, doctree):
    english = app.config.language == 'en'
    single = app.builder.name == 'singlehtml'
    context['guide_english'] = english
    context['guide_title'] = 'LinkODT User Guide' if english else 'LinkODT 使用说明'
    context['guide_authors'] = 'Dan LI and Dong YANG' if english else 'Dan LI、Dong YANG'
    if english:
        base = '../../singlehtml/' if single else '../../html/'
    else:
        base = '../en/singlehtml/' if single else '../en/html/'
    context['other_language_href'] = base + ('index' if single else pagename) + '.html'

def install_language_buttons(app, exception):
    """Place a single visible language button in Furo's native toolbars."""
    if exception or app.builder.name not in ('html', 'singlehtml'):
        return
    from bs4 import BeautifulSoup
    for path in Path(app.outdir).glob('*.html'):
        context = {}
        language_page_context(app, path.stem, '', context, None)
        soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
        for selector, identifier in [('.mobile-header .header-right', 'language-switch-mobile'), ('.content-icon-container', 'language-switch-desktop')]:
            toolbar = soup.select_one(selector)
            if toolbar is None:
                continue
            for old in toolbar.select('.guide-language-button'):
                old.decompose()
            button = soup.new_tag('a', attrs={
                'href': context['other_language_href'], 'class': 'guide-language-button',
                'id': identifier, 'role': 'button',
                'title': 'Switch to Chinese' if context['guide_english'] else '切换到英文',
                'aria-label': 'Switch to Chinese' if context['guide_english'] else '切换到英文',
                'hreflang': 'zh-CN' if context['guide_english'] else 'en',
            })
            button.string = '中文' if context['guide_english'] else 'English'
            toolbar.insert(0, button)
        path.write_text(str(soup), encoding='utf-8')

html_logo = None
html_favicon = '_static/linkodt_icon.png'
html_theme_options.update({"light_logo": "linkodt_icon.png", "dark_logo": "linkodt_icon.png"})

def validate_english_content(app):
    """Keep localized prose and runnable commands consistent with the English edition."""
    if app.config.language != 'en':
        return
    import re
    from sphinx.errors import SphinxError
    violations = []
    for path in Path(app.srcdir).glob('*.rst'):
        for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if re.search(r'[\u3400-\u9fff]', line):
                violations.append(f'{path.name}:{line_number}')
    if violations:
        raise SphinxError('Chinese text or paths found in English chapters: ' + ', '.join(violations))
