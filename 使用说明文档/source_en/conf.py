from pathlib import Path
_shared = Path(__file__).resolve().parent.parent / 'source' / 'conf.py'
exec(compile(_shared.read_text(encoding='utf-8'), str(_shared), 'exec'))
project = 'LinkODT User Guide'
html_title = project
language = 'en'
templates_path = [str(_shared.parent / '_templates')]

html_logo = None
html_theme_options.update({"light_logo": "linkodt_icon.png", "dark_logo": "linkodt_icon.png"})
