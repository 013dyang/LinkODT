"""Generate LinkODT wordmarks using the accepted original circular artwork."""
from pathlib import Path
import base64, json
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

assets = Path(__file__).resolve().parents[1] / 'assets'
app = QApplication.instance() or QApplication([])
icon = base64.b64encode((assets/'linkodt_icon.png').read_bytes()).decode('ascii')
for theme, color in [('light','#172B4D'),('dark','#F1F4FA')]:
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="900" height="270" viewBox="0 0 900 270">
    <image x="0" y="10" width="250" height="250" xlink:href="data:image/png;base64,{icon}"/>
    <text x="275" y="181" font-family="Segoe UI" font-size="116" font-weight="600" fill="{color}">Link<tspan fill="#6654CE">ODT</tspan></text></svg>'''
    image = QImage(900,270,QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    QSvgRenderer(svg.encode()).render(painter)
    painter.end()
    filename = 'linkodt_wordmark'+('_dark' if theme=='dark' else '')+'.png'
    image.save(str(assets/filename))
(assets/'brand_provenance.json').write_text(json.dumps({
    'product':'LinkODT','artwork':'Original approved circular organelle artwork retained',
    'wordmark':'LinkODT; Segoe UI semibold; theme-specific foreground',
    'generator':'tools/render_brand.py'},indent=2),encoding='utf8')
print('LinkODT light and dark wordmarks generated.')
