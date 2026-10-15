"""Shared LinkODT typography, interaction cues and application identity."""
from pathlib import Path
import sys
import re
import weakref
from PySide6.QtCore import QObject, QEvent, Qt, QSettings, Signal, QTimer, QSize
from PySide6.QtGui import QFont, QIcon, QDesktopServices, QPalette, QColor, QAction, QActionGroup, QPixmap, QPainter
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication, QAbstractSpinBox, QComboBox, QMessageBox, QWidget, QLabel, QToolButton

APP_NAME = 'LinkODT'
APP_VERSION = '3.2'
FONT_FAMILY = 'Segoe UI'
FONT_FALLBACK = 'Microsoft YaHei UI'
ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
DISABLED_STYLE = '''
QPushButton:disabled, QPushButton:checked:disabled,
QPushButton#Primary:disabled, QPushButton#Delete:disabled,
QPushButton#Save:disabled, QPushButton#SaveMask:disabled,
QPushButton#RemoveCurrent:disabled, QPushButton#SemanticPredict:disabled,
QPushButton#SemanticPredictAll:disabled, QPushButton#SemanticModelTool:disabled,
QPushButton#ProjectionButton:disabled, QPushButton#AutoQC:disabled,
QPushButton#Good:disabled, QPushButton#Bad:disabled {
    color: #788898; background: #EEF2F5; border: 1px solid #CAD4DE;
}
'''

STYLE = '''
QWidget { font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif; font-size: 14px; color: #202020; }
QToolTip { background: #202020; color: white; border: 0; padding: 7px; font-size: 13px; }
QPushButton, QToolButton { font-weight: 600; }
QLabel#ModuleTitle { font-size: 16px; font-weight: 600; background: transparent; }
QLabel#ModuleSubtitle { font-size: 13px; font-weight: 400; color: #616161; background: transparent; }
QPushButton:focus, QToolButton:focus { border: 2px solid #3265AC; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 2px solid #3265AC; }
QPushButton:disabled, QToolButton:disabled { color: #788898; background: #EEF2F5; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 26px; selection-background-color: #2458A6; }
QListWidget::item { padding: 4px 2px; }
QListWidget::item:selected { background: #E1EAF8; color: #202020; }
QHeaderView::section { padding: 6px; font-size: 13px; font-weight: 600; }
QMenu { background: white; padding: 4px; }
QMenu::item { padding: 7px 20px; }
QMenu::item:selected { background: #E1EAF8; }
QProgressBar { min-height: 16px; border: 1px solid #D2D2D2; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: #3265AC; }
'''

class _IntentionalWheelFilter(QObject):
    """Scrolling a panel must not accidentally change an unfocused parameter."""
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and isinstance(obj, (QAbstractSpinBox, QComboBox)):
            if not obj.hasFocus():
                event.ignore()
                return True
        return False

def configure_application(app):
    if getattr(app, '_linkodt_configured', False):
        return
    app._linkodt_configured = True
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setStyle('Fusion')
    app.setFont(QFont(FONT_FAMILY, 10))
    app.setStyleSheet(STYLE)
    icon = ROOT / 'assets/linkodt_app.ico'
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    app._linkodt_wheel_filter = _IntentionalWheelFilter(app)
    app.installEventFilter(app._linkodt_wheel_filter)
    app.theme_manager = ThemeManager(app)

def open_user_guide(parent=None):
    app_dir = Path(sys.executable).parent if getattr(sys, 'frozen', False) else ROOT
    for candidate in (app_dir/'UserGuide/index.html', app_dir.parent/'UserGuide/index.html', ROOT.parent/'使用说明文档/index.html', ROOT/'使用说明文档/index.html'):
        if candidate.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(candidate.resolve())))
            return True
    QMessageBox.information(parent, 'User guide', 'Place the UserGuide folder beside the application to open the local guide.')
    return False

def show_about(parent=None):
    QMessageBox.about(parent, 'About LinkODT',
        '<h2>LinkODT</h2><p>Organelle imaging, analysis and visualization</p>'
        '<p>Version 3.2</p><p>Dan LI &amp; Dong YANG<br>Dayong Jin’s ODT Group<br>'
        'Eastern Institute of Technology, Ningbo<br>宁波东方理工大学 · 细胞器精准诊疗团队</p>')

def _dark_color(value, property_name):
    color = QColor(value)
    if not color.isValid():
        return value
    red, green, blue, alpha = color.getRgbF()
    lightness = .2126 * red + .7152 * green + .0722 * blue
    saturation = max(red, green, blue) - min(red, green, blue)
    background = property_name in ('background', 'background-color', 'alternate-background-color', 'selection-background-color')
    border = property_name.startswith('border')
    if background:
        if saturation < .18 and lightness > .7:
            return '#242424' if lightness > .97 else '#181818'
        if red > .55 and green > .35 and blue < .5:
            return '#86652C'
        if red > .5 and red > green * 1.3:
            return '#9B4350'
        if lightness < .2:
            return '#2458A6' if blue > red * 1.2 else '#181818'
        if green > red * 1.4 and green > blue * 1.1:
            return '#23704E'
        if blue > red * 1.3 or green > red * 1.3:
            return '#2458A6' if lightness < .6 else '#303030'
        return value
    if border:
        return '#454545' if lightness > .45 and saturation < .35 else '#78A9ED'
    if lightness > .9:
        return '#F2F2F2'
    if saturation < .22:
        return '#E8E8E8' if lightness < .35 else '#B8B8B8'
    if red > .55 and red > green * 1.4:
        return '#FF9399'
    if blue > red * 1.25 or green > red * 1.25:
        return '#E8E8E8' if lightness < .2 else '#8DB9F5'
    return '#E8E8E8' if lightness < .4 else value

def dark_stylesheet(stylesheet):
    """Convert interface declarations only; data colours and renderer settings stay intact."""
    declarations = r'\b(background(?:-color)?|alternate-background-color|selection-background-color|selection-color|color|border(?:-color)?)\s*:\s*([^;{}]+)'
    def convert(match):
        name, value = match.groups()
        value = re.sub(r'#[0-9a-fA-F]{3,8}\b', lambda color: _dark_color(color[0], name), value)
        if value.strip() == 'white':
            value = '#242424' if 'background' in name else '#F2F2F2'
        elif value.strip() == 'black':
            value = '#181818' if 'background' in name else '#E8E8E8'
        return f'{name}: {value}'
    return re.sub(declarations, convert, stylesheet)

class ThemeManager(QObject):
    changed = Signal(str)

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.settings = QSettings('LinkODT', 'Appearance')
        self.theme = 'light'
        self._original_palette = QPalette(app.palette())
        self._styles = weakref.WeakKeyDictionary()
        self._applying = False
        app.installEventFilter(self)
        saved = self.settings.value('theme', 'light')
        self.set_theme(saved if saved in ('light', 'dark') else 'light', persist=False)

    def eventFilter(self, obj, event):
        if self._applying:
            return False
        if isinstance(obj, QWidget) and event.type() in (QEvent.Show, QEvent.StyleChange):
            current = obj.styleSheet()
            record = self._styles.get(obj)
            if record is None or current != record[1]:
                self._styles[obj] = (current, current)
            self._apply_widget(obj)
            if event.type() == QEvent.Show and obj.isWindow():
                QTimer.singleShot(0, lambda window=obj: self._apply_titlebar(window))
        if isinstance(obj, QWidget) and obj.isWindow() and event.type() == QEvent.WindowActivate:
            self.settings.sync()
            saved = self.settings.value('theme', self.theme)
            if saved in ('light', 'dark') and saved != self.theme:
                self.set_theme(saved, persist=False)
        return False

    def _apply_widget(self, widget):
        record = self._styles.get(widget)
        if record is None:
            record = (widget.styleSheet(), widget.styleSheet())
        base = record[0]
        rendered = dark_stylesheet(base) if self.theme == 'dark' else base
        self._applying = True
        try:
            if widget.styleSheet() != rendered:
                widget.setStyleSheet(rendered)
            self._styles[widget] = (base, rendered)
            if isinstance(widget, QLabel) and widget.objectName() in ('LogoImage', 'HomeLogoImage', 'logoLabel'):
                pixmap = widget.pixmap()
                if pixmap is not None and not pixmap.isNull():
                    from PySide6.QtGui import QPixmap
                    path = ROOT/'assets'/('linkodt_wordmark_dark.png' if self.theme == 'dark' else 'linkodt_wordmark.png')
                    if path.exists():
                        widget.setPixmap(QPixmap(str(path)).scaled(pixmap.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        finally:
            self._applying = False

    def register_window(self, window):
        for widget in [window, *window.findChildren(QWidget)]:
            self._apply_widget(widget)
        self._apply_titlebar(window)

    def _apply_titlebar(self, window):
        """Match supported Windows title bars to the interface, preserving native controls."""
        if sys.platform != 'win32' or not window.isWindow() or not window.isVisible():
            return
        import ctypes
        try:
            handle = int(window.winId())
            dark = ctypes.c_int(self.theme == 'dark')
            caption = ctypes.c_uint(0x181818 if self.theme == 'dark' else 0xFAFAFA)
            text = ctypes.c_uint(0xF2F2F2 if self.theme == 'dark' else 0x202020)
            for attribute, value in ((20, dark), (35, caption), (36, text)):
                ctypes.windll.dwmapi.DwmSetWindowAttribute(ctypes.c_void_p(handle), attribute, ctypes.byref(value), ctypes.sizeof(value))
        except (OSError, AttributeError):
            pass

    def set_theme(self, theme, persist=True):
        if theme not in ('light', 'dark'):
            raise ValueError('Theme must be light or dark')
        self.theme = theme
        if persist:
            self.settings.setValue('theme', theme)
            self.settings.sync()
        palette = QPalette(self._original_palette)
        roles = ([(QPalette.Window, '#181818'), (QPalette.WindowText, '#E8E8E8'),
                    (QPalette.Base, '#242424'), (QPalette.AlternateBase, '#202020'),
                    (QPalette.Text, '#E8E8E8'), (QPalette.Button, '#242424'),
                    (QPalette.ButtonText, '#E8E8E8'), (QPalette.Highlight, '#2458A6'),
                    (QPalette.HighlightedText, '#FFFFFF'), (QPalette.PlaceholderText, '#B8B8B8')]
                 if theme == 'dark' else
                 [(QPalette.Window, '#FAFAFA'), (QPalette.WindowText, '#202020'),
                    (QPalette.Base, '#FFFFFF'), (QPalette.AlternateBase, '#F3F3F3'),
                    (QPalette.Text, '#202020'), (QPalette.Button, '#FFFFFF'),
                    (QPalette.ButtonText, '#202020'), (QPalette.Highlight, '#2458A6'),
                    (QPalette.HighlightedText, '#FFFFFF'), (QPalette.PlaceholderText, '#616161')])
        for role, color in roles:
            palette.setColor(role, QColor(color))
        self._applying = True
        try:
            self.app.setPalette(palette)
            style = dark_stylesheet(STYLE) if theme == 'dark' else STYLE
            if theme == 'dark':
                tick = (ROOT/'assets/check_white.svg').as_posix()
                style += f'''QCheckBox::indicator {{ width: 14px; height: 14px; background: #242424; border: 1px solid #888888; border-radius: 2px; }}
QCheckBox::indicator:checked {{ background: #2458A6; border-color: #8DB9F5; image: url("{tick}"); }}
QCheckBox::indicator:disabled {{ background: #202020; border-color: #454545; }}'''
            self.app.setStyleSheet(style)
        finally:
            self._applying = False
        for window in self.app.topLevelWidgets():
            self.register_window(window)
        self.changed.emit(theme)

    def toggle(self):
        self.set_theme('dark' if self.theme == 'light' else 'light')

def install_appearance_controls(window, view_menu):
    app = QApplication.instance()
    configure_application(app)
    manager = app.theme_manager
    theme_menu = view_menu.addMenu('Theme')
    group = QActionGroup(window)
    group.setExclusive(True)
    actions = {}
    for name in ('light', 'dark'):
        action = theme_menu.addAction(name.capitalize())
        action.setCheckable(True)
        group.addAction(action)
        action.triggered.connect(lambda checked=False, value=name: manager.set_theme(value))
        actions[name] = action
    button = QToolButton(window)
    button.setObjectName('ThemeToggle')
    button.setToolButtonStyle(Qt.ToolButtonIconOnly)
    button.setFixedSize(34, 30)
    button.setIconSize(QSize(18, 18))
    button.setStyleSheet('QToolButton { background: transparent; border: none; border-radius: 4px; padding: 4px; } QToolButton:hover { background: #E8EDF5; } QToolButton:focus { border: 1px solid #3265AC; }')
    button.setToolTip('Switch between light and dark interface themes')
    button.clicked.connect(manager.toggle)
    window.menuBar().setCornerWidget(button, Qt.TopRightCorner)
    toggle_action = QAction('Toggle theme', window)
    toggle_action.setShortcut('Ctrl+Shift+T')
    toggle_action.triggered.connect(manager.toggle)
    window.addAction(toggle_action)
    def update(theme):
        destination = 'Switch to light theme' if theme == 'dark' else 'Switch to dark theme'
        button.setText('')
        button.setToolTip(destination + ' (Ctrl+Shift+T)')
        button.setAccessibleName(destination)
        from PySide6.QtSvg import QSvgRenderer
        color = '#E8E8E8' if theme == 'dark' else '#303030'
        shape = ('<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>'
                 if theme == 'dark' else '<path d="M20 14A8.5 8.5 0 0 1 10 4a8.5 8.5 0 1 0 10 10Z"/>')
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><g fill="none" stroke="{color}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{shape}</g></svg>'
        pixmap = QPixmap(36,36); pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap); QSvgRenderer(svg.encode()).render(painter); painter.end()
        button.setIcon(QIcon(pixmap))
        for name, action in actions.items():
            action.setChecked(name == theme)
    manager.changed.connect(update)
    update(manager.theme)
    manager.register_window(window)
    window.theme_button = button
    window.theme_actions = actions
    return manager
