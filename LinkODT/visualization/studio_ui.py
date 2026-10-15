"""LinkODT UI adapter for Dan LI's Studio release.

The original release supplied compiled Python 3.11 modules, retained alongside
this file. Keep the rendering, TIFF interpretation and movie logic unchanged.
"""
import sys
from pathlib import Path
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QPushButton,
    QToolButton, QMenu, QTabWidget, QScrollArea, QGroupBox, QFormLayout,
    QDoubleSpinBox, QSlider, QLabel, QWidget, QSizePolicy)
from display_controls import DisplayHistogram, display_statistics, effective_window
if sys.version_info[:2] != (3,11):
    raise RuntimeError('This Studio release requires Python 3.11. Use the LinkODT Windows package or the project Python 3.11 environment.')
import _studio_core as core

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ui_theme import configure_application, install_appearance_controls, open_user_guide, show_about

APP_NAME = 'LinkODT Visualization Studio'
SceneObject = core.SceneObject
ROI3D = core.ROI3D

class VisualizationWindow(core.VisualizationWindow):
    def __init__(self):
        configure_application(QApplication.instance())
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self._build_menu_bar()
        bounds = (self.screen() or QApplication.primaryScreen()).availableGeometry()
        self.setMinimumSize(min(980, bounds.width()-24), min(620, bounds.height()-60))
        self.resize(min(1440, bounds.width()-24), min(900, bounds.height()-60))
        QApplication.instance().theme_manager.register_window(self)

    def _apply_style(self):
        # All panels inherit the same stylesheet and font as the main program.
        self.setStyleSheet('')

    def _build_left_panel(self):
        panel = super()._build_left_panel()
        self.btn_add_objects.setText('Add files…')
        self.btn_add_objects.setToolTip('Open TIFF images, OBJ meshes, or TXT / CSV point clouds')
        return panel

    def _build_appearance_tab(self):
        panel=super()._build_appearance_tab()
        self.display_group=QGroupBox('Display range')
        form=QFormLayout(self.display_group)
        self.display_histogram=DisplayHistogram()
        form.addRow(self.display_histogram)
        for name in ('minimum','maximum'):
            row=QWidget()
            layout=QHBoxLayout(row)
            layout.setContentsMargins(0,0,0,0)
            slider=QSlider(Qt.Horizontal)
            slider.setRange(0,1000)
            spin=QDoubleSpinBox()
            spin.setRange(-1e30,1e30)
            spin.setKeyboardTracking(False)
            spin.setMinimumWidth(90)
            spin.setToolTip('Raw intensity at the '+name+' end of the display range')
            layout.addWidget(slider,1);layout.addWidget(spin)
            setattr(self,'display_'+name+'_slider',slider)
            setattr(self,'display_'+name+'_spin',spin)
            slider.valueChanged.connect(lambda value,key=name:self._display_slider_changed(key,value))
            spin.valueChanged.connect(lambda value,key=name:self._display_spin_changed(key,value))
            form.addRow(name.capitalize(),row)
        buttons=QWidget();layout=QHBoxLayout(buttons);layout.setContentsMargins(0,0,0,0)
        self.btn_display_auto=QPushButton('Auto')
        self.btn_display_auto.setToolTip('Optimize the display window from the selected channel’s intensity distribution')
        self.btn_display_reset=QPushButton('Reset')
        self.btn_display_reset.setToolTip('Restore the full intensity range, brightness 0, contrast 1 and background cutoff 0')
        layout.addWidget(self.btn_display_auto);layout.addWidget(self.btn_display_reset)
        self.btn_display_auto.clicked.connect(self.auto_display_contrast)
        self.btn_display_reset.clicked.connect(self.reset_display_contrast)
        form.addRow(buttons)
        self.display_hint=QLabel('Display only. Original pixel values remain unchanged.')
        self.display_hint.setWordWrap(True)
        self.display_hint.setObjectName('infoLabel')
        form.addRow(self.display_hint)
        layout=panel.layout()
        layout.insertWidget(layout.indexOf(self.volume_group),self.display_group)
        for child_form in panel.findChildren(QFormLayout):
            child_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
            child_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.display_hint.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        self.display_group.setEnabled(False)
        self.display_group.setVisible(False)
        return panel

    def _display_object(self):
        obj=self.current_object()
        return obj if obj and obj.geometry.kind=='volume' else None

    def _display_statistics(self,obj):
        if not hasattr(obj,'_display_stats'):
            data=obj.geometry.time_numpy_ref
            if data is None:
                data=obj.geometry.numpy_ref
            obj._display_stats=display_statistics(data) if data is not None else None
        return obj._display_stats

    def _selected_object_changed(self,*args):
        super()._selected_object_changed(*args)
        if hasattr(self,'display_group'):
            self._sync_display_controls()

    def _volume_controls_changed(self,*args):
        super()._volume_controls_changed(*args)
        if hasattr(self,'display_group') and not self._updating_ui:
            self._sync_display_controls()

    def _sync_display_controls(self):
        obj=self._display_object()
        stats=self._display_statistics(obj) if obj else None
        self.display_group.setVisible(obj is not None)
        self.display_group.setEnabled(stats is not None)
        if stats is None:
            self.display_histogram.set_data(None,None)
            return
        window=effective_window(obj)
        lo,hi=stats['full']
        for name,value in zip(('minimum','maximum'),window):
            spin=getattr(self,'display_'+name+'_spin')
            slider=getattr(self,'display_'+name+'_slider')
            with QSignalBlocker(spin), QSignalBlocker(slider):
                spin.setDecimals(0 if stats['integer'] else 12)
                spin.setSingleStep(1. if stats['integer'] else max((hi-lo)/1000,stats['gap']))
                spin.setValue(value)
                slider.setValue(round(max(0,min(1,(value-lo)/(hi-lo)))*1000))
        self.display_histogram.set_data(stats,window)
        self.display_hint.setText('Display only · all Z slices'+(' and time points' if obj.geometry.frame_count>1 else '')+'. Original pixels are unchanged.')

    def _display_spin_changed(self,name,value):
        obj=self._display_object()
        if obj is None or self._updating_ui:
            return
        lo=self.display_minimum_spin.value();hi=self.display_maximum_spin.value()
        stats=self._display_statistics(obj)
        if hi<=lo:
            if name=='minimum':hi=lo+stats['gap']
            else:lo=hi-stats['gap']
        self._set_display_window(obj,(lo,hi))

    def _display_slider_changed(self,name,value):
        obj=self._display_object()
        if obj:
            stats=self._display_statistics(obj)
            lo,hi=stats['full']
            getattr(self,'display_'+name+'_spin').setValue(lo+(hi-lo)*value/1000)

    def _set_display_window(self,obj,window):
        obj.geometry.display_range=tuple(window)
        obj.volume_brightness=0.;obj.volume_contrast=1.;obj.volume_threshold_percent=0.
        for widget,value in [(self.volume_brightness_spin,0.),(self.volume_contrast_spin,1.),(self.volume_threshold_spin,0.)]:
            with QSignalBlocker(widget):widget.setValue(value)
        self._update_volume_transfer(obj)
        self.render_window.Render()
        self._sync_display_controls()

    def _update_volume_transfer(self,obj):
        stats=getattr(obj,'_display_stats',None) if obj else None
        if stats:
            original=obj.geometry.intensity_range
            bounds=stats['full'];window=obj.geometry.display_range or bounds
            obj.geometry.intensity_range=(min(bounds[0],window[0]),max(bounds[1],window[1]))
            try:super()._update_volume_transfer(obj)
            finally:obj.geometry.intensity_range=original
        else:
            super()._update_volume_transfer(obj)

    def auto_display_contrast(self):
        obj=self._display_object()
        if obj and (stats:=self._display_statistics(obj)):
            self._set_display_window(obj,stats['auto'])
            self.statusBar().showMessage('Auto display range applied to '+obj.name)

    def reset_display_contrast(self):
        obj=self._display_object()
        if obj and (stats:=self._display_statistics(obj)):
            self._set_display_window(obj,stats['full'])
            self.statusBar().showMessage('Full intensity range restored for '+obj.name)

    def _build_right_tabs(self):
        tabs = QTabWidget()
        tabs.setMinimumWidth(380)
        tabs.setMaximumWidth(520)
        for title, build in [('Appearance', self._build_appearance_tab),
                ('Scene', self._build_scene_tab), ('ROI', self._build_roi_tab),
                ('Director', self._build_director_tab)]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(build())
            tabs.addTab(scroll, title)
        return tabs

    def _build_camera_toolbar(self):
        frame = QFrame()
        frame.setObjectName('CameraToolbar')
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(4,4,4,4)
        layout.setSpacing(6)
        def button(text, tip):
            result = QPushButton(text)
            result.setToolTip(tip)
            result.setMinimumWidth(result.fontMetrics().horizontalAdvance(text)+24)
            layout.addWidget(result)
            return result
        def menu(text):
            b = QToolButton()
            b.setText(text)
            b.setPopupMode(QToolButton.InstantPopup)
            b.setMinimumWidth(b.fontMetrics().horizontalAdvance(text)+36)
            b.setMinimumHeight(32)
            result = QMenu(b)
            b.setMenu(result)
            layout.addWidget(b)
            return result
        self.btn_back = button('← Back', 'Return to the main LinkODT window')
        self.btn_back.clicked.connect(self.close)
        self.btn_reset_camera = button('Reset view', 'Restore the initial fitted camera view')
        self.btn_flat = button('Flat', 'Show the microscopy XY plane flat')
        self.camera_views_menu = menu('Views')
        for axis in 'xyz':
            setattr(self, 'btn_view_'+axis, self.camera_views_menu.addAction(axis.upper()+' axis'))
        self.camera_rotate_menu = menu('Rotate')
        self.btn_rotate_90 = self.camera_rotate_menu.addAction('Rotate 90°')
        self.btn_rotate_180 = self.camera_rotate_menu.addAction('Rotate 180°')
        # Donor signal wiring uses .clicked; QAction uses .triggered.
        for item in [self.btn_view_x,self.btn_view_y,self.btn_view_z,self.btn_rotate_90,self.btn_rotate_180]:
            item.clicked = item.triggered
        self.btn_zoom_in = button('Zoom +', 'Move closer')
        self.btn_zoom_out = button('Zoom −', 'Move farther')
        pan = menu('Pan')
        offsets = {'left':(-0.1,0),'right':(0.1,0),'up':(0,0.1),'down':(0,-0.1)}
        for direction in offsets:
            item = pan.addAction('Pan '+direction)
            item.triggered.connect(lambda checked=False, value=direction:self.camera_pan(*offsets[value]))
        focus = button('Focus selected', 'Focus the selected object')
        focus.clicked.connect(self.focus_selected)
        layout.addStretch()
        from PySide6.QtWidgets import QLabel
        self.camera_mode_label = QLabel('Mouse: left rotate · right pan · wheel zoom')
        self.statusBar().addPermanentWidget(self.camera_mode_label)
        return frame

    def _build_menu_bar(self):
        def action(menu, text, callback, shortcut=None):
            item = menu.addAction(text)
            item.triggered.connect(callback)
            if shortcut:
                item.setShortcut(QKeySequence(shortcut))
            return item
        file = self.menuBar().addMenu('File')
        action(file, 'Open images or objects…', self.open_objects, 'Ctrl+O')
        file.addSeparator()
        action(file, 'Export image…', self.export_png)
        action(file, 'Export movie…', self.export_movie)
        action(file, 'Load camera path…', self.load_camera_path)
        action(file, 'Save camera path…', self.save_camera_path)
        file.addSeparator()
        action(file, 'Close visualization', self.close, 'Ctrl+W')
        view = self.menuBar().addMenu('View')
        action(view, 'Fit scene', self.reset_camera, 'Ctrl+0')
        action(view, 'Reset view', self.restore_home_view)
        action(view, 'Flat view', self.set_camera_flat)
        view.addMenu(self.camera_views_menu)
        for text, widget in [('Object list',self.left_panel),('Controls',self.right_tabs)]:
            item = view.addAction(text)
            item.setCheckable(True)
            item.setChecked(True)
            item.toggled.connect(widget.setVisible)
        install_appearance_controls(self,view)
        objects = self.menuBar().addMenu('Objects')
        action(objects,'Add images or objects…',self.open_objects)
        action(objects,'Remove selected',self.remove_selected_object)
        action(objects,'Clear scene',self.clear_scene)
        camera = self.menuBar().addMenu('Camera')
        camera.addMenu(self.camera_rotate_menu)
        action(camera,'Focus selected',self.focus_selected)
        action(camera,'Capture keyframe',self.add_keyframe)
        action(camera,'Local camera assistant…',self.open_ai_director)
        help = self.menuBar().addMenu('Help')
        action(help,'User guide',lambda:open_user_guide(self),'F1')
        action(help,'About LinkODT',lambda:show_about(self))

    def open_ai_director(self):
        from ai_dialog import AIDirectorDialog
        AIDirectorDialog(self).exec()

    def apply_ai_plan(self, plan, start_time=0.0, replace_timeline=True):
        return self.apply_director_plan(plan,start_time,replace_timeline)

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    configure_application(app)
    window = VisualizationWindow()
    window.show()
    return app.exec()

if __name__ == '__main__':
    main()
