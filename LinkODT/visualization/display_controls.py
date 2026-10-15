"""Non-destructive display-window controls for microscopy volumes."""
import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QColor, QPalette
from PySide6.QtWidgets import QWidget

def display_statistics(data, max_samples=1_000_000):
    """Exact finite bounds; bounded sampling for the histogram and Auto."""
    array=np.asarray(data)
    if not array.size:
        return None
    if array.dtype.kind in 'biu':
        low,high=float(np.min(array)),float(np.max(array))
    else:
        low,high=np.inf,-np.inf
        for start in range(0,array.size,524288):
            block=array.flat[start:start+524288]
            finite=block[np.isfinite(block)]
            if finite.size:
                low=min(low,float(finite.min()));high=max(high,float(finite.max()))
        if not np.isfinite(low):
            return None
    stride=max(1,int(np.ceil(array.size/max_samples)))
    sample=np.asarray(array.flat[::stride],dtype=np.float64)
    sample=sample[np.isfinite(sample)]
    if not sample.size:
        sample=np.asarray([low,high])
    gap=max(abs(low)*1e-9,1e-9) if array.dtype.kind=='f' else 1.0
    upper=high if high>low else low+gap
    auto_low,auto_high=np.percentile(sample,[0.175,99.825])
    if auto_high<=auto_low:
        auto_low,auto_high=low,upper
    histogram,edges=np.histogram(sample,bins=256,range=(low,upper))
    return {'full':(low,upper),'observed':(low,high),
        'auto':(float(auto_low),float(auto_high)), 'histogram':histogram,
        'integer':array.dtype.kind in 'biu','gap':gap}

def effective_window(obj):
    low,high=obj.geometry.display_range or obj.geometry.intensity_range or (0.,1.)
    span=max(high-low,1e-12)
    width=span/max(float(obj.volume_contrast),0.1)
    center=(low+high)/2-np.clip(obj.volume_brightness,-100,100)/100*span/2
    bounds=getattr(obj,'_display_stats',None)
    bounds=bounds['full'] if bounds else obj.geometry.intensity_range or (low,high)
    limits=(min(bounds[0],low),max(bounds[1],high))
    minimum=float(np.clip(center-width/2,*limits))
    maximum=float(np.clip(center+width/2,*limits))
    if maximum<=minimum:
        minimum,maximum=limits
    return minimum,maximum

class DisplayHistogram(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setMaximumHeight(70)
        self.stats=None
        self.window=None
        self.setToolTip('Intensity histogram. Blue markers show the displayed minimum and maximum.')

    def set_data(self,stats,window):
        self.stats,self.window=stats,window
        self.update()

    def paintEvent(self,event):
        painter=QPainter(self)
        area=QRectF(5,5,max(1,self.width()-10),max(1,self.height()-10))
        painter.fillRect(area,self.palette().color(QPalette.Base))
        if not self.stats:
            return
        heights=np.log1p(self.stats['histogram'])
        peak=max(float(heights.max()),1.)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.palette().color(QPalette.Mid))
        for i,height in enumerate(heights):
            bar=height/peak*area.height()
            painter.drawRect(QRectF(area.left()+i/256*area.width(),area.bottom()-bar,max(1,area.width()/256),bar))
        if self.window:
            lo,hi=self.stats['full']
            painter.setPen(QColor('#4C83D4'))
            for value in self.window:
                x=area.left()+float(np.clip((value-lo)/(hi-lo),0,1))*area.width()
                painter.drawLine(int(x),int(area.top()),int(x),int(area.bottom()))
        painter.end()
