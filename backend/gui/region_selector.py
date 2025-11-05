"""
Region selector for capturing specific screen areas
"""
from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QPainter, QPen, QColor

class RegionSelector(QWidget):
    region_selected = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Get screen geometry
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        
        # Selection state
        self.selecting = False
        self.start_pos = None
        self.end_pos = None
        self.selection_rect = None
        
        self.setCursor(Qt.CursorShape.CrossCursor)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.selecting = True
            self.start_pos = event.pos()
            self.end_pos = self.start_pos
            self.selection_rect = QRect()
            
    def mouseMoveEvent(self, event):
        if self.selecting:
            self.end_pos = event.pos()
            self.selection_rect = QRect(self.start_pos, self.end_pos).normalized()
            self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.selecting:
            self.selecting = False
            
            if self.selection_rect and self.selection_rect.width() > 10 and self.selection_rect.height() > 10:
                # Emit the selected region
                region = {
                    'x': self.selection_rect.x(),
                    'y': self.selection_rect.y(),
                    'width': self.selection_rect.width(),
                    'height': self.selection_rect.height()
                }
                self.region_selected.emit(region)
                
            self.close()
            
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            
    def paintEvent(self, event):
        painter = QPainter(self)
        
        # Draw semi-transparent overlay
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        
        if self.selection_rect:
            # Clear the selected area
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(self.selection_rect, QColor(0, 0, 0, 0))
            
            # Draw selection border
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            pen = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self.selection_rect)
