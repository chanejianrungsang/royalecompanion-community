"""
Refactored overlay window showing elixir count with improved updating
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
import time

class OverlayWindow(QWidget):
    stop_requested = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.setGeometry(100, 100, 300, 120)
        
        # Track last update for debugging
        self.last_elixir_value = 5.0
        self.last_update_time = time.time()
        self.update_count = 0
        
        self.init_ui()
        
        # Debug timer to force refresh
        self.debug_timer = QTimer()
        self.debug_timer.timeout.connect(self.debug_refresh)
        self.debug_timer.start(1000)  # Update every second for debugging
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Main elixir display
        self.elixir_label = QLabel("Enemy Elixir: 5.0")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.elixir_label.setFont(font)
        self.elixir_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 200);
                color: #00ff00;
                padding: 8px;
                border: 2px solid #00ff00;
                border-radius: 8px;
            }
        """)
        layout.addWidget(self.elixir_label)
        
        # Debug info
        self.debug_label = QLabel("Updates: 0 | Status: Waiting")
        debug_font = QFont()
        debug_font.setPointSize(10)
        self.debug_label.setFont(debug_font)
        self.debug_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 150);
                color: #ffff00;
                padding: 4px;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.debug_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(220, 20, 20, 200);
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 50, 50, 220);
            }
        """)
        
        # Reset button for testing
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_counter)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(20, 20, 220, 200);
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(50, 50, 255, 220);
            }
        """)
        
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)
        
    def update_elixir(self, count, confidence=1.0):
        """Update the elixir display with improved tracking"""
        try:
            self.update_count += 1
            current_time = time.time()
            
            # Force update the display
            if confidence < 0.8:
                uncertainty = "±0.4"
                status_color = "#ffaa00"  # Orange for low confidence
            else:
                uncertainty = "±0.2"
                status_color = "#00ff00"  # Green for good confidence
            
            # Update main display
            elixir_text = f"Enemy Elixir: {count:.1f} ({uncertainty})"
            self.elixir_label.setText(elixir_text)
            
            # Update debug info
            time_diff = current_time - self.last_update_time
            elixir_diff = count - self.last_elixir_value
            
            debug_text = f"Updates: {self.update_count} | Δt: {time_diff:.1f}s | Δelixir: {elixir_diff:.2f}"
            self.debug_label.setText(debug_text)
            
            # Update color based on status
            self.elixir_label.setStyleSheet(f"""
                QLabel {{
                    background-color: rgba(0, 0, 0, 200);
                    color: {status_color};
                    padding: 8px;
                    border: 2px solid {status_color};
                    border-radius: 8px;
                }}
            """)
            
            # Track changes
            self.last_elixir_value = count
            self.last_update_time = current_time
            
            # Force widget repaint
            self.elixir_label.repaint()
            self.debug_label.repaint()
            self.repaint()
            
        except Exception as e:
            print(f"Overlay update error: {e}")
            
    def debug_refresh(self):
        """Debug method to show overlay is alive"""
        current_time = time.time()
        time_since_update = current_time - self.last_update_time
        
        if time_since_update > 3.0:  # No updates for 3+ seconds
            self.debug_label.setText(f"Updates: {self.update_count} | No updates for {time_since_update:.1f}s")
            self.debug_label.setStyleSheet("""
                QLabel {
                    background-color: rgba(220, 0, 0, 150);
                    color: #ffffff;
                    padding: 4px;
                    border-radius: 4px;
                }
            """)
        
    def reset_counter(self):
        """Reset the counter for testing"""
        self.update_count = 0
        self.last_elixir_value = 5.0
        self.last_update_time = time.time()
        self.update_elixir(5.0, 1.0)
        
    def mousePressEvent(self, event):
        """Allow dragging the overlay"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            
    def mouseMoveEvent(self, event):
        """Handle dragging"""
        if hasattr(self, 'drag_position') and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
