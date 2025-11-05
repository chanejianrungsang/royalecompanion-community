"""
Main window for Clash Royale Pocket Companion
"""
import sys
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QComboBox, QGroupBox, QMessageBox, QCheckBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QImage
import pygetwindow as gw
from .region_selector import RegionSelector
from .overlay_window import OverlayWindow
from capture.screen_capture import ScreenCapture
from cv.state_machine import StateMachine

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clash Royale Pocket Companion")
        self.setGeometry(100, 100, 400, 300)
        
        # Initialize components
        self.screen_capture = ScreenCapture()
        self.state_machine = StateMachine()
        self.overlay_window = None
        self.region_selector = None
        
        # Capture settings
        self.capture_mode = "window"  # "window" or "region"
        self.selected_window = None
        self.selected_region = None
        
        # Timer for main loop
        self.main_timer = QTimer()
        self.main_timer.timeout.connect(self.main_loop)
        
        self.init_ui()
        self.refresh_windows()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Source selection group
        source_group = QGroupBox("Capture Source")
        source_layout = QVBoxLayout(source_group)
        
        # Window selection
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_windows)
        window_layout.addWidget(self.window_combo)
        window_layout.addWidget(self.refresh_btn)
        source_layout.addLayout(window_layout)
        
        # Window selection change handler
        self.window_combo.currentTextChanged.connect(self.on_window_selected)
        
        # Background capture option (Windows only)
        if self.screen_capture.supports_background_capture():
            self.background_checkbox = QCheckBox("Enable background capture (minimized/hidden windows)")
            self.background_checkbox.setChecked(True)
            self.background_checkbox.setToolTip("Allows capturing windows even when minimized or covered")
            source_layout.addWidget(self.background_checkbox)
        else:
            self.background_checkbox = None
        
        # Region selection
        region_layout = QHBoxLayout()
        self.region_btn = QPushButton("Select Region")
        self.region_btn.clicked.connect(self.select_region)
        self.region_status = QLabel("No region selected")
        region_layout.addWidget(self.region_btn)
        region_layout.addWidget(self.region_status)
        source_layout.addLayout(region_layout)
        
        layout.addWidget(source_group)
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Tracking")
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn = QPushButton("Stop Tracking")
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.stop_btn.setEnabled(False)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        layout.addLayout(control_layout)
        
        # Status
        self.status_label = QLabel("Ready - Select a source and click Start")
        layout.addWidget(self.status_label)
        
        # Neural detection status
        self.neural_status_label = QLabel("🧠 Neural Detection Ready")
        self.neural_status_label.setStyleSheet("color: #006600; font-size: 10px;")
        layout.addWidget(self.neural_status_label)
        
        # Preview area
        self.preview_label = QLabel("No preview - select source and click Preview")
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview_label)
        
        # Preview controls
        preview_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Start Preview")
        self.preview_btn.clicked.connect(self.toggle_preview)
        self.preview_btn.setEnabled(False)
        preview_layout.addWidget(self.preview_btn)
        layout.addLayout(preview_layout)
        
        # Preview timer
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        
    def refresh_windows(self):
        """Refresh the list of available windows"""
        self.window_combo.clear()
        try:
            windows = gw.getAllWindows()
            for window in windows:
                if window.title and len(window.title.strip()) > 0:
                    self.window_combo.addItem(window.title, window)
        except Exception as e:
            self.status_label.setText(f"Error refreshing windows: {e}")
            
    def on_window_selected(self):
        """Handle window selection change"""
        if self.window_combo.currentData():
            self.capture_mode = "window"
            self.preview_btn.setEnabled(True)
            self.status_label.setText("Window selected - ready to preview or start tracking")
        else:
            self.preview_btn.setEnabled(False)
    
    def select_region(self):
        """Open region selector"""
        if self.region_selector:
            self.region_selector.close()
            
        self.region_selector = RegionSelector()
        self.region_selector.region_selected.connect(self.on_region_selected)
        self.region_selector.show()
        
    def on_region_selected(self, region):
        """Handle region selection"""
        self.selected_region = region
        self.capture_mode = "region"
        self.region_status.setText(f"Region: {region['x']},{region['y']} {region['width']}x{region['height']}")
        self.preview_btn.setEnabled(True)
        self.status_label.setText("Region selected - ready to preview or start tracking")
        
    def start_tracking(self):
        """Start the elixir tracking"""
        # Determine capture source
        if self.capture_mode == "window":
            if self.window_combo.currentData():
                self.selected_window = self.window_combo.currentData()
            else:
                QMessageBox.warning(self, "Warning", "Please select a window")
                return
        elif self.capture_mode == "region":
            if not self.selected_region:
                QMessageBox.warning(self, "Warning", "Please select a region")
                return
        else:
            QMessageBox.warning(self, "Warning", "Please select a capture source")
            return
                
        # Setup capture
        try:
            if self.capture_mode == "window":
                enable_background = self.background_checkbox.isChecked() if self.background_checkbox else False
                self.screen_capture.set_window_capture(self.selected_window, enable_background)
            else:
                self.screen_capture.set_region_capture(self.selected_region)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to setup capture: {e}")
            return
            
        # Show overlay
        if not self.overlay_window:
            self.overlay_window = OverlayWindow()
            self.overlay_window.stop_requested.connect(self.stop_tracking)
        self.overlay_window.show()
        
        # Start main loop
        self.main_timer.start(16)  # ~60 FPS
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        capture_info = self.screen_capture.get_capture_info()
        self.status_label.setText(f"Tracking active - {capture_info}")
        
    def stop_tracking(self):
        """Stop the elixir tracking"""
        self.main_timer.stop()
        
        if self.overlay_window:
            self.overlay_window.hide()
            
        # Reset state machine
        self.state_machine.reset()
        
        # Update UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Tracking stopped")
        
    def toggle_preview(self):
        """Toggle live preview"""
        if self.preview_timer.isActive():
            self.preview_timer.stop()
            self.preview_btn.setText("Start Preview")
            self.preview_label.setText("Preview stopped")
            return
            
        # Validate capture source
        if self.capture_mode == "window":
            if not self.window_combo.currentData():
                self.status_label.setText("Please select a window")
                return
            self.selected_window = self.window_combo.currentData()
        elif self.capture_mode == "region":
            if not self.selected_region:
                self.status_label.setText("Please select a region")
                return
        else:
            self.status_label.setText("Please select a capture source")
            return
            
        # Setup capture
        try:
            if self.capture_mode == "window":
                enable_background = self.background_checkbox.isChecked() if self.background_checkbox else False
                self.screen_capture.set_window_capture(self.selected_window, enable_background)
            else:
                self.screen_capture.set_region_capture(self.selected_region)
        except Exception as e:
            self.status_label.setText(f"Preview setup failed: {e}")
            return
            
        # Start preview
        self.preview_timer.start(100)  # 10 FPS for preview
        self.preview_btn.setText("Stop Preview")
        self.status_label.setText("Preview active")
            
    def update_preview(self):
        """Update preview display"""
        try:
            frame = self.screen_capture.capture()
            if frame is not None:
                # Convert to QPixmap for display
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                
                # Convert numpy array to bytes for QImage
                frame_bytes = frame.tobytes()
                q_image = QImage(frame_bytes, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                
                # Scale for preview
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(
                    self.preview_label.size(), 
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.preview_label.setPixmap(scaled_pixmap)
        except Exception as e:
            self.status_label.setText(f"Preview error: {e}")
        
    def main_loop(self):
        """Main processing loop"""
        try:
            # Capture frame
            frame = self.screen_capture.capture()
            if frame is None:
                return
                
            # Process through state machine
            self.state_machine.process_frame(frame)
            
            # Update overlay
            if self.overlay_window:
                elixir_count = self.state_machine.get_enemy_elixir()
                confidence = self.state_machine.get_confidence()
                self.overlay_window.update_elixir(elixir_count, confidence)
                
        except Exception as e:
            self.status_label.setText(f"Processing error: {e}")
            print(f"Main loop error: {e}")
            
    def closeEvent(self, event):
        """Handle window close"""
        self.stop_tracking()
        self.preview_timer.stop()
        if self.overlay_window:
            self.overlay_window.close()
        if self.region_selector:
            self.region_selector.close()
        event.accept()
