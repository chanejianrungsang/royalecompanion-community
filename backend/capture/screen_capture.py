"""
Screen capture functionality using mss and Windows-specific methods
"""
import mss
import numpy as np
from PIL import Image
import time
import platform
from config.constants import TARGET_FPS, CAPTURE_TIMEOUT

# Windows-specific imports
if platform.system() == "Windows":
    try:
        from .windows_capture import WindowsBackgroundCapture
        WINDOWS_CAPTURE_AVAILABLE = True
    except ImportError:
        WINDOWS_CAPTURE_AVAILABLE = False
else:
    WINDOWS_CAPTURE_AVAILABLE = False

class ScreenCapture:
    def __init__(self):
        self.sct = mss.mss()
        self.capture_region = None
        self.capture_mode = None  # "window", "region", "background_window"
        self.target_window = None
        self.last_capture_time = 0
        
        # Windows background capture
        self.background_capture = None
        if WINDOWS_CAPTURE_AVAILABLE:
            self.background_capture = WindowsBackgroundCapture()
        
        self.background_capture_enabled = False
        
    def set_window_capture(self, window, enable_background=True):
        """Set up capture for a specific window"""
        self.target_window = window
        self.background_capture_enabled = enable_background and WINDOWS_CAPTURE_AVAILABLE
        
        if self.background_capture_enabled and window:
            try:
                # Check if window has a valid title
                window_title = getattr(window, 'title', None)
                if window_title and len(window_title.strip()) > 0:
                    # Try to setup background capture
                    self.background_capture.set_target_window(window_title)
                    self.capture_mode = "background_window"
                    print(f"Background capture enabled for: {window_title}")
                else:
                    print("Window has no title, using standard capture")
                    self.background_capture_enabled = False
                    self.capture_mode = "window"
            except Exception as e:
                print(f"Background capture failed, using standard method: {e}")
                self.background_capture_enabled = False
                self.capture_mode = "window"
        else:
            self.capture_mode = "window"
        
        # Get window bounds for fallback
        try:
            self.capture_region = {
                'left': window.left,
                'top': window.top,
                'width': window.width,
                'height': window.height
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get window bounds: {e}")
            
    def set_region_capture(self, region):
        """Set up capture for a specific region"""
        self.capture_region = {
            'left': region['x'],
            'top': region['y'], 
            'width': region['width'],
            'height': region['height']
        }
        self.capture_mode = "region"
        
    def capture(self):
        """Capture a frame from the current source"""
        if not self.capture_region and self.capture_mode != "background_window":
            return None
            
        # Rate limiting to target FPS
        current_time = time.time()
        if current_time - self.last_capture_time < CAPTURE_TIMEOUT:
            return None
            
        try:
            # Try background capture first (Windows only)
            if self.capture_mode == "background_window" and self.background_capture_enabled:
                frame = self.background_capture.capture_background_window()
                if frame is not None:
                    # Background capture returns RGB, convert to BGR for OpenCV
                    if frame.shape[2] == 3:
                        frame = frame[:, :, ::-1]  # RGB to BGR
                    self.last_capture_time = current_time
                    return frame
                else:
                    # Background capture failed, fall back to standard method
                    print("Background capture failed, falling back to standard capture")
                    self.capture_mode = "window"
            
            # Standard MSS capture
            if self.capture_mode == "window" and self.target_window:
                try:
                    # Update window position if in window mode
                    self.capture_region.update({
                        'left': self.target_window.left,
                        'top': self.target_window.top,
                        'width': self.target_window.width,
                        'height': self.target_window.height
                    })
                except:
                    # Window might be minimized or closed
                    return None
                    
            # Capture screenshot with MSS
            screenshot = self.sct.grab(self.capture_region)
            
            # Convert to numpy array
            frame = np.array(screenshot)
            
            # Convert BGRA to BGR for OpenCV compatibility
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]  # Remove alpha channel
            
            # Convert RGB to BGR for OpenCV (MSS gives RGB, OpenCV expects BGR)
            if frame.shape[2] == 3:
                frame = frame[:, :, ::-1]  # RGB to BGR
                
            self.last_capture_time = current_time
            return frame
            
        except Exception as e:
            print(f"Capture error: {e}")
            return None
            
    def get_capture_info(self):
        """Get information about current capture setup"""
        if self.capture_mode == "background_window":
            bg_info = ""
            if self.background_capture:
                window_info = self.background_capture.get_window_info()
                if window_info:
                    status = "minimized" if window_info['minimized'] else "background"
                    bg_info = f" ({status})"
            return f"Background Window: {self.target_window.title if self.target_window else 'Unknown'}{bg_info}"
        elif self.capture_mode == "window":
            return f"Window: {self.target_window.title if self.target_window else 'Unknown'}"
        elif self.capture_mode == "region":
            r = self.capture_region
            return f"Region: {r['left']},{r['top']} {r['width']}x{r['height']}"
        else:
            return "No capture source set"
            
    def supports_background_capture(self):
        """Check if background capture is supported"""
        return WINDOWS_CAPTURE_AVAILABLE
        
    def is_background_capture_active(self):
        """Check if background capture is currently active"""
        return self.capture_mode == "background_window" and self.background_capture_enabled
