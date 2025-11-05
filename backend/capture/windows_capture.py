"""
Windows-specific capture methods for background window capture
"""
import ctypes
from ctypes import wintypes
import win32gui
import win32ui
import win32con
import numpy as np
import cv2
from PIL import Image

class WindowsBackgroundCapture:
    def __init__(self):
        self.hwnd = None
        self.window_dc = None
        self.compatible_dc = None
        self.bitmap = None
        
    def set_target_window(self, window_title_or_hwnd):
        """Set target window by title or handle"""
        if isinstance(window_title_or_hwnd, str):
            # Find window by title
            self.hwnd = win32gui.FindWindow(None, window_title_or_hwnd)
            if not self.hwnd:
                # Try partial match if exact match fails
                def enum_windows_callback(hwnd, windows):
                    if win32gui.IsWindowVisible(hwnd):
                        window_text = win32gui.GetWindowText(hwnd)
                        if window_title_or_hwnd.lower() in window_text.lower():
                            windows.append(hwnd)
                    return True
                
                windows = []
                win32gui.EnumWindows(enum_windows_callback, windows)
                if windows:
                    self.hwnd = windows[0]  # Take first match
        else:
            self.hwnd = window_title_or_hwnd
            
        if not self.hwnd:
            raise ValueError(f"Window not found: {window_title_or_hwnd}")
            
        # Get window rect
        try:
            rect = win32gui.GetWindowRect(self.hwnd)
            self.width = rect[2] - rect[0]
            self.height = rect[3] - rect[1]
            
            if self.width <= 0 or self.height <= 0:
                raise ValueError(f"Invalid window dimensions: {self.width}x{self.height}")
                
        except Exception as e:
            raise ValueError(f"Failed to get window dimensions: {e}")
        
        return True
        
    def capture_background_window(self):
        """Capture window even if it's in background/minimized"""
        try:
            # Get window device context
            hwndDC = win32gui.GetWindowDC(self.hwnd)
            
            # Create compatible DC
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # Create bitmap
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, self.width, self.height)
            
            # Select bitmap into DC
            saveDC.SelectObject(saveBitMap)
            
            # Print window to bitmap (THIS IS THE KEY - captures background windows)
            result = ctypes.windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 3)
            
            if result == 0:
                # PrintWindow failed, try BitBlt as fallback
                saveDC.BitBlt((0, 0), (self.width, self.height), mfcDC, (0, 0), win32con.SRCCOPY)
            
            # Convert to numpy array
            bmp_info = saveBitMap.GetInfo()
            bmp_str = saveBitMap.GetBitmapBits(True)
            
            # Convert to PIL Image
            img = Image.frombuffer(
                'RGB',
                (bmp_info['bmWidth'], bmp_info['bmHeight']),
                bmp_str, 'raw', 'BGRX', 0, 1
            )
            
            # Convert to numpy array and switch to OpenCV BGR
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Cleanup
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwndDC)
            
            return frame
            
        except Exception as e:
            print(f"Background capture error: {e}")
            return None
            
    def is_window_minimized(self):
        """Check if target window is minimized"""
        if not self.hwnd:
            return False
        return win32gui.IsIconic(self.hwnd)
        
    def restore_window(self):
        """Restore minimized window (optional helper)"""
        if self.hwnd:
            win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            
    def get_window_info(self):
        """Get information about the target window"""
        if not self.hwnd:
            return None
            
        try:
            rect = win32gui.GetWindowRect(self.hwnd)
            title = win32gui.GetWindowText(self.hwnd)
            is_visible = win32gui.IsWindowVisible(self.hwnd)
            is_minimized = win32gui.IsIconic(self.hwnd)
            
            return {
                'hwnd': self.hwnd,
                'title': title,
                'rect': rect,
                'width': rect[2] - rect[0],
                'height': rect[3] - rect[1],
                'visible': is_visible,
                'minimized': is_minimized
            }
        except:
            return None
