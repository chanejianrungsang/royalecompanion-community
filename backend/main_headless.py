"""
Clash Royale Pocket Companion - Headless Mode
Outputs JSON to stdout for Electron frontend consumption
"""
import sys
import json
import time
import logging
import threading
import queue
from datetime import datetime
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from clash_royale_tracker import ElixirTracker

# Configure logging to stderr only (stdout is for JSON)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler('clash_royale_headless.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

_stdin_queue = queue.Queue()
_early_command_buffer = []


def _start_stdin_reader():
    """Spawn a background thread that continuously reads stdin lines."""
    def _reader():
        while True:
            try:
                line = sys.stdin.readline()
            except Exception as exc:  # pragma: no cover - defensive; stdin rarely fails
                logger.warning(f"[IPC] stdin reader error: {exc}")
                break
            if not line:
                break  # EOF
            stripped = line.strip()
            if stripped:
                _stdin_queue.put(stripped)
    thread = threading.Thread(target=_reader, name="stdin-reader", daemon=True)
    thread.start()
    return thread


class HeadlessBackend:
    """Wraps ElixirTracker and emits JSON to stdout"""
    
    def __init__(self, tracker, stdin_queue):
        self.tracker = tracker
        self._stdin_queue = stdin_queue
        self.last_heartbeat = time.time()
        
        # Connect to all signals
        self.tracker.elixir_update.connect(self.on_elixir_update)
        self.tracker.card_cycle_update.connect(self.on_card_cycle)
        self.tracker.timer_update.connect(self.on_timer_update)
        self.tracker.status_update.connect(self.on_status)
        
        # Heartbeat timer (every 500ms)
        self.heartbeat_timer = QTimer()
        self.heartbeat_timer.timeout.connect(self.send_heartbeat)
        self.heartbeat_timer.start(500)
        
        # Stdin command listener (for Electron IPC)
        self.stdin_timer = QTimer()
        self.stdin_timer.timeout.connect(self.check_stdin)
        self.stdin_timer.start(100)  # Check every 100ms

        # Flush any commands that arrived before the backend finished booting
        global _early_command_buffer
        if _early_command_buffer:
            for pending in _early_command_buffer:
                self._stdin_queue.put(pending)
            _early_command_buffer.clear()
        
        logger.info("Headless backend initialized")
        self.emit_json({
            "type": "init",
            "data": {
                "message": "Backend started",
                "timestamp": datetime.now().isoformat()
            }
        })
        
        # Send available windows list on startup
        self.send_windows_list()
    
    def on_elixir_update(self, elixir):
        """Handle elixir updates"""
        self.emit_json({
            "type": "elixir",
            "data": {
                "elixir": round(elixir, 1),
                "timestamp": time.time()
            }
        })
        self.last_heartbeat = time.time()
    
    def on_card_cycle(self, cards_in_hand, upcoming_cards):
        """Handle card cycle updates"""
        # Combine into single 8-card array (4 in hand + 4 upcoming)
        all_cards = cards_in_hand + upcoming_cards
        
        self.emit_json({
            "type": "cards",
            "data": {
                "cards": all_cards,
                "in_hand": cards_in_hand,
                "upcoming": upcoming_cards,
                "timestamp": time.time()
            }
        })
        self.last_heartbeat = time.time()
    
    def on_timer_update(self, timer_str):
        """Handle match timer updates"""
        self.emit_json({
            "type": "timer",
            "data": {
                "timer": timer_str,
                "timestamp": time.time()
            }
        })
        self.last_heartbeat = time.time()
    
    def on_status(self, message):
        """Handle status/event messages"""
        self.emit_json({
            "type": "event",
            "data": {
                "message": message,
                "timestamp": time.time()
            }
        })
        self.last_heartbeat = time.time()
    
    def send_heartbeat(self):
        """Send periodic heartbeat to indicate backend is alive"""
        current_time = time.time()
        if current_time - self.last_heartbeat >= 0.5:
            self.emit_json({
                "type": "heartbeat",
                "data": {
                    "timestamp": current_time,
                    "alive": True
                }
            })
    
    def send_windows_list(self):
        """Send list of available windows to frontend"""
        windows = get_windows_list(refresh=True)
        
        self.emit_json({
            "type": "windows_list",
            "data": {
                "windows": windows,
                "count": len(windows)
            }
        })
    
    def check_stdin(self):
        """Drain any commands that were read by the background stdin thread."""
        try:
            while True:
                try:
                    line = self._stdin_queue.get_nowait()
                except queue.Empty:
                    break
                self._log_raw_command(line)
                try:
                    cmd = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(f"[IPC] Failed to decode JSON command: {exc} | payload={line[:200]}")
                    continue
                self.handle_command(cmd)
        except Exception as exc:  # pragma: no cover - defensive, should not happen
            logger.error(f"[IPC] Unexpected error while handling commands: {exc}")
    
    def handle_command(self, cmd):
        """Handle commands from Electron"""
        cmd_type = cmd.get("command")
        
        if cmd_type == "list_windows":
            self.send_windows_list()
            
        elif cmd_type == "select_window":
            hwnd_input = cmd.get("hwnd")
            resolved_target = hwnd_input
            if isinstance(hwnd_input, str):
                parsed_hwnd = None
                try:
                    parsed_hwnd = int(hwnd_input, 0)
                except (ValueError, TypeError):
                    try:
                        parsed_hwnd = int(hwnd_input)
                    except (ValueError, TypeError):
                        parsed_hwnd = None
                if parsed_hwnd is not None:
                    resolved_target = parsed_hwnd
            success = resolved_target is not None and self.tracker.set_target_window(resolved_target)
            resolved_hwnd = getattr(self.tracker, "hwnd", None)
            if success and resolved_hwnd:
                message = f"Selected window (hwnd={resolved_hwnd})"
                self.on_status(message)
                self.emit_json({
                    "type": "window_selected",
                    "data": {"hwnd": resolved_hwnd, "success": True}
                })
            elif success:
                self.on_status("Selected window")
                self.emit_json({
                    "type": "window_selected",
                    "data": {"hwnd": resolved_target, "success": True}
                })
            else:
                details = {"hwnd": hwnd_input, "success": False, "error": "Failed to set window"}
                self.emit_json({
                    "type": "window_selected",
                    "data": details
                })
                self.on_status("Failed to set target window")
                
        elif cmd_type == "start_tracking":
            if not self.tracker.running:
                self.tracker.start()
                self.on_status("Tracking thread started")
                self.emit_json({
                    "type": "tracking_started",
                    "data": {"success": True}
                })
            else:
                self.emit_json({
                    "type": "tracking_started",
                    "data": {"success": False, "error": "Already tracking"}
                })
                
        elif cmd_type == "stop_tracking":
            if self.tracker.running:
                self.tracker.stop()
                self.on_status("Tracking thread stopped")
                self.emit_json({
                    "type": "tracking_stopped",
                    "data": {"success": True}
                })
            else:
                self.emit_json({
                    "type": "tracking_stopped",
                    "data": {"success": False, "error": "Not tracking"}
                })
        else:
            logger.warning(f"Unknown command: {cmd_type}")
    
    def emit_json(self, data):
        """Output JSON to stdout with flush for real-time streaming"""
        try:
            json_str = json.dumps(data)
            print(json_str, flush=True)
        except Exception as e:
            logger.error(f"Failed to emit JSON: {e}")

    def _log_raw_command(self, payload: str) -> None:
        """Log incoming IPC commands for debugging integrations."""
        logger.info(f"[IPC] raw command: {payload}")

# Global cache for windows list
_cached_windows = None

def get_windows_list(refresh=False):
    """Get windows list with thumbnails (cached unless refresh requested)"""
    global _cached_windows
    
    if refresh or _cached_windows is None:
        try:
            import win32gui
            import win32ui
            import win32con
            from PIL import Image
            import io
            import base64
            
            windows = []
            seen_titles = set()  # Track seen titles to avoid duplicates
            
            def enum_windows_callback(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title:
                        # Filter out generic/system windows
                        generic_titles = ['settings', 'program manager', 'task switching', 'default ime', 'msctfime ui']
                        if title.lower() in generic_titles:
                            return True
                        
                        # Skip duplicate titles (keep first one found)
                        if title in seen_titles:
                            return True
                        
                        # Filter out windows not in taskbar
                        # Check if window has WS_EX_TOOLWINDOW or WS_EX_APPWINDOW extended styles
                        ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                        
                        # Skip tool windows (they don't appear in taskbar)
                        if ex_style & win32con.WS_EX_TOOLWINDOW:
                            return True
                        
                        # Skip windows without WS_EX_APPWINDOW that have an owner
                        if not (ex_style & win32con.WS_EX_APPWINDOW):
                            if win32gui.GetWindow(hwnd, win32con.GW_OWNER):
                                return True
                        
                        # Capture window thumbnail
                        thumbnail = None
                        try:
                            # Get window dimensions
                            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                            width = right - left
                            height = bottom - top
                            
                            # Skip if window is too small
                            if width < 50 or height < 50:
                                return True
                            
                            # Try PrintWindow first (better for some windows)
                            try:
                                hwndDC = win32gui.GetWindowDC(hwnd)
                                mfcDC = win32ui.CreateDCFromHandle(hwndDC)
                                saveDC = mfcDC.CreateCompatibleDC()
                                
                                # Create bitmap
                                saveBitMap = win32ui.CreateBitmap()
                                saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
                                saveDC.SelectObject(saveBitMap)
                                
                                # Try PrintWindow API (works better for some apps)
                                import ctypes
                                result = ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
                                
                                # If PrintWindow failed, try BitBlt
                                if result == 0:
                                    result = saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
                                
                                # Convert to PIL Image
                                bmpinfo = saveBitMap.GetInfo()
                                bmpstr = saveBitMap.GetBitmapBits(True)
                                img = Image.frombuffer(
                                    'RGB',
                                    (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                                    bmpstr, 'raw', 'BGRX', 0, 1
                                )
                                
                                # Check if image is all black (failed capture)
                                import numpy as np
                                img_array = np.array(img)
                                if img_array.max() < 10:  # Nearly all black
                                    raise Exception("Captured image is black")
                                
                                # Resize to thumbnail (300x200)
                                img.thumbnail((300, 200), Image.Resampling.LANCZOS)
                                
                                # Convert to base64
                                buffered = io.BytesIO()
                                img.save(buffered, format="PNG")
                                img_str = base64.b64encode(buffered.getvalue()).decode()
                                thumbnail = f"data:image/png;base64,{img_str}"
                                
                                # Cleanup
                                win32gui.DeleteObject(saveBitMap.GetHandle())
                                saveDC.DeleteDC()
                                mfcDC.DeleteDC()
                                win32gui.ReleaseDC(hwnd, hwndDC)
                                
                            except Exception as e:
                                logger.debug(f"Failed to capture thumbnail for {title}: {e}")
                                thumbnail = None
                            
                        except Exception as e:
                            logger.debug(f"Failed to capture thumbnail for {title}: {e}")
                            # Use None for thumbnail on error
                            thumbnail = None
                        
                        # Only add windows that have valid thumbnails OR are clearly important apps
                        # (to avoid cluttering the list with background processes)
                        if thumbnail or any(keyword in title.lower() for keyword in ['clash', 'royal', 'game', 'bluestacks', 'memu', 'nox']):
                            seen_titles.add(title)  # Mark this title as seen
                            windows.append({
                                "hwnd": hwnd, 
                                "title": title,
                                "thumbnail": thumbnail
                            })
                return True
            
            win32gui.EnumWindows(enum_windows_callback, None)
            windows.sort(key=lambda w: w["title"].lower())
            
            # Cache the result
            _cached_windows = windows
        except Exception as e:
            logger.error(f"Failed to get windows list: {e}")
            _cached_windows = []

    return _cached_windows if _cached_windows is not None else []

def send_initial_windows_list():
    """Send windows list immediately before loading heavy models"""
    windows = get_windows_list(refresh=True)
    
    # Send to stdout
    data = {
        "type": "windows_list",
        "data": {
            "windows": windows,
            "count": len(windows)
        }
    }
    print(json.dumps(data), flush=True)
    logger.info(f"Sent initial windows list: {len(windows)} windows")

def handle_early_commands():
    """Handle commands during initialization (before HeadlessBackend is ready)"""
    global _early_command_buffer

    # Drain anything that arrived
    new_data = False
    while True:
        try:
            line = _stdin_queue.get_nowait()
        except queue.Empty:
            break
        _early_command_buffer.append(line)
        new_data = True

    if not new_data:
        return

    remaining = []
    for raw in _early_command_buffer:
        try:
            cmd = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(f"[IPC] Early command JSON error: {exc} | payload={raw[:200]}")
            continue
        cmd_type = cmd.get("command")
        if cmd_type == "list_windows":
            windows = get_windows_list(refresh=True)
            data = {
                "type": "windows_list",
                "data": {
                    "windows": windows,
                    "count": len(windows)
                }
            }
            print(json.dumps(data), flush=True)
        else:
            remaining.append(raw)

    _early_command_buffer = remaining

def main():
    try:
        logger.info("=" * 80)
        logger.info("CLASH ROYALE POCKET COMPANION (HEADLESS MODE)")
        logger.info(f"Startup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("Clash Royale Companion Headless")

        # Start stdin reader thread before heavy initialization
        _start_stdin_reader()
        
        # Send windows list immediately (before loading heavy models)
        send_initial_windows_list()
        
        # Setup early stdin command handler (respond to list_windows during loading)
        stdin_timer = QTimer()
        stdin_timer.timeout.connect(lambda: handle_early_commands())
        stdin_timer.start(100)  # Check every 100ms
        
        # Create tracker thread (this loads ResNet - takes 5-10 seconds)
        logger.info("Loading neural network models...")
        tracker = ElixirTracker()
        
        # Stop early command handler (HeadlessBackend will take over)
        stdin_timer.stop()
        
        # Wrap with headless backend
        backend = HeadlessBackend(tracker, _stdin_queue)
        
        # Start tracker
        logger.info("Starting tracker thread...")
        tracker.start()
        
        # Run Qt event loop
        logger.info("Entering event loop...")
        result = app.exec()
        
        logger.info(f"Event loop exited with code: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Fatal error in headless backend: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
