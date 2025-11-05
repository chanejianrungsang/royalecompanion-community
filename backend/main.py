"""
Clash Royale Pocket Companion - Main Entry Point
"""
import sys
import logging
import traceback
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt
from clash_royale_tracker import ClashRoyaleTracker

# Version marker for tracking code changes
APP_VERSION = "v1.0.1-motion-fix"  # Updated when motion detection fixes applied

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('clash_royale_debug.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting Clash Royale Pocket Companion...")
        
        # Create QApplication
        logger.info("Creating QApplication...")
        app = QApplication(sys.argv)
        app.setApplicationName("Clash Royale Pocket Companion")
        logger.info("QApplication created successfully")
        
        # Create main window
        logger.info("Creating ClashRoyaleTracker window...")
        window = ClashRoyaleTracker()
        logger.info("ClashRoyaleTracker window created successfully")
        
        # Show window
        logger.info("Showing window...")
        window.show()
        logger.info("Window shown successfully")
        
        # Start event loop
        logger.info("Starting Qt event loop...")
        result = app.exec()
        logger.info(f"Qt event loop ended with result: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Fatal error in main(): {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Show error dialog if possible
        try:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Fatal Error")
            msg.setText(f"Application crashed: {str(e)}")
            msg.setDetailedText(traceback.format_exc())
            msg.exec()
        except:
            print(f"Fatal error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
        
        return 1

if __name__ == "__main__":
    try:
        # Print prominent startup banner with timestamp
        startup_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        logger.info("=" * 80)
        logger.info(f"🚀 CLASH ROYALE POCKET COMPANION STARTING")
        logger.info(f"📅 Startup Time: {startup_time}")
        logger.info(f"🔖 Version: {APP_VERSION}")
        logger.info(f"🐍 Python: {sys.version.split()[0]}")
        logger.info("=" * 80)
        
        exit_code = main()
        
        logger.info("=" * 80)
        logger.info(f"⏹️  Application ended with exit code: {exit_code}")
        logger.info("=" * 80)
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Unhandled exception in __main__: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
