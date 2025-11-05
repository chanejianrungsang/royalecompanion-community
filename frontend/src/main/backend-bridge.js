const { spawn } = require('child_process');
const path = require('path');

class BackendBridge {
  constructor(mainWindow) {
    this.mainWindow = mainWindow;
    this.pythonProcess = null;
    this.isRunning = false;
    this.restartAttempts = 0;
    this.maxRestartAttempts = 3;
  }

  start() {
    if (this.isRunning) {
      console.log('Backend already running');
      return;
    }

    console.log('Starting Python backend...');
    
    // Determine Python backend path
    const isDev = process.env.NODE_ENV === 'development';
    const backendPath = isDev
      ? path.resolve(__dirname, '../../..', 'backend', 'main_headless.py')
      : path.join(process.resourcesPath, 'backend/main_headless.py');
    
    console.log(`Backend path: ${backendPath}`);
    
    // Spawn Python process
    this.pythonProcess = spawn('python', [backendPath], {
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    this.isRunning = true;
    this.restartAttempts = 0;

    // Parse JSON lines from stdout
    let buffer = '';
    this.pythonProcess.stdout.on('data', (data) => {
      buffer += data.toString();
      const lines = buffer.split('\n');
      
      // Process complete lines
      for (let i = 0; i < lines.length - 1; i++) {
        const line = lines[i].trim();
        // Skip empty lines and non-JSON lines (like ====)
        if (line && line.startsWith('{')) {
          try {
            const json = JSON.parse(line);
            // Send to renderer process
            if (this.mainWindow && !this.mainWindow.isDestroyed()) {
              this.mainWindow.webContents.send('backend-update', json);
            }
          } catch (err) {
            console.error('JSON parse error:', err, 'Line:', line.substring(0, 100));
          }
        }
      }
      
      // Keep incomplete line in buffer
      buffer = lines[lines.length - 1];
    });

    // Log stderr (Python logging goes here)
    this.pythonProcess.stderr.on('data', (data) => {
      const message = data.toString();
      console.log('[Backend stderr]:', message);
      
      // Send important errors to frontend
      if (message.toLowerCase().includes('error') || message.toLowerCase().includes('failed')) {
        if (this.mainWindow && !this.mainWindow.isDestroyed()) {
          this.mainWindow.webContents.send('backend-log', {
            level: 'error',
            message: message
          });
        }
      }
    });

    // Handle process exit
    this.pythonProcess.on('exit', (code, signal) => {
      console.log(`Backend process exited with code ${code}, signal ${signal}`);
      this.isRunning = false;
      
      if (this.mainWindow && !this.mainWindow.isDestroyed()) {
        this.mainWindow.webContents.send('backend-status', {
          running: false,
          code: code,
          signal: signal
        });
      }

      // Auto-restart on unexpected exit
      if (code !== 0 && this.restartAttempts < this.maxRestartAttempts) {
        console.log(`Attempting to restart backend (attempt ${this.restartAttempts + 1}/${this.maxRestartAttempts})`);
        this.restartAttempts++;
        setTimeout(() => this.start(), 2000);
      }
    });

    // Handle errors
    this.pythonProcess.on('error', (err) => {
      console.error('Failed to start backend:', err);
      this.isRunning = false;
      
      if (this.mainWindow && !this.mainWindow.isDestroyed()) {
        this.mainWindow.webContents.send('backend-error', {
          message: `Failed to start backend: ${err.message}`,
          error: err.toString()
        });
      }
    });

    // Notify frontend that backend started
    if (this.mainWindow && !this.mainWindow.isDestroyed()) {
      this.mainWindow.webContents.send('backend-status', {
        running: true
      });
    }
  }

  stop() {
    if (!this.pythonProcess) {
      console.log('No backend process to stop');
      return;
    }

    console.log('Stopping Python backend...');
    this.isRunning = false;
    
    try {
      // Try graceful termination first
      this.pythonProcess.kill('SIGTERM');
      
      // Force kill after timeout
      setTimeout(() => {
        if (this.pythonProcess && !this.pythonProcess.killed) {
          console.log('Force killing backend process');
          this.pythonProcess.kill('SIGKILL');
        }
      }, 5000);
    } catch (err) {
      console.error('Error stopping backend:', err);
    }
    
    this.pythonProcess = null;
  }

  restart() {
    console.log('Restarting backend...');
    this.stop();
    setTimeout(() => this.start(), 1000);
  }

  sendCommand(command) {
    if (!this.pythonProcess || !this.isRunning) {
      console.error('Cannot send command: backend not running');
      return false;
    }

    try {
      const jsonCommand = JSON.stringify(command) + '\n';
      this.pythonProcess.stdin.write(jsonCommand);
      return true;
    } catch (err) {
      console.error('Failed to send command to backend:', err);
      return false;
    }
  }

  listWindows() {
    return this.sendCommand({ command: 'list_windows' });
  }

  selectWindow(hwnd) {
    return this.sendCommand({ command: 'select_window', hwnd: hwnd });
  }

  startTracking() {
    return this.sendCommand({ command: 'start_tracking' });
  }

  stopTracking() {
    return this.sendCommand({ command: 'stop_tracking' });
  }
}

module.exports = BackendBridge;
