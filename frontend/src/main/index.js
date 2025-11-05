const { app, BrowserWindow, ipcMain, shell, protocol } = require('electron');
const path = require('path');
const fs = require('fs');
const dotenv = require('dotenv');
const BackendBridge = require('./backend-bridge');

let mainWindow;
let backendBridge;

// Register custom protocol for assets BEFORE app is ready
protocol.registerSchemesAsPrivileged([
  {
    scheme: 'asset',
    privileges: {
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true,
      bypassCSP: true
    }
  }
]);

// Load environment variables so the main process can access Supabase config
const envCandidates = [
  path.resolve(__dirname, '../../..', '.env'),
  path.resolve(__dirname, '../../.env'),
  path.resolve(__dirname, '../../.env.local'),
];

envCandidates.forEach((envPath) => {
  if (fs.existsSync(envPath)) {
    dotenv.config({ path: envPath, override: true });
  }
});

const supabaseUrl = process.env.SUPABASE_URL || process.env.VITE_SUPABASE_URL;
const supabaseOrigin = supabaseUrl ? new URL(supabaseUrl).origin : null;

if (!supabaseOrigin) {
  console.warn(
    '[Auth] SUPABASE_URL / VITE_SUPABASE_URL not configured; OAuth flows will open but redirect filters are disabled.'
  );
}

// Register custom protocol for OAuth callback
if (process.defaultApp) {
  if (process.argv.length >= 2) {
    app.setAsDefaultProtocolClient('royalecompanion', process.execPath, [path.resolve(process.argv[1])]);
  }
} else {
  app.setAsDefaultProtocolClient('royalecompanion');
}

// Handle OAuth callback URL
const gotTheLock = app.requestSingleInstanceLock();

if (!gotTheLock) {
  app.quit();
} else {
  app.on('second-instance', (event, commandLine, workingDirectory) => {
    // Someone tried to run a second instance, focus our window instead
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
      
      // Handle OAuth callback from protocol
      const url = commandLine.find(arg => arg.startsWith('royalecompanion://'));
      if (url) {
        handleOAuthCallback(url);
      }
    }
  });

  // Handle OAuth callback on macOS
  app.on('open-url', (event, url) => {
    event.preventDefault();
    if (url.startsWith('royalecompanion://')) {
      handleOAuthCallback(url);
    }
  });
}

function handleOAuthCallback(url) {
  // Extract the hash/query params and send to renderer
  if (mainWindow) {
    // Convert royalecompanion:// URL to localhost URL format
    const callbackUrl = url.replace('royalecompanion://', 'http://localhost:5173/');
    console.log('[AUTH] OAuth callback received');
    console.log('[AUTH] Original URL:', url);
    console.log('[AUTH] Converted URL:', callbackUrl);
    
    // Load the callback URL in the app (preserves hash with tokens)
    mainWindow.loadURL(callbackUrl);
  }
}

function createWindow() {
  // Get primary display dimensions
  const { screen } = require('electron');
  const primaryDisplay = screen.getPrimaryDisplay();
  const { height: screenHeight } = primaryDisplay.workAreaSize;
  
  // Fixed width, full vertical height (minus taskbar)
  const windowWidth = 700;
  const windowHeight = screenHeight;
  
  mainWindow = new BrowserWindow({
    width: windowWidth,
    height: windowHeight,
    minWidth: 500,
    minHeight: 600,
    frame: false, // Remove default title bar and frame
    transparent: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    },
    title: 'RoyaleCompanion',
    backgroundColor: '#0a0a14',
    autoHideMenuBar: true
  });

  // Enforce minimum size constraints
  mainWindow.setMinimumSize(500, 600);

  // Load React app
  const isDev = process.env.NODE_ENV === 'development';
  
  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    // Don't open DevTools on startup
    // mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../../dist/index.html'));
  }

  // Force OAuth URLs to open in system browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    // Open OAuth/external URLs in system browser
    if (url.startsWith('https://accounts.google.com') || 
        (supabaseOrigin && url.startsWith(supabaseOrigin))) {
      shell.openExternal(url);
      return { action: 'deny' };
    }
    return { action: 'allow' };
  });

  // Intercept navigation events for OAuth
  mainWindow.webContents.on('will-navigate', (event, url) => {
    // If navigating to OAuth URL, open in browser instead
    if (url.startsWith('https://accounts.google.com') || 
        (supabaseOrigin && url.startsWith(supabaseOrigin) && 
         url.includes('/auth/v1/authorize'))) {
      event.preventDefault();
      shell.openExternal(url);
    }
    // Allow localhost navigation (for OAuth callback)
    else if (url.startsWith('http://localhost:5173')) {
      // Allow the navigation to complete
      return;
    }
  });

  // Start Python backend
  backendBridge = new BackendBridge(mainWindow);
  backendBridge.start();

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Handle backend control requests from renderer
ipcMain.on('backend-control', (event, action) => {
  if (!backendBridge) return;
  
  switch (action.type) {
    case 'start':
      backendBridge.start();
      break;
    case 'stop':
      backendBridge.stop();
      break;
    case 'restart':
      backendBridge.restart();
      break;
    case 'list_windows':
      backendBridge.listWindows();
      break;
    case 'select_window':
      backendBridge.selectWindow(action.hwnd);
      break;
    case 'start_tracking':
      backendBridge.startTracking();
      break;
    case 'stop_tracking':
      backendBridge.stopTracking();
      break;
  }
});

// Handle window controls for frameless window
ipcMain.on('window-minimize', () => {
  if (mainWindow) {
    mainWindow.minimize();
  }
});

ipcMain.on('window-maximize', () => {
  if (mainWindow) {
    if (mainWindow.isMaximized()) {
      mainWindow.unmaximize();
    } else {
      mainWindow.maximize();
    }
  }
});

ipcMain.on('window-close', () => {
  if (mainWindow) {
    mainWindow.close();
  }
});

app.whenReady().then(() => {
  // Register asset:// protocol to load files directly from assets/ folder
  protocol.registerFileProtocol('asset', (request, callback) => {
    const url = request.url.replace('asset://', '');
    
    // Get the root directory (2 levels up from frontend/src/main/)
    const rootPath = path.join(__dirname, '..', '..', '..');
    const assetsPath = path.join(rootPath, 'assets', url);
    
    console.log('[ASSET] Loading asset:', url);
    console.log('[ASSET] Full path:', assetsPath);
    
    // Check if file exists
    if (fs.existsSync(assetsPath)) {
      callback({ path: assetsPath });
    } else {
      // Fallback to unknown.png if file doesn't exist
      const unknownPath = path.join(rootPath, 'assets', 'unknown.png');
      console.warn('[ASSET] File not found:', assetsPath);
      console.log('[ASSET] Using fallback:', unknownPath);
      callback({ path: unknownPath });
    }
  });
  
  createWindow();
  
  // Check if app was launched with a protocol URL (Windows/Linux)
  if (process.platform === 'win32' || process.platform === 'linux') {
    const protocolUrl = process.argv.find(arg => arg.startsWith('royalecompanion://'));
    if (protocolUrl) {
      console.log('[AUTH] App launched with protocol URL:', protocolUrl);
      // Wait a bit for the window to be ready
      setTimeout(() => handleOAuthCallback(protocolUrl), 1000);
    }
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    if (backendBridge) {
      backendBridge.stop();
    }
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.on('quit', () => {
  if (backendBridge) {
    backendBridge.stop();
  }
});

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
  if (mainWindow) {
    mainWindow.webContents.send('backend-error', { 
      message: error.message,
      stack: error.stack 
    });
  }
});
