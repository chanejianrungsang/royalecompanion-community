const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Listen for backend updates
  onBackendUpdate: (callback) => {
    ipcRenderer.on('backend-update', (event, data) => callback(data));
  },
  
  // Listen for backend status changes
  onBackendStatus: (callback) => {
    ipcRenderer.on('backend-status', (event, data) => callback(data));
  },
  
  // Listen for backend errors
  onBackendError: (callback) => {
    ipcRenderer.on('backend-error', (event, data) => callback(data));
  },
  
  // Listen for backend logs
  onBackendLog: (callback) => {
    ipcRenderer.on('backend-log', (event, data) => callback(data));
  },
  
  // Remove all backend listeners
  removeBackendListeners: () => {
    ipcRenderer.removeAllListeners('backend-update');
    ipcRenderer.removeAllListeners('backend-status');
    ipcRenderer.removeAllListeners('backend-error');
    ipcRenderer.removeAllListeners('backend-log');
  },
  
  // Control backend (start/stop/restart)
  controlBackend: (action) => {
    ipcRenderer.send('backend-control', action);
  },
  
  // Get app version
  getVersion: () => '1.0.0',
  
  // Window controls for frameless window
  minimize: () => {
    ipcRenderer.send('window-minimize');
  },
  
  maximize: () => {
    ipcRenderer.send('window-maximize');
  },
  
  close: () => {
    ipcRenderer.send('window-close');
  }
});


