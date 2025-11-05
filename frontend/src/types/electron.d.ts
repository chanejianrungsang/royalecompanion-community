export interface ElectronAPI {
  controlBackend: (command: any) => void;
  onBackendUpdate: (callback: (data: any) => void) => void;
  onBackendStatus: (callback: (data: any) => void) => void;
  onBackendError: (callback: (data: any) => void) => void;
  onBackendLog: (callback: (data: any) => void) => void;
  removeBackendListeners: () => void;
  minimize: () => void;
  maximize: () => void;
  close: () => void;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}

