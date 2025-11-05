import { useEffect } from 'react';
import { useGameStore } from '../stores/gameStore';

interface BackendData {
  type: string;
  data?: any;
}

interface BackendStatusData {
  running: boolean;
  code?: number;
}

interface BackendErrorData {
  message: string;
}

interface BackendLogData {
  level: string;
  message: string;
}

export function useBackendData() {
  const store = useGameStore();

  useEffect(() => {
    // Handle backend data updates
    const handleUpdate = (data: BackendData) => {
      switch (data.type) {
        case 'elixir':
          if (data.data && typeof data.data.elixir === 'number') {
            store.updateElixir(data.data.elixir);
          }
          break;
          
        case 'timer':
          if (data.data && data.data.timer) {
            store.updateTimer(data.data.timer);
          }
          break;
          
        case 'cards':
          if (data.data) {
            store.updateCards(data.data);
          }
          break;
          
        case 'event':
          if (data.data && data.data.message) {
            store.addEvent(data.data.message);
          }
          break;
          
        case 'init':
          store.addEvent('Backend initialized');
          break;
          
        case 'heartbeat':
          // Silent heartbeat - just update connection status
          store.checkConnection();
          break;
          
        case 'windows_list':
          if (data.data && data.data.windows) {
            store.setWindows(data.data.windows);
            store.addEvent(`Found ${data.data.count} windows`);
          }
          break;
          
        case 'window_selected':
          if (data.data && data.data.success) {
            store.addEvent('Window selected successfully');
          } else {
            store.addEvent(`Failed to select window: ${data.data?.error || 'Unknown error'}`);
          }
          break;
          
        case 'tracking_started':
          if (data.data && data.data.success) {
            store.setTracking(true);
            store.addEvent('Tracking started');
          } else {
            store.addEvent(`Failed to start tracking: ${data.data?.error || 'Unknown error'}`);
          }
          break;
          
        case 'tracking_stopped':
          if (data.data && data.data.success) {
            store.setTracking(false);
            store.addEvent('Tracking stopped');
          }
          break;
          
        default:
          console.log('Unknown backend message type:', data.type);
      }
    };

    // Handle backend status changes
    const handleStatus = (data: BackendStatusData) => {
      store.setBackendStatus(data.running);
      if (data.running) {
        store.addEvent('Backend started');
      } else {
        store.addEvent(`Backend stopped (code: ${data.code || 'unknown'})`);
      }
    };

    // Handle backend errors
    const handleError = (data: BackendErrorData) => {
      console.error('Backend error:', data);
      store.addEvent(`Error: ${data.message}`);
      store.setBackendStatus(false);
    };

    // Handle backend logs
    const handleLog = (data: BackendLogData) => {
      if (data.level === 'error') {
        store.addEvent(`Backend: ${data.message}`);
      }
    };

    // Register listeners
    if (window.electronAPI) {
      window.electronAPI.onBackendUpdate(handleUpdate);
      window.electronAPI.onBackendStatus(handleStatus);
      window.electronAPI.onBackendError(handleError);
      window.electronAPI.onBackendLog(handleLog);
    } else {
      console.warn('Electron API not available - running in browser mode');
    }

    // Connection check interval
    const interval = setInterval(() => {
      store.checkConnection();
    }, 1000);

    // Cleanup
    return () => {
      if (window.electronAPI) {
        window.electronAPI.removeBackendListeners();
      }
      clearInterval(interval);
    };
  }, []);
}

