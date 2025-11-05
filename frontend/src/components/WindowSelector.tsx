import { useState, useEffect } from 'react';
import { Monitor, X } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from './ui/dialog';
import { Button } from './ui/button';
import { useGameStore } from '../stores/gameStore';

export function WindowSelector() {
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const windows = useGameStore((state) => state.windows);
  const selectedWindow = useGameStore((state) => state.selectedWindow);
  const tracking = useGameStore((state) => state.tracking);
  const setSelectedWindow = useGameStore((state) => state.setSelectedWindow);

  // Request window list when dialog opens
  useEffect(() => {
    if (isOpen && window.electronAPI) {
      setIsLoading(true);
      window.electronAPI.controlBackend({ type: 'list_windows' });
      // Stop loading after 2 seconds (should be plenty of time)
      setTimeout(() => setIsLoading(false), 2000);
    }
  }, [isOpen]);
  
  const handleRefresh = () => {
    if (window.electronAPI) {
      setIsLoading(true);
      window.electronAPI.controlBackend({ type: 'list_windows' });
      setTimeout(() => setIsLoading(false), 2000);
    }
  };

  const handleCapture = (win: { hwnd: number; title: string }) => {
    if (window.electronAPI) {
      // Select the window
      window.electronAPI.controlBackend({
        type: 'select_window',
        hwnd: win.hwnd
      });
      setSelectedWindow(win.hwnd, win.title);
      
      // Start tracking
      window.electronAPI.controlBackend({ type: 'start_tracking' });
    }
    setIsOpen(false);
  };

  const handleCancel = () => {
    if (window.electronAPI) {
      window.electronAPI.controlBackend({ type: 'stop_tracking' });
    }
    setSelectedWindow(null);
  };

  return (
    <div className="w-full">
      {!selectedWindow ? (
        <button
          onClick={() => setIsOpen(true)}
          className="flex flex-col items-center justify-center w-full py-8 px-4 rounded-lg border-2 border-dashed border-gray-600 hover:border-purple-500 transition-colors group"
        >
          <Monitor className="w-16 h-16 text-gray-500 group-hover:text-purple-500 transition-colors mb-3" />
          <span className="text-gray-400 group-hover:text-purple-400 transition-colors">
            Click to share screen
          </span>
        </button>
      ) : (
        <div className="flex items-center justify-between w-full py-4 px-6 rounded-lg bg-gray-800 border border-purple-500/50">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            <span className="text-purple-400">Streaming {selectedWindow?.title || 'Unknown'}...</span>
          </div>
          <button
            onClick={handleCancel}
            className="p-1.5 rounded-md hover:bg-gray-700 transition-colors group"
          >
            <X className="w-4 h-4 text-red-500 group-hover:text-red-400" />
          </button>
        </div>
      )}

      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="sm:max-w-3xl bg-gray-800 border-gray-700">
          <DialogHeader>
            <div className="flex items-center justify-between pr-8">
              <div>
                <DialogTitle className="text-gray-100">Select Application to Capture</DialogTitle>
                <DialogDescription className="text-gray-400">
                  Choose a window to start capturing gameplay.
                </DialogDescription>
              </div>
            </div>
            <Button
              onClick={handleRefresh}
              disabled={isLoading}
              className="bg-purple-600 hover:bg-purple-700 mt-3"
              size="sm"
            >
              {isLoading ? 'Loading...' : 'Refresh'}
            </Button>
          </DialogHeader>
          <div className="grid grid-cols-2 gap-4 mt-4 max-h-96 overflow-y-auto pr-2 window-selector-scroll">
            {isLoading && windows.length === 0 ? (
              <div className="col-span-2 text-center py-8 text-gray-400">
                <div className="animate-pulse">Loading windows...</div>
              </div>
            ) : windows.length === 0 ? (
              <div className="col-span-2 text-center py-8 text-gray-400">
                No windows found. Click refresh to scan again.
              </div>
            ) : (
              windows.map((win) => (
                <div
                  key={win.hwnd}
                  className="flex flex-col p-4 rounded-lg bg-gray-900 border-2 border-gray-700 hover:border-purple-500 transition-all cursor-pointer group"
                  onClick={() => handleCapture(win)}
                >
                  <div className="w-full aspect-video bg-gray-950 rounded-md mb-3 overflow-hidden border border-gray-800 group-hover:border-purple-500/30 transition-colors">
                    {win.thumbnail ? (
                      <img
                        src={win.thumbnail}
                        alt={win.title}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center">
                        <Monitor className="w-16 h-16 text-gray-600 group-hover:text-purple-500/50 transition-colors mb-2" />
                      </div>
                    )}
                  </div>
                  <span className="text-gray-300 text-sm text-center mb-3 line-clamp-2 group-hover:text-purple-300 transition-colors">{win.title}</span>
                  <Button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCapture(win);
                    }}
                    className="w-full bg-purple-600 hover:bg-purple-700 text-white"
                  >
                    Capture
                  </Button>
                </div>
              ))
            )}
          </div>
          <style>{`
            .window-selector-scroll {
              scrollbar-width: thin;
              scrollbar-color: rgba(168, 85, 247, 0.5) transparent;
            }
            .window-selector-scroll::-webkit-scrollbar {
              width: 8px;
            }
            .window-selector-scroll::-webkit-scrollbar-track {
              background: transparent;
            }
            .window-selector-scroll::-webkit-scrollbar-thumb {
              background: rgba(168, 85, 247, 0.5);
              border-radius: 4px;
            }
            .window-selector-scroll::-webkit-scrollbar-thumb:hover {
              background: rgba(168, 85, 247, 0.7);
            }
          `}</style>
        </DialogContent>
      </Dialog>
    </div>
  );
}

