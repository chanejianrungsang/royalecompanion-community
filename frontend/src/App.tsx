import { useBackendData } from './hooks/useBackendData';
import { WindowSelector } from './components/WindowSelector';
import { ElixirBar } from './components/ElixirBar';
import { DeckView } from './components/DeckView';
import { EventLog } from './components/EventLog';
import { useGameStore } from './stores/gameStore';
import { useState } from 'react';

export function App() {
  useBackendData();
  const [showEventLog, setShowEventLog] = useState(false);

  const handleMinimize = () => {
    if (window.electronAPI?.minimize) {
      window.electronAPI.minimize();
    }
  };

  const handleMaximize = () => {
    if (window.electronAPI?.maximize) {
      window.electronAPI.maximize();
    }
  };

  const handleClose = () => {
    if (window.electronAPI?.close) {
      window.electronAPI.close();
    }
  };

  return (
    <div className="h-screen bg-gray-950 flex flex-col overflow-hidden select-none" style={{ padding: 'clamp(1rem, 2vw, 2rem)' }}>
      {/* Custom title bar for frameless window */}
      <div 
        className="fixed top-0 left-0 right-0 h-8 bg-gray-950/95 backdrop-blur-sm flex items-center justify-end px-2 z-50"
        style={{ WebkitAppRegion: 'drag' } as any}
      >
        <div className="flex items-center gap-1" style={{ WebkitAppRegion: 'no-drag' } as any}>
          <button
            onClick={handleMinimize}
            className="w-10 h-7 flex items-center justify-center hover:bg-gray-800 rounded transition-colors"
            title="Minimize"
          >
            <span className="text-gray-400 text-xl leading-none">−</span>
          </button>
          <button
            onClick={handleMaximize}
            className="w-10 h-7 flex items-center justify-center hover:bg-gray-800 rounded transition-colors"
            title="Maximize"
          >
            <span className="text-gray-400 text-base leading-none">□</span>
          </button>
          <button
            onClick={handleClose}
            className="w-10 h-7 flex items-center justify-center hover:bg-red-600 rounded transition-colors group"
            title="Close"
          >
            <span className="text-gray-400 group-hover:text-white text-xl leading-none">×</span>
          </button>
        </div>
      </div>
      
      <div className={`w-full mx-auto pt-8 flex-1 ${showEventLog ? 'overflow-y-auto' : ''}`} style={{ maxWidth: 'min(90vw, 768px)', gap: 'clamp(1rem, 2vh, 1.5rem)', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <div className="flex items-center justify-center mb-4">
          <h1 className="font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 leading-normal py-2" style={{ fontSize: 'clamp(1.5rem, 4vw, 2.25rem)' }}>
            RoyaleCompanion Community
          </h1>
        </div>

        {/* Window Selector Section */}
        <WindowSelector />

        {/* Reset Counter Button and Event Log Toggle */}
        <div className="flex justify-center items-center gap-6">
          <button
            onClick={() => useGameStore.getState().reset()}
            className="px-6 py-2 rounded-lg border-2 border-gray-600 text-gray-300 hover:border-purple-500 hover:text-purple-400 transition-colors font-medium"
          >
            Reset Counter
          </button>
          
          <label className="flex items-center gap-2 cursor-pointer group">
            <input
              type="checkbox"
              checked={showEventLog}
              onChange={(e) => setShowEventLog(e.target.checked)}
              className="w-5 h-5 rounded border-2 border-gray-600 bg-gray-950 checked:bg-purple-500 checked:border-purple-500 cursor-pointer appearance-none checked:bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8cGF0aCBkPSJNMTMuMzMzMyA0TDYgMTEuMzMzM0wyLjY2NjY3IDgiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=')] bg-center bg-no-repeat focus:outline-none"
            />
            <span className="text-gray-400 group-hover:text-purple-400 transition-colors">Show Event Log</span>
          </label>
        </div>

        {/* Elixir Counter Section */}
        <ElixirBar />

        {/* Deck View Section */}
        <DeckView />

        {/* Event Log Section */}
        {showEventLog && <EventLog />}
      </div>
    </div>
  );
}

