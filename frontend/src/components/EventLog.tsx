import { useGameStore } from '../stores/gameStore';

export function EventLog() {
  const events = useGameStore((state) => state.events);

  const formatTime = (timestamp: number) => {
    const now = Date.now();
    const diff = Math.floor((now - timestamp) / 1000);
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
  };

  return (
    <div className="w-full p-6 rounded-lg bg-gray-800 border border-gray-700 select-text">
      <div className="flex items-center gap-2 mb-3">
        <div className="w-1 h-5 bg-blue-500 rounded-full" />
        <h3 className="text-blue-400 font-medium">Event Log</h3>
        <span className="text-gray-500 text-sm ml-auto">Dev Mode</span>
      </div>
      <div className="space-y-1 max-h-48 overflow-y-auto">
        {events.length === 0 ? (
          <div className="text-sm text-gray-500 italic text-center py-4">
            No events yet
          </div>
        ) : (
          events.map((event, index) => (
            <div
              key={`${event.time}-${index}`}
              className="text-sm text-gray-300 font-mono bg-gray-900 px-3 py-1.5 rounded border border-gray-700 flex justify-between items-center"
            >
              <span className="flex-1">{event.text}</span>
              <span className="text-xs text-gray-500 ml-2">{formatTime(event.time)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

