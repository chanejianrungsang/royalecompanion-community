import { useGameStore } from '../stores/gameStore';

export function MatchTimer() {
  const timer = useGameStore((state) => state.timer);

  return (
    <div className="flex items-center gap-2 bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5">
      <span className="text-purple-300 font-mono text-sm">{timer}</span>
    </div>
  );
}

