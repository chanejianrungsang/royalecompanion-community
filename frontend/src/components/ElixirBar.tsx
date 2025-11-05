import { Droplet } from 'lucide-react';
import { useGameStore } from '../stores/gameStore';

export function ElixirBar() {
  const elixir = useGameStore((state) => state.elixir);
  const max = 10;
  const percentage = (elixir / max) * 100;

  return (
    <div className="w-full p-6 rounded-lg bg-gray-800 border border-gray-700">
      <div className="flex items-center gap-4">
        {/* Elixir Droplet Icon */}
        <div className="relative flex-shrink-0">
          <div className="absolute inset-0 bg-purple-500 rounded-full blur-xl opacity-50" />
          <Droplet className="relative w-12 h-12 text-purple-500 fill-purple-500" />
        </div>

        {/* Elixir Bar Container */}
        <div className="flex-1">
          <div className="relative">
            {/* Background bar */}
            <div className="h-8 bg-gray-900 rounded-full border border-gray-700 overflow-hidden">
              {/* Fill bar with glow */}
              <div
                className="h-full bg-gradient-to-r from-purple-600 via-purple-500 to-purple-400 rounded-full transition-all duration-300 relative"
                style={{ width: `${percentage}%` }}
              >
                <div className="absolute inset-0 bg-purple-400 opacity-50 blur-sm" />
              </div>
            </div>

            {/* Tick marks at 1, 2, 3, 4, 5, 6, 7, 8, 9 elixir points */}
            <div className="absolute inset-0 flex items-center">
              {Array.from({ length: 9 }).map((_, i) => {
                const tickElixir = i + 1; // 1, 2, 3, 4, 5, 6, 7, 8, 9
                const tickPosition = (tickElixir / max) * 100; // 10%, 20%, 30%, etc.
                const isReached = elixir >= tickElixir;
                
                return (
                  <div
                    key={i}
                    className="absolute w-0.5 h-6 bg-gray-700 rounded-full transition-opacity"
                    style={{
                      left: `${tickPosition}%`,
                      transform: 'translateX(-50%)',
                      opacity: isReached ? 0.3 : 0.5,
                    }}
                  />
                );
              })}
            </div>
          </div>

          {/* Counter Text */}
          <div className="mt-2 text-center text-purple-300 font-medium">
            {elixir.toFixed(1)} / {max}
          </div>
        </div>
      </div>
    </div>
  );
}

