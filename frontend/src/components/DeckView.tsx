import { CardSlot } from './CardSlot';
import { MatchTimer } from './MatchTimer';
import { useGameStore } from '../stores/gameStore';

export function DeckView() {
  const cardsInHand = useGameStore((state) => state.cardsInHand);
  const upcomingCards = useGameStore((state) => state.upcomingCards);

  // Ensure we always have 4 cards in hand and 4 upcoming (fill with placeholders if needed)
  const currentHand = [...cardsInHand];
  while (currentHand.length < 4) {
    currentHand.push({ name: 'Unknown', elixirCost: 0, imageUrl: '/unknown.png' });
  }

  const nextQueue = [...upcomingCards];
  while (nextQueue.length < 4) {
    nextQueue.push({ name: 'Unknown', elixirCost: 0, imageUrl: '/unknown.png' });
  }

  return (
    <div className="w-full p-6 rounded-lg bg-gray-800 border border-gray-700">
      {/* Current Hand */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-1 h-5 bg-purple-500 rounded-full" />
          <h3 className="text-purple-400 font-medium">Current Hand</h3>
          <div className="ml-auto">
            <MatchTimer />
          </div>
        </div>
        <div className="grid grid-cols-4 gap-3">
          {currentHand.slice(0, 4).map((card, index) => (
            <CardSlot
              key={`hand-${index}`}
              name={card.name}
              elixirCost={card.elixirCost}
              imageUrl={card.imageUrl}
            />
          ))}
        </div>
      </div>

      {/* Upcoming Cards */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <div className="w-1 h-5 bg-gray-600 rounded-full" />
          <h3 className="text-gray-400 font-medium">Next in Queue</h3>
        </div>
        <div className="grid grid-cols-4 gap-3">
          {nextQueue.slice(0, 4).map((card, index) => (
            <CardSlot
              key={`queue-${index}`}
              name={card.name}
              elixirCost={card.elixirCost}
              imageUrl={card.imageUrl}
              scale={index === 0 ? 1 : 0.75}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

