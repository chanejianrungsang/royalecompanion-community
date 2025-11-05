import { getCardDisplayName } from '../utils/cardNames';

interface CardSlotProps {
  name: string;
  elixirCost: number;
  imageUrl?: string;
  dimmed?: boolean;
  scale?: number;
}

export function CardSlot({ name, elixirCost, imageUrl, dimmed = false, scale = 1 }: CardSlotProps) {
  const displayName = getCardDisplayName(name);
  
  return (
    <div 
      className={`flex flex-col items-center ${dimmed ? 'opacity-50' : ''}`}
      style={{ transform: `scale(${scale})`, transformOrigin: 'top center' }}
    >
      {/* Card Frame */}
      <div className="relative w-full aspect-[3/4] rounded-lg overflow-hidden border-2 border-gray-600 bg-gray-900 shadow-lg group hover:border-purple-500 transition-colors">
        {/* Card Image or Name */}
        {imageUrl ? (
          <img
            src={imageUrl}
            alt={displayName}
            className="w-full h-full object-cover"
            onError={(e) => {
              console.error(`Failed to load card image: ${imageUrl} for card: ${name}`);
              e.currentTarget.style.display = 'none';
            }}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center p-2">
            <span className="text-gray-300 text-xs text-center break-words">{displayName}</span>
          </div>
        )}

        {/* Elixir Cost Bubble */}
        {elixirCost > 0 && (
          <div className="absolute bottom-2 left-2 w-8 h-8 rounded-full bg-gradient-to-br from-purple-600 to-purple-700 border-2 border-purple-300 flex items-center justify-center shadow-lg">
            <span className="text-white font-bold text-sm">{elixirCost}</span>
          </div>
        )}

        {/* Shine effect */}
        <div className="absolute inset-0 bg-gradient-to-br from-white/10 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
      </div>

      {/* Card Name - hide if it's Unknown and has an image */}
      {!(displayName === 'Unknown' && imageUrl) && (
        <span className="mt-2 text-gray-300 text-center text-xs">{displayName}</span>
      )}
    </div>
  );
}

