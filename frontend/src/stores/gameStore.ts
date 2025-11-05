import { create } from 'zustand';

export interface Card {
  name: string;
  elixirCost: number;
  imageUrl?: string;
}

export interface GameEvent {
  text: string;
  time: number;
}

export interface WindowInfo {
  hwnd: number;
  title: string;
  thumbnail?: string; // base64 image data URL
}

interface GameState {
  // Game state
  elixir: number;
  timer: string;
  cards: string[]; // Legacy: array of card names
  cardsInHand: Card[];
  upcomingCards: Card[];
  events: GameEvent[];
  connected: boolean;
  lastUpdate: number | null;
  
  // Backend status
  backendRunning: boolean;
  tracking: boolean;
  
  // Window selection
  windows: WindowInfo[];
  selectedWindow: WindowInfo | null;
  
  // Actions
  updateElixir: (elixir: number) => void;
  updateTimer: (timer: string) => void;
  updateCards: (cardsData: any) => void;
  addEvent: (event: string | Partial<GameEvent>) => void;
  checkConnection: () => void;
  setBackendStatus: (running: boolean) => void;
  setWindows: (windows: WindowInfo[]) => void;
  setSelectedWindow: (hwnd: number | null, title?: string) => void;
  setTracking: (tracking: boolean) => void;
  reset: () => void;
}

export const useGameStore = create<GameState>((set, get) => ({
  // Game state
  elixir: 5.0,
  timer: '3:00',
  cards: ['Unknown', 'Unknown', 'Unknown', 'Unknown'],
  cardsInHand: [
    { name: 'Unknown', elixirCost: 0, imageUrl: 'asset://unknown.png' },
    { name: 'Unknown', elixirCost: 0, imageUrl: 'asset://unknown.png' },
    { name: 'Unknown', elixirCost: 0, imageUrl: 'asset://unknown.png' },
    { name: 'Unknown', elixirCost: 0, imageUrl: 'asset://unknown.png' },
  ],
  upcomingCards: [
    { name: 'Unknown', elixirCost: 0, imageUrl: 'asset://unknown.png' },
    { name: 'Unknown', elixirCost: 0, imageUrl: 'asset://unknown.png' },
    { name: 'Unknown', elixirCost: 0, imageUrl: 'asset://unknown.png' },
    { name: 'Unknown', elixirCost: 0, imageUrl: 'asset://unknown.png' },
  ],
  events: [],
  connected: false,
  lastUpdate: null,
  
  // Backend status
  backendRunning: false,
  tracking: false,
  
  // Window selection
  windows: [],
  selectedWindow: null,
  
  // Actions
  updateElixir: (elixir: number) => set({ 
    elixir: Math.max(0, Math.min(10, elixir)),
    lastUpdate: Date.now(), 
    connected: true 
  }),
  
  updateTimer: (timer: string) => set({ timer }),
  
  updateCards: (cardsData: any) => {
    console.log('[GameStore] updateCards called with:', cardsData);
    
    // Helper function to convert card name to Card object
    const toCardObject = (name: string): Card => {
      const isUnknown = name === 'Unknown' || name === 'unknown_png' || name === 'unknown-nobg' || !name;
      const isEvo = name && (name.includes('_evo') || name.includes('Evo'));
      const cardFolder = isEvo ? 'card_ed_evo' : 'card_ed';
      
      return {
        name: isUnknown ? 'Unknown' : name,
        elixirCost: 0,
        imageUrl: isUnknown 
          ? 'asset://unknown.png' 
          : `asset://card_images/${cardFolder}/${name}.png`
      };
    };
    
    if (Array.isArray(cardsData)) {
      // Simple array of card names - convert to Card objects
      const cardNames = cardsData.slice(0, 8);
      
      const inHand = cardNames.slice(0, 4).map((name: string) => {
        const card = toCardObject(name);
        console.log(`[GameStore] Card in hand: name="${name}" -> imageUrl="${card.imageUrl}"`);
        return card;
      });
      
      const upcoming = cardNames.slice(4, 8).map((name: string) => toCardObject(name));
      
      console.log('[GameStore] Setting cardsInHand:', inHand);
      console.log('[GameStore] Setting upcomingCards:', upcoming);
      
      set({ 
        cards: cardNames.slice(0, 4),
        cardsInHand: inHand,
        upcomingCards: upcoming
      });
    } else if (cardsData.in_hand || cardsData.upcoming || cardsData.cards) {
      // Object format with in_hand and upcoming arrays
      const inHandNames = cardsData.in_hand || cardsData.cards?.slice(0, 4) || [];
      const upcomingNames = cardsData.upcoming || cardsData.cards?.slice(4, 8) || [];
      
      console.log('[GameStore] Received in_hand:', inHandNames);
      console.log('[GameStore] Received upcoming:', upcomingNames);
      
      const inHand = inHandNames.map((name: string) => {
        const card = toCardObject(name);
        console.log(`[GameStore] Card in hand: name="${name}" -> imageUrl="${card.imageUrl}"`);
        return card;
      });
      
      const upcoming = upcomingNames.map((name: string) => toCardObject(name));
      
      console.log('[GameStore] Setting cardsInHand:', inHand);
      console.log('[GameStore] Setting upcomingCards:', upcoming);
      
      set({ 
        cards: inHandNames,
        cardsInHand: inHand,
        upcomingCards: upcoming
      });
    }
  },
  
  addEvent: (event: string | Partial<GameEvent>) => set((state) => {
    const newEvent: GameEvent = typeof event === 'string' 
      ? { text: event, time: Date.now() }
      : { text: event.text || '', time: event.time || Date.now() };
    
    return {
      events: [newEvent, ...state.events].slice(0, 100) // Keep last 100 events
    };
  }),
  
  checkConnection: () => {
    const state = get();
    const isConnected = state.lastUpdate ? (Date.now() - state.lastUpdate < 2000) : false;
    if (state.connected !== isConnected) {
      set({ connected: isConnected });
    }
  },
  
  setBackendStatus: (running: boolean) => set({ backendRunning: running }),
  
  setWindows: (windows: WindowInfo[]) => set({ windows }),
  
  setSelectedWindow: (hwnd: number | null, title?: string) => set({ 
    selectedWindow: hwnd ? { hwnd, title: title || '' } : null 
  }),
  
  setTracking: (tracking: boolean) => set({ tracking }),
  
  reset: () => set({
    elixir: 5.0,
    timer: '3:00',
    cards: ['Unknown', 'Unknown', 'Unknown', 'Unknown'],
    cardsInHand: [
      { name: 'Unknown', elixirCost: 0, imageUrl: '/unknown.png' },
      { name: 'Unknown', elixirCost: 0, imageUrl: '/unknown.png' },
      { name: 'Unknown', elixirCost: 0, imageUrl: '/unknown.png' },
      { name: 'Unknown', elixirCost: 0, imageUrl: '/unknown.png' },
    ],
    upcomingCards: [
      { name: 'Unknown', elixirCost: 0, imageUrl: '/unknown.png' },
      { name: 'Unknown', elixirCost: 0, imageUrl: '/unknown.png' },
      { name: 'Unknown', elixirCost: 0, imageUrl: '/unknown.png' },
      { name: 'Unknown', elixirCost: 0, imageUrl: '/unknown.png' },
    ],
    events: [],
    connected: false,
    lastUpdate: null,
    tracking: false
  })
}));

