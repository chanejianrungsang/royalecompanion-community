/**
 * Card name display mapping - expands abbreviated names for UI
 */
export const CARD_DISPLAY_NAMES: Record<string, string> = {
  // Mini Pekka
  'MP': 'Mini PEKKA',
  'mp': 'Mini PEKKA',
  
  // Three Musketeers
  '3M': 'Three Musketeers',
  '3m': 'Three Musketeers',
  
  // Musketeer
  'Musk': 'Musketeer',
  'musk': 'Musketeer',
  
  // Elite Barbarians
  'eBarbs': 'Elite Barbarians',
  'ebarbs': 'Elite Barbarians',
  
  // Electro Wizard
  'eWiz': 'Electro Wizard',
  'ewiz': 'Electro Wizard',
  
  // Electro Dragon
  'eDragon': 'Electro Dragon',
  'edragon': 'Electro Dragon',
  
  // Ice Wizard
  'IceWiz': 'Ice Wizard',
  'icewiz': 'Ice Wizard',
  
  // Wizard
  'Wiz': 'Wizard',
  'wiz': 'Wizard',
  
  // Royal Giant
  'RG': 'Royal Giant',
  'rg': 'Royal Giant',
  
  // Baby Dragon
  'BabyD': 'Baby Dragon',
  'babyd': 'Baby Dragon',
  
  // Inferno Dragon
  'InfernoD': 'Inferno Dragon',
  'infernod': 'Inferno Dragon',
  
  // Executioner
  'Exe': 'Executioner',
  'exe': 'Executioner',
  
  // Valkyrie
  'Valk': 'Valkyrie',
  'valk': 'Valkyrie',
  
  // Lumberjack
  'Lumber': 'Lumberjack',
  'lumber': 'Lumberjack',
  
  // Mega Minion
  'MM': 'Mega Minion',
  'mm': 'Mega Minion',
  
  // Goblins
  'Gobs': 'Goblins',
  'gobs': 'Goblins',
  
  // Spear Goblins
  'SpearGobs': 'Spear Goblins',
  'speargobs': 'Spear Goblins',
  
  // Dart Goblin
  'DartGob': 'Dart Goblin',
  'dartgob': 'Dart Goblin',
  
  // Goblin Giant
  'GobGiant': 'Goblin Giant',
  'gobgiant': 'Goblin Giant',
  
  // Goblin Hut
  'GobHut': 'Goblin Hut',
  'gobhut': 'Goblin Hut',
  
  // Goblin Gang
  'GobGang': 'Goblin Gang',
  'gobgang': 'Goblin Gang',
  
  // Skeleton Army
  'Skarmy': 'Skeleton Army',
  'skarmy': 'Skeleton Army',
  
  // Skeletons
  'Skellies': 'Skeletons',
  'skellies': 'Skeletons',
  
  // Skeleton Barrel
  'SkellyBarrel': 'Skeleton Barrel',
  'skellybarrel': 'Skeleton Barrel',
  
  // Battle Ram
  'Ram': 'Battle Ram',
  'ram': 'Battle Ram',
  
  // Goblin Barrel
  'Barrel': 'Goblin Barrel',
  'barrel': 'Goblin Barrel',
  
  // Barbarian Barrel
  'BarbBarrel': 'Barbarian Barrel',
  'barbbarrel': 'Barbarian Barrel',
  
  // Hog Rider
  'Hog': 'Hog Rider',
  'hog': 'Hog Rider',
  
  // Barbarians
  'Barbs': 'Barbarians',
  'barbs': 'Barbarians',
  
  // Barbarian Hut
  'BarbHut': 'Barbarian Hut',
  'barbhut': 'Barbarian Hut',
  
  // Royal Ghost
  'Ghost': 'Royal Ghost',
  'ghost': 'Royal Ghost',
  
  // Elixir Collector
  'Pump': 'Elixir Collector',
  'pump': 'Elixir Collector',
  
  // Lava Hound
  'Lava': 'Lava Hound',
  'lava': 'Lava Hound',
  
  // Minion Horde
  'Horde': 'Minion Horde',
  'horde': 'Minion Horde',
  
  // Inferno Tower
  'Inferno': 'Inferno Tower',
  'inferno': 'Inferno Tower',
};

/**
 * Get display name for a card, expanding abbreviations
 */
export function getCardDisplayName(cardName: string): string {
  if (!cardName) return 'Unknown';
  
  // Handle unknown card variations
  if (cardName.toLowerCase() === 'unknown' || cardName === 'unknown_png') {
    return 'Unknown';
  }
  
  // Check if we have a mapping for this name
  if (CARD_DISPLAY_NAMES[cardName]) {
    return CARD_DISPLAY_NAMES[cardName];
  }
  
  // Otherwise return the original name (might already be expanded)
  return cardName;
}
