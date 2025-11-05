"""
Configuration constants and settings
"""

# Elixir system constants
ELIXIR_PHASES = {
    'single': 1 / 2.8,    # Single elixir: 1 per 2.8 seconds
    'double': 1 / 1.4,    # Double elixir: 1 per 1.4 seconds  
    'triple': 1 / 0.933   # Triple elixir: 1 per 0.933 seconds
}

INITIAL_ELIXIR = 5.0
MAX_ELIXIR = 10.0

# Spell costs (MVP set)
SPELL_COSTS = {
    'the_log': 2,
    'fireball': 4,
    'arrows': 3,
    'zap': 2,
    'rocket': 6,
    'poison': 4,
    'freeze': 4,
    'tornado': 3,
    'snowball': 2,
    'earthquake': 3
}

DEFAULT_SPELL_COST = 4

# Detection parameters
STOPWATCH_DETECTION_WINDOW = 300  # ms
EVENT_ATTRIBUTION_WINDOW = (-250, 150)  # ms before/after
PROXIMITY_THRESHOLD = 150  # pixels

# HSV color ranges for stopwatch detection
RED_HSV_RANGES = [
    ((0, 50, 50), (10, 255, 255)),      # Lower red range
    ((170, 50, 50), (180, 255, 255))    # Upper red range
]

BLUE_HSV_RANGE = ((110, 50, 50), (130, 255, 255))

# Capture settings
TARGET_FPS = 60
CAPTURE_TIMEOUT = 1.0 / TARGET_FPS
