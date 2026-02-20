"""
Constants for the PHOLUS analysis pipeline.

Defines file paths, the seven Diplomacy Great Powers, and per-game mappings
of which powers were controlled by experienced human players versus novices
during the Centaur (PHOLUS-assisted) games.
"""

# Path to the Centaur game JSON files (old payload format used in the study)
CENTAUR_PATH = "data/games/old_payload_format"

# The seven Great Powers in standard Diplomacy
POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

# Mapping from Centaur game ID to the list of powers controlled by
# experienced (veteran) human players in each game.
HUMAN_PLAYERS = {
    "Centaur0": ["FRANCE", "RUSSIA"],
    "Centaur1": ["FRANCE", "RUSSIA", "ITALY"],
    "Centaur2": ["FRANCE", "RUSSIA", "GERMANY"],
    "Centaur3": ["AUSTRIA", "ENGLAND", "TURKEY"],
    "Centaur4": ["TURKEY", "AUSTRIA", "GERMANY", "ITALY"],
    "Centaur5": ["ENGLAND", "TURKEY", "ITALY"],
    "Centaur6": ["ENGLAND", "GERMANY", "ITALY"],
    "Centaur7": ["ENGLAND", "AUSTRIA"],
    "Centaur8": ["GERMANY", "TURKEY", "ITALY"],
    "Centaur9": ["ITALY", "GERMANY", "RUSSIA"],
    "Centaur10": ["AUSTRIA", "TURKEY"],
    "Centaur11": ["GERMANY", "FRANCE", "RUSSIA"],
}

# Mapping from Centaur game ID to the list of powers controlled by
# novice players (no prior Diplomacy experience). Only games 8-11
# included novice participants.
NOVICES = {
    "Centaur8": ["RUSSIA"],
    "Centaur9": ["FRANCE", "TURKEY"],
    "Centaur10": ["ITALY"],
    "Centaur11": ["ENGLAND"],
}