# PHOLUS: Personalized Help for Optimizing Low-Skilled Users' Strategy

Analysis code and data for the paper:

> **Personalized Help for Optimizing Low-Skilled Users' Strategy**
> Feng Gu, Wichayaporn Wongkamjan, Jonathan K. Kummerfeld, Denis Peskoff, Jonathan May, Jordan Lee Boyd-Graber
> *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2025), Volume 2: Short Papers, pages 65-74*
>
> Paper: https://aclanthology.org/2025.naacl-short.6/

## Overview

PHOLUS is a natural language advisor built on top of [Cicero](https://github.com/ALLAN-DIP/diplomacy_cicero/), a superhuman Diplomacy-playing AI. Unlike Cicero, which plays the game directly, PHOLUS passively observes the game and provides **move advice** and **message advice** to human players in real time. This repository contains the analysis code and collected data from a user study with 12 Diplomacy games, 41 players, 1,070 player turns, and 117 playing hours.

### Key Findings

1. **Advice helps novices more than veterans.** Novice players with PHOLUS advice can compete with — and sometimes surpass — experienced players. Novices accept move advice ~32.6% of the time vs. ~6.4% for veterans.
2. **Move advice is more helpful than message advice.** Move advice has a positive correlation with supply center gains. Message-only advice can negatively affect outcomes. Combined move + message advice yields the greatest benefit.
3. **Players do not blindly follow advice.** Even novices synthesize PHOLUS's suggestions with their own strategy. Average move agreement drops ~10% from start to end of turn as players refine their decisions.
4. **Just reading advice helps.** Even when players do not directly follow the advice, exposure to PHOLUS's strategic reasoning can positively inform their choices.

## Repository Structure

```
Pholus/
├── main.py                  # Entry point (placeholder launcher)
├── pyproject.toml           # Project metadata and dependencies
├── data/
│   ├── games/
│   │   ├── old_payload_format/   # Centaur game JSONs (PHOLUS-assisted, 12 games)
│   │   ├── new_payload_format/   # CiceroAlbert game JSONs (Cicero + Albert bots)
│   │   └── old_games/            # Baseline AIGame JSONs (Cicero vs. humans, no PHOLUS)
│   ├── amr_centaur_games/        # AMR-parsed Centaur game data for SMATCH analysis
│   ├── players.json              # Player metadata
│   ├── qualitative.json          # Qualitative analysis examples
│   └── scs_delta_data.json       # Pre-computed feature matrix for regression
├── src/
│   ├── constants.py         # Game IDs, power lists, player-type mappings
│   ├── agreement.py         # Move agreement/equivalence analysis (novice focus)
│   ├── centaur_stats.py     # Advice acceptance stats (novice vs. veteran, Table 1)
│   ├── extraction.py        # Baseline game data extraction and analysis utilities
│   ├── process_centaur.py   # Detailed per-phase advice processing and visualization
│   ├── figures.py           # Figure 2 generation (regression coefficient plot)
│   └── lin_reg.py           # Ridge regression analysis with bootstrap CIs
└── results/
    └── smatch_results.json  # SMATCH scores for message advice similarity
```

## Setup

Requires Python 3.11+. Install dependencies using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with pip:

```bash
pip install matplotlib numpy pandas plotnine scikit-learn
```

## Usage

### Regression Analysis (Figure 2 / Figure 3)

Generate the regression coefficient plots from the paper:

```bash
python -m src.figures
# or
python -m src.lin_reg
```

This fits a Ridge regression model predicting per-turn supply center changes from features including player experience, assigned power, and advice setting. Output is saved to `results/regression_coefs.pdf`.

### Advice Acceptance Statistics (Table 1)

Compute move and message advice acceptance rates split by player type:

```bash
python -m src.centaur_stats
```

### Move Agreement Analysis

Analyze the overlap between PHOLUS's suggested moves and players' actual moves:

```bash
python -m src.agreement
```

### Baseline Game Analysis

Extract and analyze data from the non-PHOLUS Cicero vs. human games:

```bash
python -m src.extraction
```

### Centaur Game Processing

Run the full advice processing pipeline with per-phase visualizations:

```bash
python -m src.process_centaur
```

## Data Format

Each game JSON file contains:

| Field | Description |
|---|---|
| `game_id` | Unique game identifier (e.g., `Centaur0`, `AIGame_0`) |
| `state_history` | Board state per phase (units, centers, influence) |
| `order_history` | Final submitted orders per power per phase |
| `message_history` | All messages including PHOLUS advice (type: `suggested_move_full`, `suggested_move_partial`, `suggested_message`) |
| `order_log_history` | Timestamped log of order additions/removals |
| `annotated_messages` | Player annotations (accept/reject) for PHOLUS advice |
| `is_bot_history` | Player perceptions of which powers are bots |
| `stance_history` | Player-reported friendliness stances toward other powers |

### Advice Types in Message History

PHOLUS advice entries use `sender: "omniscient_type"` and are categorized by `type`:

- `has_suggestions` — Indicates which advice types were offered (1=message, 2=move, 3=both)
- `suggested_move_full` — Complete set of move orders for a power
- `suggested_move_partial` — Individual unit move suggestion
- `suggested_message` — Suggested message content and recipient

## Experiment Design

Each game involved 2-5 human players alongside Cicero bots. Per turn, each player was randomly assigned one of four advice settings:

1. **No advice** — No assistance from PHOLUS
2. **Message advice** — PHOLUS suggests message recipients and content
3. **Move advice** — PHOLUS recommends unit orders
4. **Message + move advice** — Both types combined

Players placed initial orders before communicating, then PHOLUS recomputed advice as new messages were sent during the turn.

## Evaluation Metrics

- **Supply center delta**: Net gain/loss of supply centers per turn (regression target)
- **Move agreement**: Proportion of overlap between player moves and PHOLUS advice — `A = |player_moves ∩ advice_moves| / |player_moves|`
- **Move equivalence**: Whether the player's move set exactly matches PHOLUS advice — `E = 1 if player_moves == advice_moves`
- **SMATCH score**: Semantic similarity between PHOLUS message advice and player-sent messages via Abstract Meaning Representation parsing

## Citation

```bibtex
@inproceedings{gu-etal-2025-personalized,
    title = "Personalized Help for Optimizing Low-Skilled Users' Strategy",
    author = "Gu, Feng and Wongkamjan, Wichayaporn and Kummerfeld, Jonathan K. and Peskoff, Denis and May, Jonathan and Boyd-Graber, Jordan Lee",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers)",
    year = "2025",
    pages = "65--74",
    url = "https://aclanthology.org/2025.naacl-short.6/",
}
```

## License

See the original [Cicero repository](https://github.com/ALLAN-DIP/diplomacy_cicero/) for code licensing details.
