"""
Centaur game statistics: advice acceptance rates for novice vs. experienced players.

Computes acceptance statistics for both move and message advice across all
Centaur (PHOLUS-assisted) games, broken down by player type (novice vs.
experienced/veteran). Also tracks move agreement between initial suggestions
and final submitted orders, producing the statistics reported in Table 1
and Section 3.1 of the paper.
"""

import json
import os

from src.constants import CENTAUR_PATH, HUMAN_PLAYERS, NOVICES, POWERS

# Collects qualitative examples of cases where experienced players accepted
# PHOLUS's move advice but had varied message acceptance.
qualitative = []


def get_all_games():
    """Load all Centaur game JSON files from the data directory."""
    games = []
    for filename in os.listdir(CENTAUR_PATH):
        if filename.endswith(".json"):
            with open(f"{CENTAUR_PATH}/{filename}", "r") as f:
                games.append(json.load(f))

    return games


def get_phases(game):
    """Reconstruct a list of phase dicts (state, messages, orders) from game history."""
    phases = []
    for phase_name in game["state_history"]:
        phases.append(
            {
                "name": phase_name,
                "state": game["state_history"][phase_name],
                "messages": game["message_history"].get(phase_name, []),
                "orders": game["order_history"].get(phase_name, {}),
                "order_logs": game["order_log_history"].get(phase_name, {}),
            }
        )
    return phases


def get_initial_moves(locations: list, power: str, order_logs: dict) -> list[str]:
    """
    Extract the initial move orders placed by a power before any communication.

    Parses the order log to find the first 'added' order for each unit location,
    giving us the player's initial strategic intent before negotiation.
    """
    orders = {}

    for _, v in order_logs.items():
        if len(locations) == len(orders):
            break
        if power not in v or "added" not in v:
            continue

        _, _, *order_parts = v.split(" ")
        unit_location = order_parts[1]
        assert unit_location in locations, (
            f"Unit location {unit_location} not in {locations}"
        )
        if unit_location not in orders:
            orders[unit_location] = " ".join(order_parts)

    return list(orders.values())


def get_move_suggestions(
    message_history: list, power: str, initial: bool = True
) -> list[str]:
    """
    Retrieve PHOLUS's full move suggestions for a given power.

    Returns the earliest (initial=True) or latest (initial=False) set of
    suggested moves from the omniscient advisor during the phase.
    """
    full_move_suggestions = [
        x
        for x in message_history
        if x["type"] == "suggested_move_full"
        and x["sender"] == "omniscient_type"
        and x["recipient"] == "GLOBAL"
        and x["message"].startswith(power)
    ]

    timestamp = (
        min([x["timestamp"] for x in full_move_suggestions])
        if initial
        else max([x["timestamp"] for x in full_move_suggestions])
    )

    for s in full_move_suggestions:
        if s["timestamp"] == timestamp:
            splitted = s["message"].split(":")[1]
            orders = splitted.split(" ")
            orders = [x.strip() for x in orders]
            return orders

    raise ValueError("No move suggestions found")


games = get_all_games()

# Global counters for the user study
total_player_turns = 0
total_hours = 0

# Lie detection stats: lies directed at humans vs. bots
lies2humans, total2humans = 0, 0
lies2bots, total2bots = 0, 0

# Move and message advice acceptance counters, split by player type.
# These produce the statistics in Table 1 of the paper.
novice_move_accept, novice_move_all_suggestions = 0, 0
novice_message_accept, novice_message_all = 0, 0
experienced_move_accept, experienced_move_all_suggestions = 0, 0
experienced_message_accept, experienced_message_all = 0, 0

# Initial suggestion acceptance (first suggestion shown in a turn)
novice_intial_accept, novice_initial_all = 0, 0
experienced_initial_accept, experienced_initial_all = 0, 0

# Move agreement: overlap between player's final moves and PHOLUS's advice
novice_agree, novice_all = 0, 0
experienced_agree, experienced_all = 0, 0

# Novice equivalence tracking: how often initial/final suggestions match
# the player's final submitted moves (set-level and move-level)
novice_initial_final_same, novice_initial_final_all = 0, 0
novice_initial_final_moves_same, novice_initial_final_moves_all = 0, 0
novice_final_suggestion, novice_final_suggestion_moves = 0, 0


for game in games:
    phases = get_phases(game)
    id = game["game_id"]
    humans = len(HUMAN_PLAYERS.get(id, []) + NOVICES.get(id, []))
    total_player_turns += (len(phases) - 1) * humans
    total_hours += humans * 3
    last_phase = phases[-1]
    max_units = max([len(last_phase["state"]["units"][power]) for power in POWERS])
    max_unit_powers = [
        power
        for power in POWERS
        if len(last_phase["state"]["units"][power]) == max_units
    ]
    max_unit_players = []

    for pp in max_unit_powers:
        if pp in HUMAN_PLAYERS.get(id, []):
            max_unit_players.append("human")
        elif pp in NOVICES.get(id, []):
            max_unit_players.append("novice")
        else:
            max_unit_players.append("bot")
    # print(f"Game {id}: {max_unit_players}")
    for phase in phases:
        messages = [x for x in phase["messages"] if x["truth"]]
        for m in messages:
            if m["recipient"] in HUMAN_PLAYERS.get(id, []) + NOVICES.get(id, []):
                total2humans += 1
                if m["truth"] == "Lie":
                    lies2humans += 1
            else:
                total2bots += 1
                if m["truth"] == "Lie":
                    lies2bots += 1

        suggested_moves = [
            x for x in phase["messages"] if x["type"] == "suggested_move_full"
        ]
        suggested_messages = [
            x for x in phase["messages"] if x["type"] == "suggested_message"
        ]
        for sm in suggested_messages:
            power = sm["message"].split("-")[0]
            if power in HUMAN_PLAYERS.get(id, []):
                experienced_message_all += 1
                if (
                    str(sm["time_sent"]) in game["annotated_messages"]
                    and game["annotated_messages"][str(sm["time_sent"])] == "accept"
                ):
                    experienced_message_accept += 1
            elif power in NOVICES.get(id, []):
                novice_message_all += 1
                if (
                    str(sm["time_sent"]) in game["annotated_messages"]
                    and game["annotated_messages"][str(sm["time_sent"])] == "accept"
                ):
                    novice_message_accept += 1

            else:
                continue

        for hp in HUMAN_PLAYERS.get(id, []):
            locs = phase["state"]["units"][hp]
            order_num = len(locs)
            if phase["name"][-1] != "M":
                continue
            initial_moves = sorted(
                get_initial_moves(
                    [x.split(" ")[1] for x in locs], hp, phase["order_logs"]
                )
            )

            suggested_moves_hp = [
                x
                for x in phase["messages"]
                if (
                    x["type"] == "suggested_move_full"
                    or x["type"] == "suggested_move_partial"
                )
                and x["message"].startswith(hp)
            ]

            full = [
                x
                for x in phase["messages"]
                if (x["type"] == "suggested_move_full") and x["message"].startswith(hp)
            ]

            if len(full) > 0:
                min_time = str(min([x["time_sent"] for x in full]))
                min_suggestions = sorted(
                    [x for x in full if x["time_sent"] == int(min_time)][0]["message"]
                    .split(":")[1]
                    .split(", ")
                )
                if min_suggestions == initial_moves:
                    suggested_messages = [
                        x
                        for x in phase["messages"]
                        if x["type"] == "suggested_message"
                        and x["message"].startswith(hp)
                    ]

                    if len(suggested_messages) > 0:
                        print(f"game {id} phase {phase['name']} player {hp}")
                        actual_message = [
                            x["sender"] + "-" + x["recipient"] + ": " + x["message"]
                            for x in phase["messages"]
                            if x["sender"] == hp or x["recipient"] == hp
                        ]
                        num_accepted = len(
                            [
                                x
                                for x in suggested_messages
                                if str(x["time_sent"]) in game["annotated_messages"]
                                and game["annotated_messages"][str(x["time_sent"])]
                                == "accept"
                            ]
                        )
                        percent_accepted = (
                            str(num_accepted / len(suggested_messages) * 100) + "%"
                        )
                        if sorted(phase["orders"][hp]) == initial_moves:
                            item = {
                                "game": id,
                                "phase": phase["name"],
                                "player": hp,
                                "initial_moves": initial_moves,
                                "final_moves": sorted(phase["orders"][hp]),
                                "message acceptance": percent_accepted,
                                "suggested_messages": [
                                    x["message"] for x in suggested_messages
                                ],
                                "actual_message": actual_message,
                            }
                            qualitative.append(item)
                else:
                    pass
                    # print(min_suggestions, initial_moves)

                experienced_initial_all += order_num
                if (
                    min_time in game["annotated_messages"]
                    and game["annotated_messages"][min_time] == "accept all"
                ):
                    experienced_initial_accept += order_num
                elif (
                    min_time in game["annotated_messages"]
                    and "accept" in game["annotated_messages"][min_time]
                ):
                    experienced_initial_accept += 1

            for sm in suggested_moves_hp:
                num_s = len(sm["message"].split(", "))
                experienced_move_all_suggestions += num_s
                if (
                    str(sm["time_sent"]) in game["annotated_messages"]
                    and game["annotated_messages"][str(sm["time_sent"])] == "accept all"
                ):
                    experienced_move_accept += num_s
                elif (
                    str(sm["time_sent"]) in game["annotated_messages"]
                    and "accept" in game["annotated_messages"][str(sm["time_sent"])]
                ):
                    experienced_move_accept += 1

        for hp in NOVICES.get(id, []):
            order_num = len(phase["state"]["units"][hp])
            suggested_moves_hp = [
                x
                for x in phase["messages"]
                if (
                    x["type"] == "suggested_move_full"
                    or x["type"] == "suggested_move_partial"
                )
                and x["message"].startswith(hp)
            ]
            full = [
                x
                for x in phase["messages"]
                if (x["type"] == "suggested_move_full") and x["message"].startswith(hp)
            ]

            if len(full) > 0:
                min_time = str(min([x["time_sent"] for x in full]))
                min_suggestions = sorted(
                    [x for x in full if x["time_sent"] == int(min_time)][0]["message"]
                    .split(":")[1]
                    .split(", ")
                )
                max_time = max([x["time_sent"] for x in full])
                max_suggestions = sorted(
                    [x for x in full if x["time_sent"] == max_time][0]["message"]
                    .split(":")[1]
                    .split(", ")
                )

                novice_initial_all += len(min_suggestions)

                final_moves = sorted(phase["orders"][hp])
                novice_initial_final_all += 1
                novice_initial_final_moves_all += len(final_moves)
                if min_suggestions == final_moves:
                    novice_initial_final_same += 1

                if max_suggestions == final_moves:
                    novice_final_suggestion += 1

                novice_initial_final_moves_same += len(
                    set(min_suggestions) & set(final_moves)
                )

                novice_final_suggestion_moves += len(
                    set(max_suggestions) & set(final_moves)
                )

                if (
                    min_time in game["annotated_messages"]
                    and game["annotated_messages"][min_time] == "accept all"
                ):
                    novice_intial_accept += order_num
                elif (
                    min_time in game["annotated_messages"]
                    and "accept" in game["annotated_messages"][min_time]
                ):
                    novice_intial_accept += 1

            for sm in suggested_moves_hp:
                num_s = len(sm["message"].split(", "))
                novice_move_all_suggestions += num_s
                if (
                    str(sm["time_sent"]) in game["annotated_messages"]
                    and game["annotated_messages"][str(sm["time_sent"])] == "accept all"
                ):
                    novice_move_accept += num_s
                elif (
                    str(sm["time_sent"]) in game["annotated_messages"]
                    and "accept" in game["annotated_messages"][str(sm["time_sent"])]
                ):
                    novice_move_accept += 1


# print(f"Total player turns: {total_player_turns}")
# print(f"Total hours: {total_hours}")

# print(f"Lies to humans: {lies2humans}/{total2humans}")
# print(f"Lies to bots: {lies2bots}/{total2bots}")
print(
    f"Experienced message accept: {experienced_message_accept}/{experienced_message_all}"
)
print(f"Novice message accept: {novice_message_accept}/{novice_message_all}")

print(
    f"Experienced move accept: {experienced_move_accept}/{experienced_move_all_suggestions}"
)

print(f"Novice move accept: {novice_move_accept}/{novice_move_all_suggestions}")

print(
    f"Experienced initial accept: {experienced_initial_accept}/{experienced_initial_all}"
)

print(f"Novice initial accept: {novice_intial_accept}/{novice_initial_all}")

print(
    f"Novice initial final set agreement: {novice_initial_final_same}/{novice_initial_final_all}"
)
print(
    f"Novice initial final agreement: {novice_initial_final_moves_same}/{novice_initial_final_moves_all}"
)
print(
    f"Novice final set agreement: {novice_final_suggestion}/{novice_initial_final_all}"
)
print(
    f"Novice final moves agreement: {novice_final_suggestion_moves}/{novice_initial_final_moves_all}"
)
