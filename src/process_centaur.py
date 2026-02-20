"""
Centaur game processing: detailed advice acceptance analysis and visualization.

Processes the Centaur (PHOLUS-assisted) game data to compute per-phase and
per-unit statistics on move and message advice acceptance. Generates stacked
bar charts and percentage plots showing how advice acceptance varies across
game phases. Includes separate analysis pipelines for experienced players
and novice players.

This module supports the quantitative results in Section 3.1 and Table 1
of the paper.
"""

import json
import os

from matplotlib import pyplot as plt

from src.constants import CENTAUR_PATH, HUMAN_PLAYERS, NOVICES, POWERS

# Powers excluded from Cicero classification due to disconnects or
# non-standard control (e.g., taken over by dummy bots mid-game).
EXCLUSION = {
    "Centaur5": ["RUSSIA"],
    "Centaur10": ["GERMANY"],
    "Centaur11": ["TURKEY"],
}


def check_all_human_powers() -> bool:
    """Verify that all entries in HUMAN_PLAYERS map to valid Diplomacy powers."""
    human_by_game = []
    for pp in HUMAN_PLAYERS.values():
        human_by_game.append(all([x in POWERS for x in pp]))

    return all(human_by_game)


def check_all_game_names(game_ids: list) -> bool:
    """Verify that all loaded game IDs appear in the HUMAN_PLAYERS mapping."""
    return all([x in HUMAN_PLAYERS.keys() for x in game_ids])


def get_all_games():
    """Load all Centaur game JSON files from the data directory."""
    games = []
    for filename in os.listdir(CENTAUR_PATH):
        if filename.endswith(".json"):
            with open(f"{CENTAUR_PATH}/{filename}", "r") as f:
                games.append(json.load(f))

    return games


def get_initial_moves(locations: list, power: str, order_logs: dict) -> list[str]:
    """Extract the first orders placed by a power for each unit location."""
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
    if unit_location not in locations:
        orders[unit_location] = " ".join(order_parts)

    return list(orders.keys())


def get_suggestion_type(power: str, message_history: list) -> int:
    """
    0: no advice
    1: message
    2: move
    3: both
    """
    suggested_messages = [
        x
        for x in message_history
        if x["type"] == "suggested_message" and x["message"].startswith(power)
    ]
    suggested_moves = [
        x
        for x in message_history
        if x["type"] == "suggested_move_full" and x["message"].startswith(power)
    ]
    if len(suggested_moves) > 0 and len(suggested_messages) > 0:
        return 3
    elif len(suggested_moves) > 0:
        return 2
    elif len(suggested_messages) > 0:
        return 1

    return 0


def get_move_suggestions(
    message_history: list, power: str, initial: bool = True
) -> list[str]:
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


def human_players(game: dict) -> list:
    """Return the list of powers controlled by experienced human players in this game."""
    return HUMAN_PLAYERS[game["game_id"]]


def cicero_players(game: dict) -> list:
    """Return the list of powers controlled by Cicero (AI) in this game."""
    id = game["game_id"]
    excluded = EXCLUSION.get(id, [])
    humans = human_players(game)
    return [x for x in POWERS if x not in humans and x not in excluded]


def get_annotations(game: dict) -> dict:
    """Return player annotations (accept/reject) for advice, keyed by integer timestamp."""
    annotations = game["annotated_messages"]
    result = {}

    for k, v in annotations.items():
        result[int(k)] = v

    return result


def get_phases(game: dict) -> list:
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


def get_phase_messages(phase: dict) -> list:
    """Return all messages from a phase dict."""
    return phase["messages"]


def get_system_messages(messages: list) -> list:
    """Filter for PHOLUS system messages (omniscient_type sender, GLOBAL recipient)."""
    return [
        m
        for m in messages
        if m["sender"] == "omniscient_type" and m["recipient"] == "GLOBAL"
    ]


def get_suggestion_types(system_messages: list) -> list:
    """Filter for messages indicating which type of advice was offered to a player."""
    return [m for m in system_messages if m["type"] == "has_suggestions"]


def get_suggested_moves_full(system_messages: list) -> list:
    """Filter for complete move suggestion sets from PHOLUS."""
    return [m for m in system_messages if m["type"] == "suggested_move_full"]


def get_suggested_moves_partial(system_messages: list) -> list:
    """Filter for partial (individual unit) move suggestions from PHOLUS."""
    return [m for m in system_messages if m["type"] == "suggested_move_partial"]


def get_all_suggested_moves(system_messages: list) -> list:
    """
    returns both full and partial move suggestions
    """
    return get_suggested_moves_full(system_messages) + get_suggested_moves_partial(
        system_messages
    )


def get_orderable_units(phase: dict, power: str) -> list:
    """Return the list of units a power can order in the given phase."""
    return phase["state"]["units"][power]


def get_suggested_messages(system_messages: list) -> list:
    """Filter for message suggestions from PHOLUS."""
    return [m for m in system_messages if m["type"] == "suggested_message"]


def draw_move_advice_by_unit(acc_moves: dict):
    """Plot a stacked bar chart of accepted vs. rejected move advice per phase."""
    while acc_moves["accepted"][-1] == 0 and acc_moves["other"][-1] == 0:
        acc_moves["accepted"].pop()
        acc_moves["other"].pop()

    # stacked bar for acc_moves
    fig, ax = plt.subplots()
    ax.bar(
        range(len(acc_moves["accepted"])),
        acc_moves["accepted"],
        label="Accepted",
        color="green",
        align="center",
    )
    ax.bar(
        range(len(acc_moves["other"])),
        acc_moves["other"],
        bottom=acc_moves["accepted"],
        label="Other",
        color="red",
        align="center",
    )
    ax.set_ylabel("#advice")
    ax.set_xlabel("Movement Phase")
    ax.set_title("Move advice by phase")
    ax.legend()
    plt.show()


def draw_message_advice_by_phase(acc_msgs: dict):
    """Plot a stacked bar chart of accepted vs. rejected message advice per phase."""
    while acc_msgs["accepted"][-1] == 0 and acc_msgs["other"][-1] == 0:
        acc_msgs["accepted"].pop()
        acc_msgs["other"].pop()

    fig, ax = plt.subplots()
    ax.bar(
        range(len(acc_msgs["accepted"])),
        acc_msgs["accepted"],
        label="Accepted",
        color="green",
        align="center",
    )
    ax.bar(
        range(len(acc_msgs["other"])),
        acc_msgs["other"],
        bottom=acc_msgs["accepted"],
        label="Other",
        color="red",
        align="center",
    )
    ax.set_ylabel("#advice")
    ax.set_xlabel("Movement Phase")
    ax.set_title("Message advice by phase")
    ax.legend()
    plt.show()


def draw_moves_percentage_by_unit(acc_moves: dict):
    while acc_moves["accepted"][-1] == 0 and acc_moves["other"][-1] == 0:
        acc_moves["accepted"].pop()
        acc_moves["other"].pop()

    # stacked bar for acc_moves
    fig, ax = plt.subplots()

    percentage = [
        x / (x + y) for x, y in zip(acc_moves["accepted"], acc_moves["other"])
    ]

    ax.bar(
        range(len(percentage)),
        percentage,
        color="blue",
        align="center",
    )
    ax.set_ylabel("Percentage of accepted advice")
    ax.set_xlabel("Movement Phase")
    ax.set_title("Move advice by phase")
    plt.show()


def draw_messages_percentage_by_phase(acc_msgs: dict):
    while acc_msgs["accepted"][-1] == 0 and acc_msgs["other"][-1] == 0:
        acc_msgs["accepted"].pop()
        acc_msgs["other"].pop()

    # stacked bar for acc_moves
    fig, ax = plt.subplots()

    percentage = [x / (x + y) for x, y in zip(acc_msgs["accepted"], acc_msgs["other"])]

    ax.bar(
        range(len(percentage)),
        percentage,
        color="blue",
        align="center",
    )
    ax.set_ylabel("Percentage of accepted advice")
    ax.set_xlabel("Movement Phase")
    ax.set_title("Message advice by phase")
    plt.show()


def get_percentage_accepted_moves(acc_moves: dict):
    accepted_moves, partial_moves, total_moves = 0, 0, 0

    for i in range(len(acc_moves["accepted"])):
        accepted_moves += acc_moves["accepted"][i]
        partial_moves += acc_moves["partial"][i]
        total_moves += (
            acc_moves["accepted"][i] + acc_moves["other"][i] + acc_moves["partial"][i]
        )

    print(accepted_moves, partial_moves, total_moves)
    return accepted_moves / total_moves


def get_percentage_accepted_messages(acc_msgs: dict):
    accepted_messages, total_messages = 0, 0

    for i in range(len(acc_msgs["accepted"])):
        accepted_messages += acc_msgs["accepted"][i]
        total_messages += acc_msgs["accepted"][i] + acc_msgs["other"][i]

    print(accepted_messages, total_messages)
    return accepted_messages / total_messages


def first_outgoing_message(msgs: list, power: str) -> int:
    """
    returns the timestamp of the first message sent by the given power
    """
    min_timestamp = float("inf")
    for mm in msgs:
        if mm["sender"] == power and mm["time_sent"] < min_timestamp:
            min_timestamp = mm["time_sent"]

    return -1 if min_timestamp == float("inf") else int(min_timestamp)


def submitted_moves(phase: dict, power: str) -> list:
    """Return the final submitted orders for a power in the given phase."""
    return phase["orders"][power]


def initial_moves(phase: dict, power: str) -> dict:
    """Extract orders placed before the player's first outgoing message (pre-negotiation intent)."""
    msgs = phase["messages"]
    units = phase["state"]["units"][power]

    initial_msg_timestamp = first_outgoing_message(msgs, power)
    if initial_msg_timestamp == -1:
        return {}

    orders = {}
    order_logs = phase["order_logs"]
    for timestamp, order in order_logs.items():
        int_timestamp = int(timestamp)
        if int_timestamp < initial_msg_timestamp and "add" in order and power in order:
            oo = order.split(" added: ")[1]
            orders[oo] = int_timestamp

    # process order updates
    final_initial_orders = {}
    for unit in units:
        unit_orders = [x for x in orders.keys() if unit == " ".join(x.split(" ")[:2])]

        latest_timestamp = -1
        for order in unit_orders:
            if orders[order] > latest_timestamp:
                latest_timestamp = orders[order]
                final_initial_orders[unit] = order

    return final_initial_orders


def main():
    """Main pipeline: compute advice acceptance stats for all Centaur games."""
    assert check_all_human_powers(), "Not all human players are in the powers list"
    games = get_all_games()
    game_ids = [x["game_id"] for x in games]
    assert check_all_game_names(game_ids), "Not all game names are in the list"

    total_move_advice_by_unit = {}
    total_message_advice_by_phase = {}
    max_phase_len = 0

    all_accepted_messages = []
    all_other_messages = []

    total_message_collected = 0
    total_moves_advice_collected = 0
    total_message_advice_collected = 0
    total_message_advice_accepted = 0
    total_turns = 0
    total_player_turns = {
        "none": 0,
        "move": 0,
        "message": 0,
        "both": 0,
        "total": 0,
    }
    total_partial_accepted = 0
    total_moves_accepted = 0
    total_sept_turns = 0

    for gg in games:
        print(f"Processing game {gg['game_id']}")

        annotations = get_annotations(gg)
        phases = get_phases(gg)
        total_turns += len(phases)
        humans = human_players(gg)
        total_player_turns["total"] += len(humans) * len(phases)
        if gg["game_id"] in NOVICES:
            total_player_turns["total"] += len(NOVICES[gg["game_id"]]) * len(phases)
            total_sept_turns += len(NOVICES[gg["game_id"]]) * len(phases)
            total_sept_turns += len(humans) * len(phases)

        if len(phases) > max_phase_len:
            max_phase_len = len(phases)

        game_move_advice_by_unit = {
            "accepted": [],
            "partial": [],
            "other": [],
        }

        game_message_advice_by_phase = {
            "accepted": [],
            "other": [],
        }

        for pp in phases:
            if "M" not in pp["name"]:
                continue

            print(f"Processing phase {pp['name']}")

            msgs = get_phase_messages(pp)
            system_msgs = get_system_messages(msgs)

            total_message_collected += len(msgs) - len(system_msgs)

            move_suggestions = get_all_suggested_moves(system_msgs)
            message_suggestions = get_suggested_messages(system_msgs)

            total_message_advice_collected += len(message_suggestions)
            message_advice_accepted = [
                x
                for x in message_suggestions
                if x["time_sent"] in annotations
                and annotations[x["time_sent"]] == "accept"
            ]
            total_message_advice_accepted += len(message_advice_accepted)

            cnt_moves_accepted = len(
                [
                    x
                    for x in move_suggestions
                    if x["time_sent"] in annotations
                    and "all" in annotations[x["time_sent"]]
                ]
            )
            total_moves_accepted += cnt_moves_accepted

            total_partial_accepted += (
                len(
                    [
                        x
                        for x in move_suggestions
                        if x["time_sent"] in annotations
                        and "accept" in annotations[x["time_sent"]]
                    ]
                )
                - cnt_moves_accepted
            )

            total_moves_advice_collected += len(move_suggestions)

            message_accepted = 0

            # calculate how many message are accepted
            for mm in message_suggestions:
                if (
                    mm["time_sent"] in annotations
                    and annotations[mm["time_sent"]] == "accept"
                ):
                    message_accepted += 1
                    all_accepted_messages.append(mm["message"])
                else:
                    all_other_messages.append(mm["message"])

            game_message_advice_by_phase["accepted"].append(message_accepted)
            game_message_advice_by_phase["other"].append(
                len(message_suggestions) - message_accepted
            )

            # calculate how many moves are accepted
            for human in humans:
                human_msg_suggestions = [
                    x for x in message_suggestions if x["message"].startswith(human)
                ]

                human_move_suggestions = [
                    x for x in move_suggestions if human in x["message"]
                ]

                if len(human_move_suggestions) == 0 and len(human_msg_suggestions) == 0:
                    total_player_turns["none"] += 1

                elif (
                    len(human_move_suggestions) > 0 and len(human_msg_suggestions) == 0
                ):
                    total_player_turns["move"] += 1

                elif (
                    len(human_move_suggestions) == 0 and len(human_msg_suggestions) > 0
                ):
                    total_player_turns["message"] += 1
                else:
                    total_player_turns["both"] += 1

        # sum over all phases
        total_move_advice_by_unit[gg["game_id"]] = game_move_advice_by_unit
        total_message_advice_by_phase[gg["game_id"]] = game_message_advice_by_phase

    acc_moves = {
        "accepted": [0] * max_phase_len,
        "other": [0] * max_phase_len,
        "partial": [0] * max_phase_len,
    }
    acc_messages = {
        "accepted": [0] * max_phase_len,
        "other": [0] * max_phase_len,
    }

    for gg in games:
        for idx, val in enumerate(total_move_advice_by_unit[gg["game_id"]]["accepted"]):
            acc_moves["accepted"][idx] += val
        for idx, val in enumerate(total_move_advice_by_unit[gg["game_id"]]["other"]):
            acc_moves["other"][idx] += val
        for idx, val in enumerate(total_move_advice_by_unit[gg["game_id"]]["partial"]):
            acc_moves["partial"][idx] += val

        for idx, val in enumerate(
            total_message_advice_by_phase[gg["game_id"]]["accepted"]
        ):
            acc_messages["accepted"][idx] += val
        for idx, val in enumerate(
            total_message_advice_by_phase[gg["game_id"]]["other"]
        ):
            acc_messages["other"][idx] += val

    # print(acc_moves)
    # print(acc_messages)
    # print(get_percentage_accepted_moves(acc_moves) * 100)
    # print(get_percentage_accepted_messages(acc_messages) * 100)

    # draw_move_advice_by_unit(acc_moves)
    # draw_message_advice_by_phase(acc_messages)
    # draw_moves_percentage_by_unit(acc_moves)
    # draw_messages_percentage_by_phase(acc_messages)

    # print(total_message_collected)
    print(total_moves_accepted, total_partial_accepted, total_moves_advice_collected)
    print(total_message_advice_accepted, total_message_advice_collected)
    # print(total_turns)
    # print(total_player_turns)
    print(total_player_turns, total_sept_turns)

    total_move_advice_by_unit = {}
    total_message_advice_by_phase = {}
    max_phase_len = 0

    all_accepted_messages = []
    all_other_messages = []

    total_message_collected = 0
    total_moves_advice_collected = 0
    total_message_advice_collected = 0
    total_message_advice_accepted = 0
    total_turns = 0
    total_player_turns = {
        "none": 0,
        "move": 0,
        "message": 0,
        "both": 0,
        "total": 0,
    }
    total_partial_accepted = 0
    total_moves_accepted = 0

    # ---- Separate analysis for novice players only (games 8 & 9) ----
    for gg in games:
        id = gg["game_id"]
        if id == "Centaur8" or id == "Centaur9":
            annotations = get_annotations(gg)
            phases = get_phases(gg)
            novices = NOVICES[id]

            if len(phases) > max_phase_len:
                max_phase_len = len(phases)

            game_move_advice_by_unit = {
                "accepted": [],
                "partial": [],
                "other": [],
            }

            game_message_advice_by_phase = {
                "accepted": [],
                "other": [],
            }

            for pp in phases:
                if "M" not in pp["name"]:
                    continue

                msgs = get_phase_messages(pp)
                system_msgs = get_system_messages(msgs)

                total_message_collected += len(msgs) - len(system_msgs)

                move_suggestions = get_all_suggested_moves(system_msgs)
                message_suggestions = get_suggested_messages(system_msgs)

                total_message_advice_collected += len(message_suggestions)
                message_advice_accepted = [
                    x
                    for x in message_suggestions
                    if x["time_sent"] in annotations
                    and annotations[x["time_sent"]] == "accept"
                ]
                total_message_advice_accepted += len(message_advice_accepted)

                cnt_moves_accepted = len(
                    [
                        x
                        for x in move_suggestions
                        if x["time_sent"] in annotations
                        and "all" in annotations[x["time_sent"]]
                    ]
                )
                total_moves_accepted += cnt_moves_accepted

                total_partial_accepted += (
                    len(
                        [
                            x
                            for x in move_suggestions
                            if x["time_sent"] in annotations
                            and "accept" in annotations[x["time_sent"]]
                        ]
                    )
                    - cnt_moves_accepted
                )

                total_moves_advice_collected += len(move_suggestions)

                message_accepted = 0

                # calculate how many message are accepted
                for mm in message_suggestions:
                    if (
                        mm["time_sent"] in annotations
                        and annotations[mm["time_sent"]] == "accept"
                    ):
                        message_accepted += 1
                        all_accepted_messages.append(mm["message"])
                    else:
                        all_other_messages.append(mm["message"])

                game_message_advice_by_phase["accepted"].append(message_accepted)
                game_message_advice_by_phase["other"].append(
                    len(message_suggestions) - message_accepted
                )

                # calculate how many moves are accepted
                for human in novices:
                    human_msg_suggestions = [
                        x for x in message_suggestions if x["message"].startswith(human)
                    ]

                    human_move_suggestions = [
                        x for x in move_suggestions if human in x["message"]
                    ]

                    if (
                        len(human_move_suggestions) == 0
                        and len(human_msg_suggestions) == 0
                    ):
                        total_player_turns["none"] += 1

                    elif (
                        len(human_move_suggestions) > 0
                        and len(human_msg_suggestions) == 0
                    ):
                        total_player_turns["move"] += 1

                    elif (
                        len(human_move_suggestions) == 0
                        and len(human_msg_suggestions) > 0
                    ):
                        total_player_turns["message"] += 1
                    else:
                        total_player_turns["both"] += 1

            # sum over all phases
            total_move_advice_by_unit[gg["game_id"]] = game_move_advice_by_unit
            total_message_advice_by_phase[gg["game_id"]] = game_message_advice_by_phase

    acc_moves = {
        "accepted": [0] * max_phase_len,
        "other": [0] * max_phase_len,
        "partial": [0] * max_phase_len,
    }
    acc_messages = {
        "accepted": [0] * max_phase_len,
        "other": [0] * max_phase_len,
    }

    print("moves", total_move_advice_by_unit)

    for gg in games:
        id = gg["game_id"]
        if id == "Centaur8" or id == "Centaur9":
            for idx, val in enumerate(
                total_move_advice_by_unit[gg["game_id"]]["accepted"]
            ):
                acc_moves["accepted"][idx] += val
            for idx, val in enumerate(
                total_move_advice_by_unit[gg["game_id"]]["other"]
            ):
                acc_moves["other"][idx] += val
            for idx, val in enumerate(
                total_move_advice_by_unit[gg["game_id"]]["partial"]
            ):
                acc_moves["partial"][idx] += val

            for idx, val in enumerate(
                total_message_advice_by_phase[gg["game_id"]]["accepted"]
            ):
                acc_messages["accepted"][idx] += val
            for idx, val in enumerate(
                total_message_advice_by_phase[gg["game_id"]]["other"]
            ):
                acc_messages["other"][idx] += val

    # print("acc moves", acc_moves)
    # print(acc_messages)
    # print(get_percentage_accepted_moves(acc_moves) * 100)
    # print(get_percentage_accepted_messages(acc_messages) * 100)

    print(total_message_collected)
    print(total_moves_accepted, total_partial_accepted, total_moves_advice_collected)
    print(total_message_advice_accepted, total_message_advice_collected)


if __name__ == "__main__":
    main()
