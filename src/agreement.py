"""
Move agreement analysis between PHOLUS's advice and novice players' moves.

Computes agreement (proportion of overlapping moves) and equivalence
(exact match) between PHOLUS's suggested move sets and the moves actually
submitted by novice players, both at the start and end of each turn.
Also tracks supply center (SC) and territory gains per advice type.

Corresponds to Section 3.1 of the paper (quantitative analysis of advice
acceptance and the "Novices do not fully trust move advice" finding).
"""

import json
import os

from src.constants import CENTAUR_PATH, HUMAN_PLAYERS, NOVICES

dir = CENTAUR_PATH


def agreement(start, end):
    """Compute the set intersection between two move lists."""
    return set(start) & set(end)


# Only analyze games that had novice players
novice_dict = NOVICES

# Combined dict of all human players (experienced + novice) per game
player_dict = {
    game_id: HUMAN_PLAYERS.get(game_id, []) + NOVICES.get(game_id, [])
    for game_id in HUMAN_PLAYERS
}

# Counters for initial (start-of-turn) and final (end-of-turn) agreement
initial_agree = 0
initial_all = 0

final_agree = 0
final_all = 0

# Total advice sets generated and individual advice items within them
total_advice_set = 0
total_advice_cnt = 0

# Turn and advice counters across all games
total_game_turns = 0
total_player_turns = 0
message_advice_turns = 0
move_advice_turns = 0
total_units = 0

# Per-advice-type supply center and territory gain tracking for novices
novice_move_sc_gains, novice_move_t_gains = [], []
novice_message_sc_gains, novice_message_t_gains = [], []
novice_both_sc_gains, novice_both_t_gains = [], []
novice_sc_gains, novice_t_gains = [], []

for server_file in os.listdir(dir):
    if (
        server_file.startswith("Centaur")
        and server_file.endswith(".json")
        and server_file.replace(".json", "") in novice_dict
    ):
        with open(f"{dir}/{server_file}") as f:
            game = json.load(f)
        id = game["game_id"]
        print(id)
        order_history = game["order_history"]
        message_history = game["message_history"]

        total_game_turns += len(game["is_bot_history"])

        for phase, msgs in message_history.items():
            uniques = [
                x["message"]
                for x in msgs
                if x["sender"] == "omniscient_type" and x["type"] == "has_suggestions"
            ]

            message_advice = [x for x in uniques if "1" in x or "3" in x]
            move_advice = [x for x in uniques if "2" in x or "3" in x]

            total_player_turns += len(uniques)
            message_advice_turns += len(message_advice)
            move_advice_turns += len(move_advice)

        for power in player_dict[id]:
            for phase, messages in message_history.items():
                uniques = [
                    x["message"]
                    for x in messages
                    if x["sender"] == "omniscient_type"
                    and x["type"] == "has_suggestions"
                    and power in x["message"]
                    and ("2" in x["message"] or "3" in x["message"])
                ]

                if len(uniques):
                    power_units = game["state_history"][phase]["units"][power]
                    total_units += len(power_units)

        for novice_power in novice_dict[id]:
            prev_sc, prev_t = None, None
            has_any_move_advice = False
            has_any_message_advice = False
            for phase, orders_dict in order_history.items():
                if phase[-1] != "M":
                    continue

                final_novice_orders = orders_dict[novice_power]
                novice_messages = [
                    x for x in message_history[phase] if x["sender"] == novice_power
                ]
                move_advice = [
                    x
                    for x in message_history[phase]
                    if x["sender"] == "omniscient_type"
                    and x["type"] == "suggested_move_full"
                    and x["message"].startswith(novice_power)
                ]

                all_move_advice = [
                    x
                    for x in message_history[phase]
                    if x["sender"] == "omniscient_type"
                    and (
                        x["type"] == "suggested_move_full"
                        or x["type"] == "suggested_move_partial"
                    )
                    and x["message"].startswith(novice_power)
                ]

                all_message_advice = [
                    x
                    for x in message_history[phase]
                    if x["sender"] == "omniscient_type"
                    and x["type"] == "suggested_message"
                    and x["message"].startswith(novice_power)
                ]

                if len(all_move_advice):
                    has_any_move_advice = True
                if len(all_message_advice):
                    has_any_message_advice = True

                total_advice_set += len(all_move_advice)

                for advice in all_move_advice:
                    total_advice_cnt += len(advice["message"].split(","))

                units = game["state_history"][phase]["units"][novice_power]

                order_logs = game["order_log_history"][phase].values()
                order_logs = [
                    x for x in order_logs if x.startswith(novice_power) and "added" in x
                ]
                order_logs = [x.split("added: ")[1] for x in order_logs]

                initial_moves = {}

                if prev_sc is None and prev_t is None:
                    prev_sc = len(game["state_history"][phase]["centers"][novice_power])
                    prev_t = len(
                        game["state_history"][phase]["influence"][novice_power]
                    )
                else:
                    assert prev_sc is not None
                    assert prev_t is not None
                    curr_sc = len(game["state_history"][phase]["centers"][novice_power])
                    curr_t = len(
                        game["state_history"][phase]["influence"][novice_power]
                    )
                    if len(all_message_advice) and len(all_move_advice):
                        novice_message_sc_gains.append(curr_sc - prev_sc)
                        novice_message_t_gains.append(curr_t - prev_t)
                        novice_move_sc_gains.append(curr_sc - prev_sc)
                        novice_move_t_gains.append(curr_t - prev_t)
                    elif len(all_message_advice):
                        novice_message_sc_gains.append(curr_sc - prev_sc)
                        novice_message_t_gains.append(curr_t - prev_t)
                    elif len(all_move_advice):
                        novice_move_sc_gains.append(curr_sc - prev_sc)
                        novice_move_t_gains.append(curr_t - prev_t)
                    else:
                        novice_sc_gains.append(curr_sc - prev_sc)
                        novice_t_gains.append(curr_t - prev_t)

                    prev_sc = curr_sc
                    prev_t = curr_t

                for unit in units:
                    for log in order_logs:
                        if log.startswith(unit):
                            initial_moves[unit] = log
                            break

                if len(move_advice) and len(initial_moves):
                    initial_move_advice = move_advice[0]["message"].split(":", 1)[1]
                    initial_move_advice = initial_move_advice.split(",")

                    final_move_advice = move_advice[-1]["message"].split(":", 1)[1]
                    final_move_advice = final_move_advice.split(",")

                    len_to_add = len(units)

                    initial_all += len_to_add
                    initial_move_advice = [x.strip() for x in initial_move_advice]
                    final_move_advice = [x.strip() for x in final_move_advice]
                    initial_moves_values = [
                        x.strip() for x in list(initial_moves.values())
                    ]
                    print(initial_move_advice, initial_moves_values)
                    initial_agree += len(
                        agreement(initial_move_advice, initial_moves_values)
                    )

                    final_all += len(units)
                    final_agree += len(
                        agreement(final_move_advice, final_novice_orders)
                    )

            if prev_sc is not None:
                assert prev_t is not None
                final_sc = len(game["powers"][novice_power]["centers"])
                final_t = len(game["powers"][novice_power]["influence"])
                if has_any_message_advice and has_any_move_advice:
                    novice_message_sc_gains.append(final_sc - prev_sc)
                    novice_message_t_gains.append(final_t - prev_t)
                    novice_move_sc_gains.append(final_sc - prev_sc)
                    novice_move_t_gains.append(final_t - prev_t)
                elif has_any_message_advice:
                    novice_message_sc_gains.append(final_sc - prev_sc)
                    novice_message_t_gains.append(final_t - prev_t)
                elif has_any_move_advice:
                    novice_move_sc_gains.append(final_sc - prev_sc)
                    novice_move_t_gains.append(final_t - prev_t)
                else:
                    novice_sc_gains.append(final_sc - prev_sc)
                    novice_t_gains.append(final_t - prev_t)

print(initial_agree, initial_all, final_agree, final_all)
print(total_advice_set, total_advice_cnt)
print(total_game_turns, total_player_turns, message_advice_turns, move_advice_turns)
print(total_units)

print(
    sum(novice_move_sc_gains) / len(novice_move_sc_gains) if novice_move_sc_gains else 0
)
print(sum(novice_move_t_gains) / len(novice_move_t_gains) if novice_move_t_gains else 0)
print(
    sum(novice_message_sc_gains) / len(novice_message_sc_gains)
    if novice_message_sc_gains
    else 0
)
print(
    sum(novice_message_t_gains) / len(novice_message_t_gains)
    if novice_message_t_gains
    else 0
)
print(sum(novice_sc_gains) / len(novice_sc_gains) if novice_sc_gains else 0)
print(sum(novice_t_gains) / len(novice_t_gains) if novice_t_gains else 0)
