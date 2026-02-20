"""
Linear regression analysis of advice effectiveness on player performance.

Builds feature vectors encoding (1) the player's assigned Great Power (one-hot),
(2) player experience level (novice vs. experienced), (3) whether the player is
a novice with PHOLUS advice, (4) the game phase number, and (5) the advice
setting (no advice / message / move / both). The target variable is the
per-turn change in supply center count.

A Ridge regression with cross-validated regularization is fit to this data,
and bootstrap confidence intervals are computed for each coefficient. The
resulting plot (Figure 2 / Figure 3 in the paper) shows which factors most
strongly predict supply center gains.

The commented-out code block at the top was originally used to construct the
feature matrix from raw game data; the pre-computed version is now loaded
from data/scs_delta_data.json.
"""

import json

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_text,
    facet_grid,
    geom_errorbar,
    geom_point,
    ggplot,
    theme,
    theme_minimal,
)
from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

"""human_dict = {
    "AIGame_0": ["FRANCE", "GERMANY"],
    "AIGame_1": ["FRANCE", "RUSSIA", "TURKEY"],
    "AIGame_2": ["FRANCE", "RUSSIA", "TURKEY"],
    "AIGame_3": ["FRANCE", "RUSSIA", "TURKEY"],
    "AIGame_4": ["FRANCE", "RUSSIA", "TURKEY"],
    "AIGame_5": ["ENGLAND", "FRANCE", "TURKEY"],
    "AIGame_6": ["RUSSIA", "TURKEY"],
    "AIGame_7": ["AUSTRIA", "FRANCE", "GERMANY"],
    "AIGame_8": ["FRANCE", "GERMANY"],
    "AIGame_9": ["FRANCE", "GERMANY", "ITALY", "TURKEY"],
    "AIGame_10": ["AUSTRIA", "GERMANY", "ITALY"],
    "AIGame_11": ["AUSTRIA", "FRANCE", "GERMANY", "ITALY"],
    "AIGame_12": ["AUSTRIA", "FRANCE", "GERMANY", "ITALY"],
    "AIGame_13": ["FRANCE", "GERMANY", "TURKEY"],
    "AIGame_14": ["RUSSIA", "TURKEY"],
    "AIGame_15": ["FRANCE", "GERMANY"],
    "AIGame_16": ["AUSTRIA", "TURKEY"],
    "AIGame_17": ["AUSTRIA", "FRANCE", "GERMANY", "ITALY"],
    "AIGame_18": ["AUSTRIA", "GERMANY"],
    "AIGame_19": ["AUSTRIA", "ITALY", "TURKEY"],
    "AIGame_20": ["ENGLAND", "RUSSIA", "TURKEY"],
    "AIGame_21": ["ENGLAND", "ITALY"],
    "AIGame_22": ["AUSTRIA", "ENGLAND", "RUSSIA"],
    "AIGame_23": ["AUSTRIA", "ENGLAND", "ITALY", "RUSSIA"],
    "Centaur0": ["FRANCE", "RUSSIA"],
    "Centaur1": ["FRANCE", "RUSSIA", "ITALY"],
    "Centaur2": ["FRANCE", "RUSSIA", "GERMANY"],
    "Centaur3": ["AUSTRIA", "ENGLAND", "TURKEY"],
    "Centaur4": ["TURKEY", "AUSTRIA", "GERMANY", "ITALY"],
    "Centaur5": ["ENGLAND", "TURKEY", "ITALY"],
    "Centaur6": ["ENGLAND", "GERMANY", "ITALY"],
    "Centaur7": ["ENGLAND", "AUSTRIA"],
    "Centaur8": ["GERMANY", "TURKEY", "ITALY", "RUSSIA"],
    "Centaur9": ["ITALY", "GERMANY", "RUSSIA", "FRANCE", "TURKEY"],
    "Centaur10": ["ITALY", "AUSTRIA", "TURKEY"],
    "Centaur11": ["ENGLAND", "FRANCE", "RUSSIA", "GERMANY"],
}

games = []
for filename in os.listdir("human_games/"):
    if filename.endswith(".json"):
        with open("human_games/" + filename, "r") as f:
            games.append(json.load(f))

for filename in os.listdir("centaur_games/server/"):
    if filename.endswith(".json"):
        with open("centaur_games/server/" + filename, "r") as f:
            games.append(json.load(f))
scs = [
    "BUD",
    "BER",
    "GRE",
    "HOL",
    "BRE",
    "WAR",
    "VIE",
    "LVP",
    "POR",
    "DEN",
    "BEL",
    "BUL",
    "NAP",
    "KIE",
    "PAR",
    "MAR",
    "CON",
    "RUM",
    "NWY",
    "SPA",
    "EDI",
    "STP",
    "LON",
    "VEN",
    "TUN",
    "MUN",
    "ROM",
    "MOS",
    "SEV",
    "SWE",
    "TRI",
    "SER",
    "SMY",
    "ANK",
]

power_one_hot = {
    "AUSTRIA": [1, 0, 0, 0, 0, 0, 0],
    "ENGLAND": [0, 1, 0, 0, 0, 0, 0],
    "FRANCE": [0, 0, 1, 0, 0, 0, 0],
    "GERMANY": [0, 0, 0, 1, 0, 0, 0],
    "ITALY": [0, 0, 0, 0, 1, 0, 0],
    "RUSSIA": [0, 0, 0, 0, 0, 1, 0],
    "TURKEY": [0, 0, 0, 0, 0, 0, 1],
}

games_played_one_hot = {
    1: [1, 0],
    2: [1, 0],  # novice with help
    3: [0, 1],  # experienced
    4: [0, 1],  # tournament player
    5: [0, 1],  # tournament winner
}

player_dict = {
    "pyxxy": 5,
    "Conq": 5,
    "totonchyms": 3,
    "parip": 4,
    "sloth.dc@gmail.com": 5,
    "CMRawles": 4,
    "ben9990": 4,  # ?
    "pbansal674@gmail.com": 2,  # ?
    "NewEnglandFireSquad": 5,
    "Klaus Mikaelson ": 2,
    "abhishekh.singhal@gmail.com": 3,
    "eddie": 3,
    "ian": 5,
    "AIGame_21": 3,
    "Sploack": 5,
    "Maaike Blom": 3,
    "tim": 5,
    "ShotAway": 3,
    "david.s.graff@icloud.com": 5,
    "aguoman": 3,
    "JHenrichs": 3,
    "harvey_birdman": 4,
    "Dan Wang": 3,
    "zandersirlin@gmail.com": 4,
    "gauty7": 1,
    "Sheringford": 5,
    "Mikalis Kamaritis": 5,
    "wardiecj": 5,
    "sloth": 5,
    "jashper": 4,
    "Connorknight94": 4,  # ?
    "connorknight94": 4,  # ?
    "slothDC": 5,
    "wrenue": 3,
    "sirli100": 4,
    "penguin": 4,
    "andrew": 3,
    "ziptie": 4,
    "dakotawinslow": 2,
    "not_steven_zhang": 2,
    "Muchalnia": 2,
    "atm": 3,
    "yaya": 2,
}


def phase_to_num(phase: str):
    year = int(phase[3:-1])
    return year * 2 if phase[0] == "F" else year * 2 - 1


X = []
Y = []

sc_delta_no_advisor = []
num_no_advisor_phase = 0
sc_delta_move_advisor = []
num_move_advisor_phase = 0
sc_delta_message_advisor = []
num_message_advisor_phase = 0
sc_delta_both_advisors = []
num_both_advisors_phase = 0


def get_suggestion_type(power: str, message_history: list) -> int:
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


for game in games:
    id = game["game_id"]
    human_powers = human_dict[id]
    print(f"Game {id}")
    state_history = game["state_history"]
    movement_phases = {k: v for k, v in state_history.items() if k[-1] == "M"}
    movement_history = list(movement_phases.values())
    movement_phase_abbrs = list(movement_phases.keys())

    for hp in human_powers:
        inputs = []
        vec = []
        scs_history = []
        human_power_one_hot = power_one_hot[hp]
        controllers = list(game["powers"][hp]["controller"].values())
        controller = [
            x
            for x in controllers
            if "cicero" not in x and "type" not in x and "dummy" not in x
        ]

        if len(controller) != 1:
            print(f"Controller num != 1: {controller}")

        human_level = player_dict[controller[0]]
        vec = human_power_one_hot.copy()
        vec += games_played_one_hot[human_level]
        # 7 for power, 2 for games played

        # assert len(vec) == 10, f"Vector length is {len(vec)}, {vec}"

        if id == "Centaur8" and hp == "RUSSIA":
            vec.append(1)
        elif id == "Centaur9" and hp in ["TURKEY", "FRANCE"]:
            vec.append(1)
        elif id == "Centaur10" and hp == "ITALY":
            vec.append(1)
        elif id == "Centaur11" and hp == "ENGLAND":
            vec.append(1)
        else:
            vec.append(0)

        for phase in movement_phase_abbrs:
            num_phases = phase_to_num(phase)
            features = vec.copy()
            features.append(num_phases)

            if "AI" in id:
                features += [1, 0, 0, 0]  # advisor level one hot
            else:
                advisor_level = 0
                message_history = game["message_history"][phase]
                advisor_type = get_suggestion_type(hp, message_history)
                has_advisor = [
                    x
                    for x in message_history
                    if x["sender"] == "omniscient_type"
                    and "type" in x
                    and x["type"] == "has_suggestions"
                    and x["phase"] == phase
                    and hp in x["message"]
                ]
                if len(has_advisor):
                    advisor_level = int(has_advisor[0]["message"][-1])
                empty_one_hot = [0, 0, 0, 0]

                empty_one_hot[advisor_type] = 1
                features += empty_one_hot

            # assert len(features) == 4, f"features length: {len(features)} {features}"

            inputs.append(features)

        for phase in movement_history:
            influence = phase["influence"][hp]
            scs_history.append(sum(item in scs for item in influence))

        final_scs = game["powers"][hp]["influence"]
        scs_history.append(sum(item in scs for item in final_scs))

        # removing trailing zeros
        while len(scs_history) > 1 and scs_history[-1] == 0 and scs_history[-2] == 0:
            scs_history.pop()

        scs_delta = [
            scs_history[i + 1] - scs_history[i] for i in range(len(scs_history) - 1)
        ]

        if "AI" in id:
            final_scs_cnt = sum(item in scs for item in final_scs)
            final_scs_cnt -= 3
            if hp == "RUSSIA":
                final_scs_cnt -= 1
            sc_delta_no_advisor.append(final_scs_cnt)
            num_no_advisor_phase += len(scs_delta)
        else:
            for phase in movement_phase_abbrs[: len(scs_delta)]:
                num_phases = phase_to_num(phase) - 1
                advisor_level = 0
                message_history = game["message_history"][phase]
                has_advisor = [
                    x
                    for x in message_history
                    if x["sender"] == "omniscient_type"
                    and "type" in x
                    and x["type"] == "has_suggestions"
                    and x["phase"] == phase
                    and hp in x["message"]
                ]
                if len(has_advisor):
                    advisor_level = int(has_advisor[0]["message"][-1])
                if advisor_level == 0:
                    sc_delta_no_advisor.append(scs_delta[num_phases])
                    num_no_advisor_phase += 1
                elif advisor_level == 1:
                    sc_delta_message_advisor.append(scs_delta[num_phases])
                    num_message_advisor_phase += 1
                elif advisor_level == 2:
                    sc_delta_move_advisor.append(scs_delta[num_phases])
                    num_move_advisor_phase += 1
                elif advisor_level == 3:
                    sc_delta_both_advisors.append(scs_delta[num_phases])
                    num_both_advisors_phase += 1

        if len(inputs) > len(scs_delta):
            inputs = inputs[: len(scs_delta)]

        X += inputs
        Y += scs_delta"""

# Load pre-computed feature matrix and supply center delta targets
with open("data/scs_delta_data.json", "r") as f:
    item = json.load(f)
    X = item["X"]
    Y = item["Y"]

X = np.array(X)
Y = np.array(Y)

# Standardize features for regularized regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

coefs = []
alphas = np.logspace(-6, 6, 200)
for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_scaled, Y)
    coefs.append(ridge.coef_)


def ridge_ci_bootstrap(X, y, alpha, n_bootstraps=10000, ci=95):
    """Compute bootstrap confidence intervals for Ridge regression coefficients."""
    n_samples, n_features = X.shape
    bootstrapped_coefs = np.zeros((n_bootstraps, n_features))

    for i in range(n_bootstraps):
        indices = resample(range(n_samples), n_samples=n_samples)
        X_resampled, y_resampled = X[indices], y[indices]

        model = Ridge(alpha=alpha)
        model.fit(X_resampled, y_resampled)

        bootstrapped_coefs[i] = model.coef_

    ci_lower = np.percentile(bootstrapped_coefs, (100 - ci) / 2, axis=0)
    ci_upper = np.percentile(bootstrapped_coefs, 100 - (100 - ci) / 2, axis=0)

    return ci_lower, ci_upper


# ci_lower, ci_upper = ridge_ci_bootstrap(X, Y, 0.1, 10000, 95)

# print("lower", ci_lower)
# print("upper", ci_upper)

"""ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale("log")
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("Ridge coefficients as a function of the regularization")
plt.axis("tight")
plt.show()"""

# reg = Ridge().fit(X, Y)
# print(reg.score(X, Y))

alphase = np.logspace(-6, 6, 13)
ridge_cv = RidgeCV(alphas=alphase, store_cv_results=True)
ridge_cv.fit(X, Y)

print("best alpha", ridge_cv.alpha_)


def calculate_std_err(X, y):
    """Compute standard errors for Ridge regression coefficients."""
    model = Ridge()
    model.fit(X, y)

    y_pred = model.predict(X)

    residuals = y - y_pred
    n = X.shape[0]
    p = X.shape[1]
    df = n - p

    mse = np.sum(residuals**2) / df

    X_with_intercept = np.column_stack([np.ones(n), X])
    var_covar_mat = mse * np.linalg.inv(
        np.dot(X_with_intercept.T, X_with_intercept) + 1e-1 * np.eye(p + 1)
    )

    std_errs = np.sqrt(np.diag(var_covar_mat))

    return std_errs[1:]


# Feature names corresponding to the one-hot encoded columns in X:
# [0-6] Power, [7-9] Player type, [10] Phase count, [11-14] Advice setting
coef_names = [
    "AUSTRIA",
    "ENGLAND",
    "FRANCE",
    "GERMANY",
    "ITALY",
    "RUSSIA",
    "TURKEY",
    "novice",
    "experienced",
    "novice + advice",
    "#phases",
    "no advice",
    "message advice",
    "moves advice",
    "message + moves advice",
]

std_errs = calculate_std_err(X, Y)
# fig, ax = plt.subplots()
# ax.bar(coef_names[7:], reg.coef_[7:])
# plt.xticks()
# plt.show()

# plot player
# fig, ax = plt.subplots()

# ax.bar(coef_names, reg.coef_)

# player_coef_names = np.append(coef_names[7:11], coef_names[-1])
# player_coef = np.append(reg.coef_[7:11], reg.coef_[-1])
# ax.bar(player_coef_names, player_coef)

# plot power
# ax.bar(coef_names[7:], reg.coef_[7:])

# plot advisor
# ax.bar(coef_names[11:16], reg.coef_[11:16])
#
# print(reg.coef_)

# plt.xticks()
# plt.tight_layout()
# plt.show()

t_value = stats.t.ppf(0.975, df=X.shape[0] - X.shape[1] - 1)

# print(std_errs)

# drop russia for dummy variable trap
ci_lower = Ridge().fit(X_scaled, Y).coef_ - t_value * std_errs
ci_upper = Ridge().fit(X_scaled, Y).coef_ + t_value * std_errs

# for name, coef, lower, upper in zip(coef_names, reg.coef_, ci_lower, ci_upper):
#    print(f"{name}: {coef} ({lower}, {upper})")


def plot_coef_with_ci(X, y, alpha):
    """
    Generate Figure 2/3: regression coefficients with bootstrap CIs.

    Plots player experience and advisor setting coefficients as a faceted
    dot-and-whisker plot showing each feature's effect on supply center gains.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)

    coef = model.coef_

    ci_lower, ci_upper = ridge_ci_bootstrap(X, y, alpha, 10000, 95)
    category = ["player experience"] * 3 + ["advisor setting"] * 4

    df = pd.DataFrame(
        {
            "feature": coef_names[7:10] + coef_names[11:],
            "coefficient": np.concatenate([coef[7:10], coef[11:]]),
            "ci_lower": np.concatenate([ci_lower[7:10], ci_lower[11:]]),
            "ci_upper": np.concatenate([ci_upper[7:10], ci_upper[11:]]),
            "category": category,
        }
    )

    df["feature"] = pd.Categorical(df["feature"], categories=coef_names, ordered=True)
    df["category"] = pd.Categorical(
        df["category"],
        categories=["player experience", "advisor setting"],
        ordered=True,
    )

    # df = df.iloc[np.argsort(np.abs(df["coefficient"]))].reset_index(drop=True)

    plot = (
        ggplot(df, aes(x="feature", y="coefficient"))
        + geom_point(size=3)
        + geom_errorbar(aes(ymin="ci_lower", ymax="ci_upper"), width=0.2)
        + facet_grid("category ~ ", scales="free")
        + theme_minimal()
        + theme(
            axis_text_x=element_text(rotation=45, hjust=1, size=14),
            axis_title=element_blank(),
        )
        + coord_flip()
        # + labs(
        # title=f"Regression Coefficients for Player Level",
        # x="Feature",
        # y="Coefficient",
        # )
    )

    return plot


plot = plot_coef_with_ci(X, Y, 10.0)
plot.save("results/regression_coefs.pdf", dpi=300)
