# Strategy: an ensemble of predictors scored online.
# Each predictor guesses the opponent's next move; the one with the best
# recent track record is trusted and we play the move that beats its guess.

import random

BEATS = {"R": "P", "P": "S", "S": "R"}
MOVES = ["R", "P", "S"]


def _most_common(items, default="R"):
    if not items:
        return default
    return max(MOVES, key=items.count)


def _markov(history, order):
    """Predict next symbol after the last `order` symbols of `history`."""
    if len(history) <= order:
        return None
    key = tuple(history[-order:])
    counts = {m: 0 for m in MOVES}
    for i in range(len(history) - order):
        if tuple(history[i:i + order]) == key:
            counts[history[i + order]] += 1
    if sum(counts.values()) == 0:
        return None
    return max(MOVES, key=counts.get)


def player(prev_play, state={}):
    if not state or prev_play == "":
        state.clear()
        state.update(opp=[], mine=[], guesses={}, scores={})

    opp, mine = state["opp"], state["mine"]
    if prev_play:
        opp.append(prev_play)
        # Reward predictors that guessed this move correctly.
        for name, guess in state["guesses"].items():
            hit = 1 if guess == prev_play else -1
            state["scores"][name] = state["scores"].get(name, 0) * 0.9 + hit

    guesses = {}
    # Opponent repeats its own patterns.
    for order in (1, 2, 3, 4, 5):
        g = _markov(opp, order)
        if g:
            guesses[f"opp{order}"] = g
    # Opponent reacts to our moves: predict from our recent sequence.
    combined = [o + m for o, m in zip(opp, mine)]
    for order in (1, 2, 3):
        if len(combined) > order:
            key = tuple(combined[-order:])
            counts = {m: 0 for m in MOVES}
            for i in range(len(combined) - order):
                if tuple(combined[i:i + order]) == key and i + order < len(opp):
                    counts[opp[i + order]] += 1
            if sum(counts.values()):
                guesses[f"pair{order}"] = max(MOVES, key=counts.get)
    # Opponent counters our most frequent recent move (mrugesh-like).
    if mine:
        guesses["freq10"] = BEATS[_most_common(mine[-10:])]
    # Opponent counters our previous move (kris-like).
    if mine:
        guesses["last"] = BEATS[mine[-1]]
    # Opponent predicts us with a 2-gram model of our moves (abbey-like).
    if len(mine) >= 2:
        g = _markov(mine, 1)
        if g:
            guesses["abbey"] = BEATS[g]

    state["guesses"] = guesses
    if guesses:
        best = max(guesses, key=lambda n: state["scores"].get(n, 0))
        move = BEATS[guesses[best]]
    else:
        move = random.choice(MOVES)

    mine.append(move)
    return move
