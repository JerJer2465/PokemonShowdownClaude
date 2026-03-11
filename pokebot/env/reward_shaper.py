"""Reward function for Pokemon battles."""


def compute_reward(
    prev_state: dict,
    curr_state: dict,
    result: str | None,
) -> float:
    """
    Compute the step reward for the transition prev_state → curr_state.

    result: "win", "loss", "tie", or None (non-terminal)
    """
    r = 0.0

    # Terminal reward (primary signal)
    if result == "win":
        r += 1.0
    elif result == "loss":
        r -= 1.0
    # tie stays at 0

    if prev_state is None:
        return r

    # --- Shaping: faint differential ---
    def fainted_count(side: dict) -> int:
        n = sum(1 for m in side.get("reserve", []) if m.get("is_fainted"))
        if side.get("active", {}).get("is_fainted"):
            n += 1
        return n

    own_fainted_prev = fainted_count(prev_state["side_one"])
    own_fainted_curr = fainted_count(curr_state["side_one"])
    opp_fainted_prev = fainted_count(prev_state["side_two"])
    opp_fainted_curr = fainted_count(curr_state["side_two"])

    r += 0.15 * (opp_fainted_curr - opp_fainted_prev)
    r -= 0.15 * (own_fainted_curr - own_fainted_prev)

    # --- Shaping: HP advantage (very small, prevents reward hacking) ---
    def hp_sum(side: dict) -> float:
        total = 0.0
        active = side.get("active", {})
        if active and not active.get("is_fainted"):
            hp = active.get("hp", 0)
            maxhp = max(active.get("maxhp", 1), 1)
            total += hp / maxhp
        for m in side.get("reserve", []):
            if not m.get("is_fainted"):
                hp = m.get("hp", 0)
                maxhp = max(m.get("maxhp", 1), 1)
                total += hp / maxhp
        return total

    own_hp = hp_sum(curr_state["side_one"])
    opp_hp = hp_sum(curr_state["side_two"])
    r += 0.05 * (own_hp - opp_hp) / 6.0

    return r
