# Deep Reinforcement Learning with function approximation over engineered features
# Statespace: Bitboard Representation
# Search: Actor-Critic Policy over Legal Moves
# Objective: Learn state-dependent use of engineered heuristics

import numpy as np
from collections import Counter
import random
import pygame

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


# --- GAME IMPLEMENTATION CODE ---

BOARD_SIZE = 9
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
FULL_MASK = (1 << NUM_CELLS) - 1


def rc_to_idx(r, c):
    return r * BOARD_SIZE + c


def idx_to_rc(idx):
    return divmod(idx, BOARD_SIZE)


def bit_at(r, c):
    return 1 << rc_to_idx(r, c)


def iter_set_bits(bits):
    while bits:
        lsb = bits & -bits
        idx = lsb.bit_length() - 1
        yield idx
        bits ^= lsb


def bitboards_to_board(state):
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

    for color, bits in state["color_bits"].items():
        for idx in iter_set_bits(bits):
            r, c = idx_to_rc(idx)
            board[r, c] = color

    return board


def get_color_at_idx(state, idx):
    mask = 1 << idx
    for color, bits in state["color_bits"].items():
        if bits & mask:
            return color
    return 0


def get_color_at_rc(state, r, c):
    return get_color_at_idx(state, rc_to_idx(r, c))


def set_color_at_idx(state, idx, color):
    mask = 1 << idx

    for c in state["color_bits"]:
        state["color_bits"][c] &= ~mask

    if color == 0:
        state["occupied_bits"] &= ~mask
    else:
        state["color_bits"][color] |= mask
        state["occupied_bits"] |= mask


def copy_state(state):
    return {
        "color_bits": state["color_bits"].copy(),
        "occupied_bits": state["occupied_bits"]
    }


def empty_board_bits(num_colors=6):
    return {
        "color_bits": {color: 0 for color in range(1, num_colors + 1)},
        "occupied_bits": 0
    }


def color_selector():
    return [random.randint(1, 6) for _ in range(3)]


def spawn_tiles_bits(state, colors):
    empty = list(iter_set_bits(FULL_MASK ^ state["occupied_bits"]))

    num_tiles = min(3, len(empty))
    chosen = random.sample(empty, num_tiles)

    new_state = copy_state(state)
    spawned_coords = []

    for i, idx in enumerate(chosen):
        color = colors[i]
        set_color_at_idx(new_state, idx, color)
        spawned_coords.append(idx_to_rc(idx))

    return new_state, spawned_coords


# --- REACHABILITY HELPERS ---

LEFT_EDGE = 0
RIGHT_EDGE = 0
TOP_EDGE = 0
BOTTOM_EDGE = 0

for r in range(BOARD_SIZE):
    LEFT_EDGE |= 1 << rc_to_idx(r, 0)
    RIGHT_EDGE |= 1 << rc_to_idx(r, BOARD_SIZE - 1)

for c in range(BOARD_SIZE):
    TOP_EDGE |= 1 << rc_to_idx(0, c)
    BOTTOM_EDGE |= 1 << rc_to_idx(BOARD_SIZE - 1, c)


def shift_left(bits):
    return (bits & ~LEFT_EDGE) >> 1


def shift_right(bits):
    return ((bits & ~RIGHT_EDGE) << 1) & FULL_MASK


def shift_up(bits):
    return (bits & ~TOP_EDGE) >> BOARD_SIZE


def shift_down(bits):
    return ((bits & ~BOTTOM_EDGE) << BOARD_SIZE) & FULL_MASK


def reachable_empty_bits(state, source_idx):
    source_bit = 1 << source_idx
    empty_bits = FULL_MASK ^ state["occupied_bits"]

    frontier = source_bit
    visited = source_bit

    while frontier:
        nbrs = (
            shift_left(frontier)
            | shift_right(frontier)
            | shift_up(frontier)
            | shift_down(frontier)
        ) & FULL_MASK

        new_frontier = nbrs & empty_bits & ~visited

        if new_frontier == 0:
            break

        visited |= new_frontier
        frontier = new_frontier

    return visited & empty_bits


def check_move_bits(state, source, destination):
    sr, sc = source
    dr, dc = destination

    source_idx = rc_to_idx(sr, sc)
    dest_idx = rc_to_idx(dr, dc)

    if get_color_at_idx(state, source_idx) == 0:
        return False
    if source_idx == dest_idx:
        return False
    if get_color_at_idx(state, dest_idx) != 0:
        return False

    reachable = reachable_empty_bits(state, source_idx)
    return ((reachable >> dest_idx) & 1) == 1


def apply_move_bits(state, source_idx, dest_idx):
    color = get_color_at_idx(state, source_idx)

    if color == 0:
        raise ValueError("Source is empty")
    if get_color_at_idx(state, dest_idx) != 0:
        raise ValueError("Destination is occupied")

    new_state = copy_state(state)

    src_mask = 1 << source_idx
    dst_mask = 1 << dest_idx

    new_state["color_bits"][color] &= ~src_mask
    new_state["color_bits"][color] |= dst_mask

    new_state["occupied_bits"] &= ~src_mask
    new_state["occupied_bits"] |= dst_mask

    return new_state


# --- CLEARING MECHANIC ---

DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def scan_line_from_cell(state, r, c, color, dr, dc):
    cells = [(r, c)]

    rr, cc = r + dr, c + dc
    while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and get_color_at_rc(state, rr, cc) == color:
        cells.append((rr, cc))
        rr += dr
        cc += dc

    rr, cc = r - dr, c - dc
    while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and get_color_at_rc(state, rr, cc) == color:
        cells.append((rr, cc))
        rr -= dr
        cc -= dc

    return cells


def five_more_clear_bits(state, cell):
    r, c = cell
    color = get_color_at_rc(state, r, c)

    if color == 0:
        return copy_state(state), []

    to_clear = set()
    lengths = []

    for dr, dc in DIRECTIONS:
        line_cells = scan_line_from_cell(state, r, c, color, dr, dc)

        if len(line_cells) >= 5:
            lengths.append(len(line_cells))
            to_clear.update(line_cells)

    new_state = copy_state(state)

    for rr, cc in to_clear:
        set_color_at_idx(new_state, rc_to_idx(rr, cc), 0)

    return new_state, lengths


POINT_SYSTEM = {
    5: 10,
    6: 12,
    7: 18,
    8: 28,
    9: 42,
}


# --- ENGINEERED HEURISTICS / FEATURES ---

LINE_POINTS = {
    2: 1,
    3: 4,
    4: 10,
}


def build_ordered_lines():
    horizontal = []
    vertical = []
    diag_dr = []
    diag_dl = []

    for r in range(BOARD_SIZE):
        horizontal.append([rc_to_idx(r, c) for c in range(BOARD_SIZE)])

    for c in range(BOARD_SIZE):
        vertical.append([rc_to_idx(r, c) for r in range(BOARD_SIZE)])

    for c0 in range(BOARD_SIZE):
        line = []
        r, c = 0, c0
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            line.append(rc_to_idx(r, c))
            r += 1
            c += 1
        if len(line) >= 2:
            diag_dr.append(line)

    for r0 in range(1, BOARD_SIZE):
        line = []
        r, c = r0, 0
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            line.append(rc_to_idx(r, c))
            r += 1
            c += 1
        if len(line) >= 2:
            diag_dr.append(line)

    for c0 in range(BOARD_SIZE):
        line = []
        r, c = 0, c0
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            line.append(rc_to_idx(r, c))
            r += 1
            c -= 1
        if len(line) >= 2:
            diag_dl.append(line)

    for r0 in range(1, BOARD_SIZE):
        line = []
        r, c = r0, BOARD_SIZE - 1
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            line.append(rc_to_idx(r, c))
            r += 1
            c -= 1
        if len(line) >= 2:
            diag_dl.append(line)

    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "diag_dr": diag_dr,
        "diag_dl": diag_dl
    }


ORDERED_LINES = build_ordered_lines()


def score_ordered_line_bits(state, line):
    total = 0
    n = len(line)
    i = 0

    while i < n:
        idx = line[i]
        color = get_color_at_idx(state, idx)

        if color == 0:
            i += 1
            continue

        j = i
        while j + 1 < n and get_color_at_idx(state, line[j + 1]) == color:
            j += 1

        run_length = j - i + 1

        if run_length not in LINE_POINTS:
            i = j + 1
            continue

        base = LINE_POINTS[run_length]

        L = i
        while L - 1 >= 0:
            left_color = get_color_at_idx(state, line[L - 1])
            if left_color == 0 or left_color == color:
                L -= 1
            else:
                break

        R = j
        while R + 1 < n:
            right_color = get_color_at_idx(state, line[R + 1])
            if right_color == 0 or right_color == color:
                R += 1
            else:
                break

        segment_length = R - L + 1

        left_open = i - 1 >= 0 and get_color_at_idx(state, line[i - 1]) == 0
        right_open = j + 1 < n and get_color_at_idx(state, line[j + 1]) == 0

        empties_in_segment = sum(
            1 for k in range(L, R + 1)
            if get_color_at_idx(state, line[k]) == 0
        )

        score = base

        if segment_length >= 5:
            if left_open and right_open:
                score = 1.6 * base + 0.4 * empties_in_segment
            elif left_open or right_open:
                score = 1.35 * base + 0.25 * empties_in_segment
            else:
                score = 1.15 * base + 0.15 * empties_in_segment

        total += score
        i = j + 1

    return total


def directional_line_score_bits(state, direction_name):
    return sum(score_ordered_line_bits(state, line) for line in ORDERED_LINES[direction_name])


def line_potential_score_bits(state):
    return (
        directional_line_score_bits(state, "horizontal")
        + directional_line_score_bits(state, "vertical")
        + directional_line_score_bits(state, "diag_dr")
        + directional_line_score_bits(state, "diag_dl")
    )


def build_five_cell_windows():
    windows = []

    for dr, dc in DIRECTIONS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cells = []

                for i in range(5):
                    rr = r + i * dr
                    cc = c + i * dc

                    if not (0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE):
                        break

                    cells.append(rc_to_idx(rr, cc))

                if len(cells) == 5:
                    mask = 0
                    for idx in cells:
                        mask |= 1 << idx
                    windows.append((cells, mask))

    return windows


FIVE_WINDOWS = build_five_cell_windows()


def popcount(x):
    return x.bit_count()


def gap_potential_score_bits(state):
    five_minus_gap_points = {
        4: 60,
        3: 12,
        2: 2
    }

    total = 0

    for cells, mask in FIVE_WINDOWS:
        occupied_in_window = state["occupied_bits"] & mask

        if occupied_in_window == 0:
            continue

        colors_present = []

        for color in state["color_bits"]:
            if state["color_bits"][color] & mask:
                colors_present.append(color)

        if len(colors_present) != 1:
            continue

        color = colors_present[0]
        n = popcount(state["color_bits"][color] & mask)

        if n not in five_minus_gap_points:
            continue

        base_score = five_minus_gap_points[n]

        max_run = 0
        run = 0

        for idx in cells:
            if (state["color_bits"][color] >> idx) & 1:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0

        total += base_score + 2 * max_run

    return total


def build_neighbour_masks():
    neighbour_masks = [0] * NUM_CELLS

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            idx = rc_to_idx(r, c)
            mask = 0

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue

                    rr, cc = r + dr, c + dc

                    if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                        mask |= 1 << rc_to_idx(rr, cc)

            neighbour_masks[idx] = mask

    return neighbour_masks


NEIGHBOUR_MASKS = build_neighbour_masks()


def path_mobility_score_bits(state):
    total = 0

    for source_idx in iter_set_bits(state["occupied_bits"]):
        color = get_color_at_idx(state, source_idx)
        reachable = reachable_empty_bits(state, source_idx)

        for dest_idx in iter_set_bits(reachable):
            neighbour_mask = NEIGHBOUR_MASKS[dest_idx]
            same_color_neighbours = state["color_bits"][color] & neighbour_mask & ~(1 << source_idx)

            if same_color_neighbours != 0:
                total += 1

    return total


def color_availability_bits(state, preview_colors, preview_weight=1.0):
    preview_count = Counter(preview_colors)
    availability = {}

    for color in state["color_bits"]:
        board_count = state["color_bits"][color].bit_count()
        availability[color] = board_count + preview_weight * preview_count[color]

    return availability


def preview_color_bonus_bits(state, preview_colors):
    preview_count = Counter(preview_colors)
    availability = color_availability_bits(state, preview_colors, preview_weight=1.0)

    total = 0

    for cells, mask in FIVE_WINDOWS:
        occupied_in_window = state["occupied_bits"] & mask

        if occupied_in_window == 0:
            continue

        colors_present = []

        for color in state["color_bits"]:
            if state["color_bits"][color] & mask:
                colors_present.append(color)

        if len(colors_present) != 1:
            continue

        color = colors_present[0]
        n = popcount(state["color_bits"][color] & mask)
        empties = 5 - popcount(occupied_in_window)

        if empties == 0:
            continue

        k = preview_count[color]

        if k == 0:
            continue

        usable_preview = min(k, empties)
        supply_factor = min(availability[color], 8) / 8.0

        reachable_targets = 0
        empty_targets = [idx for idx in cells if get_color_at_idx(state, idx) == 0]

        for target_idx in empty_targets:
            reachable_by_same_color = False

            for source_idx in iter_set_bits(state["color_bits"][color]):
                if source_idx in cells:
                    continue

                reachable = reachable_empty_bits(state, source_idx)

                if ((reachable >> target_idx) & 1) == 1:
                    reachable_by_same_color = True
                    break

            if reachable_by_same_color:
                reachable_targets += 1

        if n == 4:
            access_factor = 1.0 if reachable_targets == 1 else 0.25
            base = 20
        elif n == 3:
            if reachable_targets == 2:
                access_factor = 1.0
            elif reachable_targets == 1:
                access_factor = 0.6
            else:
                access_factor = 0.2

            base = 6
        else:
            continue

        total += base * usable_preview * supply_factor * access_factor

    return total


def completion_access_score_bits(state):
    total = 0

    for cells, mask in FIVE_WINDOWS:
        occupied_in_window = state["occupied_bits"] & mask

        if occupied_in_window == 0:
            continue

        colors_present = []

        for color in state["color_bits"]:
            if state["color_bits"][color] & mask:
                colors_present.append(color)

        if len(colors_present) != 1:
            continue

        color = colors_present[0]
        n = popcount(state["color_bits"][color] & mask)

        if n not in (3, 4):
            continue

        empty_targets = [idx for idx in cells if get_color_at_idx(state, idx) == 0]
        reachable_targets = 0

        for target_idx in empty_targets:
            reachable_by_same_color = False

            for source_idx in iter_set_bits(state["color_bits"][color]):
                if source_idx in cells:
                    continue

                reachable = reachable_empty_bits(state, source_idx)

                if ((reachable >> target_idx) & 1) == 1:
                    reachable_by_same_color = True
                    break

            if reachable_by_same_color:
                reachable_targets += 1

        if n == 4:
            if reachable_targets == 1:
                total += 40
            else:
                total += 2

        elif n == 3:
            if reachable_targets == 2:
                total += 14
            elif reachable_targets == 1:
                total += 6
            else:
                total += 1

    return total


def inaccessible_four_count_bits(state):
    count = 0

    for cells, mask in FIVE_WINDOWS:
        occupied_in_window = state["occupied_bits"] & mask

        if occupied_in_window == 0:
            continue

        colors_present = []

        for color in state["color_bits"]:
            if state["color_bits"][color] & mask:
                colors_present.append(color)

        if len(colors_present) != 1:
            continue

        color = colors_present[0]
        n = popcount(state["color_bits"][color] & mask)

        if n != 4:
            continue

        empty_targets = [idx for idx in cells if get_color_at_idx(state, idx) == 0]

        if len(empty_targets) != 1:
            continue

        target_idx = empty_targets[0]
        reachable_by_same_color = False

        for source_idx in iter_set_bits(state["color_bits"][color]):
            if source_idx in cells:
                continue

            reachable = reachable_empty_bits(state, source_idx)

            if ((reachable >> target_idx) & 1) == 1:
                reachable_by_same_color = True
                break

        if not reachable_by_same_color:
            count += 1

    return count


def extract_state_features(state, preview_colors):
    occupied = state["occupied_bits"].bit_count()
    empty = NUM_CELLS - occupied
    fill_ratio = occupied / NUM_CELLS

    line_score = line_potential_score_bits(state)
    gap_score = gap_potential_score_bits(state)
    mobility_score = path_mobility_score_bits(state)
    preview_score = preview_color_bonus_bits(state, preview_colors)
    access_score = completion_access_score_bits(state)
    inaccessible_fours = inaccessible_four_count_bits(state)
    mobility_per_piece = mobility_score / max(occupied, 1)

    return {
        "occupied": occupied,
        "empty": empty,
        "fill_ratio": fill_ratio,
        "line_score": line_score,
        "gap_score": gap_score,
        "mobility_score": mobility_score,
        "preview_score": preview_score,
        "access_score": access_score,
        "inaccessible_fours": inaccessible_fours,
        "mobility_per_piece": mobility_per_piece,
    }


def features_to_vector(current_features, next_features, move_reward=0):
    delta_gap = next_features["gap_score"] - current_features["gap_score"]
    delta_path = next_features["mobility_score"] - current_features["mobility_score"]
    delta_access = next_features["access_score"] - current_features["access_score"]

    return np.array([
        next_features["fill_ratio"],
        next_features["line_score"] / 70.0,
        next_features["gap_score"] / 500.0,
        next_features["mobility_score"] / 650.0,
        next_features["mobility_per_piece"] / 20.0,
        next_features["preview_score"] / 100.0,
        next_features["access_score"] / 200.0,
        delta_gap / 250.0,
        delta_path / 250.0,
        delta_access / 120.0,
        move_reward / 20.0,
    ], dtype=np.float32)


def game_phase(features):
    fill = features["fill_ratio"]

    if fill < 0.30:
        return "early"
    elif fill < 0.50:
        return "mid"
    else:
        return "late"


def adaptive_weights(phase, features):
    fill = features["fill_ratio"]
    occupied = max(features["occupied"], 1)

    line_score = features["line_score"]
    gap_score = features["gap_score"]
    mobility_score = features["mobility_score"]
    preview_score = features["preview_score"]
    access_score = features["access_score"]
    inaccessible_fours = features["inaccessible_fours"]

    mobility_per_piece = mobility_score / occupied
    no_obvious_clear = access_score == 0

    if phase == "early":
        alpha, beta, gamma, rho, eta = 0.40, 0.18, 0.003, 0.05, 0.04
        delta_g, delta_p, k = 0.05, 0.0005, 14

    elif phase == "mid":
        alpha, beta, gamma, rho, eta = 0.45, 0.14, 0.008, 0.04, 0.07
        delta_g, delta_p, k = 0.07, 0.001, 14

    else:
        alpha, beta, gamma, rho, eta = 0.40, 0.10, 0.020, 0.02, 0.11
        delta_g, delta_p, k = 0.06, 0.003, 14

    if mobility_per_piece < 8:
        gamma *= 1.8
        delta_p *= 2.0
        eta *= 1.35
        k *= 1.2
        beta *= 0.85

    elif mobility_per_piece < 12:
        gamma *= 1.3
        delta_p *= 1.4
        eta *= 1.15

    if gap_score > 180:
        beta *= 1.20
        delta_g *= 1.25
    elif gap_score < 80:
        beta *= 0.90

    if line_score > 45:
        alpha *= 1.10
        k *= 1.05
    elif line_score < 15:
        alpha *= 0.95

    if fill > 0.75:
        gamma *= 1.4
        delta_p *= 1.4
        eta *= 1.25
        k *= 1.25
        beta *= 0.85
        rho *= 0.75

    elif fill < 0.25:
        beta *= 1.10
        delta_g *= 1.10
        rho *= 1.20

    if preview_score > 20 and fill < 0.65:
        rho *= 1.20
    elif fill > 0.70:
        rho *= 0.80

    if inaccessible_fours >= 1:
        eta *= 1.35

    if inaccessible_fours >= 2:
        eta *= 1.15
        delta_p *= 1.20

    if no_obvious_clear:
        eta *= 1.20
        delta_g *= 1.10

    return alpha, beta, gamma, rho, eta, delta_g, delta_p, k


def adaptive_expert_score(current_state, next_state, preview_colors, move_reward):
    current_features = extract_state_features(current_state, preview_colors)
    next_features = extract_state_features(next_state, preview_colors)

    phase = game_phase(current_features)
    alpha, beta, gamma, rho, eta, delta_g, delta_p, k = adaptive_weights(
        phase,
        current_features
    )

    delta_access = next_features["access_score"] - current_features["access_score"]
    delta_gap = next_features["gap_score"] - current_features["gap_score"]
    delta_mobility = next_features["mobility_score"] - current_features["mobility_score"]

    score = (
        alpha * next_features["line_score"]
        + beta * next_features["gap_score"]
        + gamma * next_features["mobility_score"]
        + rho * next_features["preview_score"]
        + eta * delta_access
        + delta_g * delta_gap
        + delta_p * delta_mobility
        + k * move_reward
    )

    return score


# --- ACTOR-CRITIC MODEL ---

class ActorCritic(nn.Module):
    # Actor: Scores each candidate action using the engineered feature vector of the resulting state.
    # Critic: Estimates the value of a state using the same engineered features.
    # This allows the model to learn nonlinear, state-dependent use of the handcrafted heuristics.

    def __init__(self, input_dim=11, hidden_dim=64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.actor_head = nn.Linear(hidden_dim, 1)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def actor_score(self, x):
        z = self.shared(x)
        return self.actor_head(z).squeeze(-1)

    def critic_value(self, x):
        z = self.shared(x)
        return self.critic_head(z).squeeze(-1)


# --- ACTION GENERATION ---

def get_action_candidates(state, preview_colors):

    # Returns candidate legal actions.
    
    actions = []

    for source_idx in iter_set_bits(state["occupied_bits"]):
        reachable = reachable_empty_bits(state, source_idx)
        current_features = extract_state_features(state, preview_colors)

        for dest_idx in iter_set_bits(reachable):
            temp_state = apply_move_bits(state, source_idx, dest_idx)
            next_state, cleared = five_more_clear_bits(temp_state, idx_to_rc(dest_idx))

            move_reward = sum(POINT_SYSTEM.get(L, 0) for L in cleared)

            next_features = extract_state_features(next_state, preview_colors)

            features_vec = features_to_vector(
                current_features,
                next_features,
                move_reward
            )

            move = (idx_to_rc(source_idx), idx_to_rc(dest_idx))

            actions.append({
                "next_state": next_state,
                "move": move,
                "move_reward": move_reward,
                "cleared": cleared,
                "features_vec": features_vec
            })

    return actions


def adaptive_expert_action_index(current_state, preview_colors, actions):
    best_score = -float("inf")
    best_idx = 0

    for i, action in enumerate(actions):
        score = adaptive_expert_score(
            current_state=current_state,
            next_state=action["next_state"],
            preview_colors=preview_colors,
            move_reward=action["move_reward"]
        )

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def pretrain_actor_with_adaptive_expert(
    model,
    optimizer,
    num_games=20,
    print_every=1,
):
    #Behaviour cloning stage: the actor learns to imitate your adaptive heuristic expert.

    imitation_losses = []
    imitation_scores = []

    for game in range(1, num_games + 1):
        state = empty_board_bits()

        initial_colors = color_selector()
        preview_colors = color_selector()
        state, _ = spawn_tiles_bits(state, initial_colors)

        total_score = 0
        total_loss = 0.0
        steps = 0

        while state["occupied_bits"].bit_count() < NUM_CELLS:
            actions = get_action_candidates(state, preview_colors)

            if not actions:
                break

            expert_idx = adaptive_expert_action_index(
                state,
                preview_colors,
                actions
            )

            action_features = torch.tensor(
                np.array([a["features_vec"] for a in actions]),
                dtype=torch.float32
            )

            logits = model.actor_score(action_features).unsqueeze(0)
            target = torch.tensor([expert_idx], dtype=torch.long)

            loss = nn.CrossEntropyLoss()(logits, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            chosen = actions[expert_idx]

            env_state, next_preview_colors, spawn_reward = apply_environment_after_move(
                chosen["next_state"],
                preview_colors,
                chosen["cleared"]
            )

            reward = chosen["move_reward"] + spawn_reward
            total_score += reward

            state = env_state
            preview_colors = next_preview_colors

            total_loss += loss.item()
            steps += 1

        mean_loss = total_loss / max(steps, 1)

        imitation_losses.append(mean_loss)
        imitation_scores.append(total_score)

        print(
            f"Imitation Game {game:4d} | "
            f"Expert Score = {total_score:5d} | "
            f"Steps = {steps:4d} | "
            f"Loss = {mean_loss:.4f}"
        )

        if game % print_every == 0:
            print(
                f"--- Imitation summary {game} | "
                f"Avg score last {print_every}: {np.mean(imitation_scores[-print_every:]):.2f} | "
                f"Avg loss last {print_every}: {np.mean(imitation_losses[-print_every:]):.4f}"
            )

    return imitation_losses, imitation_scores


def select_action(model, actions, temperature=1.0):

    #Samples an action from the actor's probability distribution.
    action_features = torch.tensor(
        np.array([a["features_vec"] for a in actions]),
        dtype=torch.float32
    )

    logits = model.actor_score(action_features)

    logits = logits / temperature
    probs = torch.softmax(logits, dim=0)

    dist = torch.distributions.Categorical(probs)
    action_idx = dist.sample()

    log_prob = dist.log_prob(action_idx)
    entropy = dist.entropy()

    return action_idx.item(), log_prob, entropy


def greedy_action(model, actions):
    #Evaluation: Chooses highest-scoring action from actor.
    action_features = torch.tensor(
        np.array([a["features_vec"] for a in actions]),
        dtype=torch.float32
    )

    with torch.no_grad():
        logits = model.actor_score(action_features)

    return int(torch.argmax(logits).item())


# --- ENVIRONMENT STEP ---

def apply_environment_after_move(post_move_state, preview_colors, cleared):
    state = post_move_state
    total_reward = 0
    next_preview = list(preview_colors)

    if cleared == []:
        state, spawned_tiles = spawn_tiles_bits(state, preview_colors)
        next_preview = color_selector()

        for new_r, new_c in spawned_tiles:
            if get_color_at_rc(state, new_r, new_c) != 0:
                state, spawned_cleared = five_more_clear_bits(state, (new_r, new_c))

                for L in spawned_cleared:
                    total_reward += POINT_SYSTEM.get(L, 0)

    return state, next_preview, total_reward


# --- TRAINING ---
def state_features_to_vector(features):
    return np.array([
        features["fill_ratio"],
        features["line_score"] / 70.0,
        features["gap_score"] / 500.0,
        features["mobility_score"] / 650.0,
        features["mobility_per_piece"] / 20.0,
        features["preview_score"] / 100.0,
        features["access_score"] / 200.0,
        0.0,  # delta_gap
        0.0,  # delta_path
        0.0,  # delta_access
        0.0,  # move_reward
    ], dtype=np.float32)

def train_actor_critic(
    num_episodes=2000,
    gamma=0.99,
    lr=1e-3,
    entropy_coef=0.0,
    value_coef=0.5,
    print_every=50,
    plot_every=50,
    model=None,
    optimizer=None,
    start_episode=1,
):
    if model is None:
        model = ActorCritic(input_dim=11, hidden_dim=64)

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    episode_scores = []
    episode_losses = []
    episode_steps_list = []
    running_means = []

    for local_episode in range(1, num_episodes + 1):
        episode = start_episode + local_episode - 1
        state = empty_board_bits()

        initial_colors = color_selector()
        preview_colors = color_selector()
        state, _ = spawn_tiles_bits(state, initial_colors)

        total_score = 0
        total_loss = 0.0
        steps = 0

        reward_scale = 100.0
        bc_coef = max(0.2, 2.0 * (0.99 ** episode))
        value_coef = 0.1
        entropy_coef = 0.001

        while state["occupied_bits"].bit_count() < NUM_CELLS:
            actions = get_action_candidates(state, preview_colors)

            if not actions:
                break

            current_features = extract_state_features(state, preview_colors)
            current_vec = torch.tensor(
                state_features_to_vector(current_features),
                dtype=torch.float32
            )

            temperature = max(0.05, 0.3 * (0.995 ** episode))
            action_idx, log_prob, entropy = select_action(model, actions, temperature=temperature)
            chosen = actions[action_idx]

            post_move_state = chosen["next_state"]
            move_reward = chosen["move_reward"]
            cleared = chosen["cleared"]

            env_state, next_preview_colors, spawn_reward = apply_environment_after_move(
                post_move_state,
                preview_colors,
                cleared
            )

            raw_reward = move_reward + spawn_reward
            reward = raw_reward / reward_scale
            total_score += raw_reward

            terminal = env_state["occupied_bits"].bit_count() >= NUM_CELLS

            next_features = extract_state_features(env_state, next_preview_colors)
            next_vec = torch.tensor(
                state_features_to_vector(next_features),
                dtype=torch.float32
            )

            value = model.critic_value(current_vec)

            with torch.no_grad():
                if terminal:
                    target = torch.tensor(float(reward), dtype=torch.float32)
                else:
                    next_value = model.critic_value(next_vec)
                    target = reward + gamma * next_value

            advantage = target - value

            actor_loss = -log_prob * advantage.detach()
            critic_loss = advantage.pow(2)
            entropy_loss = -entropy_coef * entropy

            expert_idx = adaptive_expert_action_index(state, preview_colors, actions)

            action_features = torch.tensor(
                np.array([a["features_vec"] for a in actions]),
                dtype=torch.float32
            )

            expert_logits = model.actor_score(action_features).unsqueeze(0)
            expert_target = torch.tensor([expert_idx], dtype=torch.long)

            bc_loss = nn.CrossEntropyLoss()(expert_logits, expert_target)

            loss = actor_loss + value_coef * critic_loss + entropy_loss + bc_coef * bc_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

            state = env_state
            preview_colors = next_preview_colors

            if terminal:
                break

        mean_loss = total_loss / max(steps, 1)

        episode_scores.append(total_score)
        episode_losses.append(mean_loss)
        episode_steps_list.append(steps)

        recent_mean = np.mean(episode_scores[-print_every:])
        running_means.append(recent_mean)

        print(
            f"Episode {episode:4d} | "
            f"Score = {total_score:5d} | "
            f"Avg({print_every}) = {recent_mean:8.2f} | "
            f"Steps = {steps:4d} | "
            f"Mean Loss = {mean_loss:.4f}"
        )

        if episode % print_every == 0:
            print(
                f"--- Episode {episode} summary | "
                f"Average score last {print_every}: {recent_mean:.2f} | "
                f"Best so far: {max(episode_scores)} | "
                f"Mean steps last {print_every}: {np.mean(episode_steps_list[-print_every:]):.2f}"
            )

        if episode % plot_every == 0:
            plot_training_progress(
                episode_scores,
                episode_losses,
                episode_steps_list,
                print_every
            )

    return model, episode_scores, episode_losses, episode_steps_list



def moving_average(values, window):
    if len(values) < window:
        return values

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid"
    )


def plot_training_progress(scores, losses, steps, window=50):
    plt.figure(figsize=(12, 4))

    plt.plot(scores, label="Episode Score", alpha=0.35)

    if len(scores) >= window:
        ma_scores = moving_average(scores, window)
        plt.plot(
            range(window - 1, len(scores)),
            ma_scores,
            label=f"Score Moving Avg ({window})"
        )

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Training Performance")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    plt.pause(0.001)

    plt.figure(figsize=(12, 4))
    plt.plot(losses, label="Mean Episode Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Actor-Critic Loss")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    plt.pause(0.001)

    plt.figure(figsize=(12, 4))
    plt.plot(steps, label="Episode Length")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode Length Over Training")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    plt.pause(0.001)

# --- EVALUATION ---

def evaluate_actor_critic(model, num_games=50, verbose=False):
    scores = []

    for game in range(1, num_games + 1):
        state = empty_board_bits()

        initial_colors = color_selector()
        preview_colors = color_selector()

        state, _ = spawn_tiles_bits(state, initial_colors)

        total_score = 0

        while state["occupied_bits"].bit_count() < NUM_CELLS:
            actions = get_action_candidates(state, preview_colors)

            if not actions:
                break

            action_idx = greedy_action(model, actions)
            chosen = actions[action_idx]

            post_move_state = chosen["next_state"]
            move_reward = chosen["move_reward"]
            cleared = chosen["cleared"]

            env_state, next_preview_colors, spawn_reward = apply_environment_after_move(
                post_move_state,
                preview_colors,
                cleared
            )

            reward = move_reward + spawn_reward
            total_score += reward

            state = env_state
            preview_colors = next_preview_colors

            if verbose:
                print(bitboards_to_board(state))

            if state["occupied_bits"].bit_count() >= NUM_CELLS:
                break

        scores.append(total_score)
        print(f"Evaluation game {game} | Score = {total_score}")

    print("\nEvaluation summary")
    print(f"Mean score = {np.mean(scores):.2f}")
    print(f"Max score  = {np.max(scores)}")
    print(f"Min score  = {np.min(scores)}")

    return scores


# --- RUN ---
def save_checkpoint(model, optimizer, training_scores, training_losses, training_steps, filename="five_or_more_checkpoint.pth"):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_scores": training_scores,
        "training_losses": training_losses,
        "training_steps": training_steps,
    }, filename)


def load_checkpoint(filename="five_or_more_checkpoint.pth"):
    model = ActorCritic(input_dim=11, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, checkpoint

# --- RUN ---

if __name__ == "__main__":
    RESUME_TRAINING = False   # Checkpoint Saver if True

    if RESUME_TRAINING:
        model, optimizer, checkpoint = load_checkpoint("five_or_more_checkpoint.pth")

        old_scores = checkpoint.get("training_scores", [])
        old_losses = checkpoint.get("training_losses", [])
        old_steps = checkpoint.get("training_steps", [])

        start_episode = len(old_scores) + 1

        trained_model, new_scores, new_losses, new_steps = train_actor_critic(
            num_episodes=50,
            gamma=0.99,
            lr=1e-4,
            entropy_coef=0.000,
            value_coef=0.5,
            print_every=1,
            plot_every=50,
            model=model,
            optimizer=optimizer,
            start_episode=start_episode,
        )

        training_scores = old_scores + new_scores
        training_losses = old_losses + new_losses
        training_steps = old_steps + new_steps

    else:
        model = ActorCritic(input_dim=11, hidden_dim=64)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        imitation_losses, imitation_scores = pretrain_actor_with_adaptive_expert(
            model=model,
            optimizer=optimizer,
            num_games=10,
            print_every=1,
        )

        print("\nEvaluation after imitation only:")
        evaluate_actor_critic(model, num_games=10, verbose=False)

        trained_model, training_scores, training_losses, training_steps = train_actor_critic(
            num_episodes=50,
            gamma=0.99,
            lr=1e-4,
            entropy_coef=0.0,
            value_coef=0.5,
            print_every=1,
            plot_every=50,
            model=model,
            optimizer=optimizer,
        )

    plot_training_progress(
        training_scores,
        training_losses,
        training_steps,
        window=10
    )

    plt.show()

    # Saves model for gameplay
    torch.save(trained_model.state_dict(), "five_or_more_actor_critic.pth")

    # Saves model + optimizer + training logs for resume training
    save_checkpoint(
        trained_model,
        optimizer,
        training_scores,
        training_losses,
        training_steps,
        filename="five_or_more_checkpoint.pth"
    )

    eval_scores = evaluate_actor_critic(
        trained_model,
        num_games=20,
        verbose=False
    )