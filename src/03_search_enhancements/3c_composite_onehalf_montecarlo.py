# Statespace: Bitboard Representation 
# Search: Greedy Algorithm
# Objective Function: Composite Heuristic + Monte Carlo Simulation

#0 Imports
import numpy as np
from collections import deque, Counter
import random
import pygame


# --- GAME IMPLEMENTATION CODE ---

#0 Helper functions for Bitboard
BOARD_SIZE = 9
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
FULL_MASK = (1 << NUM_CELLS) - 1

#0 (a) Coordinates to Bits Index Function
def rc_to_idx(r, c):
    return r * BOARD_SIZE + c

#0 (b) Bits Index to Coordinates Function
def idx_to_rc(idx):
    return divmod(idx, BOARD_SIZE)

#0 (c) Returns mask with exactly one bit on, i.e. bit corresponding to coordinate/square (r,c)
def bit_at(r, c):
    return 1 << rc_to_idx(r, c)

#0 (d) Bitboard to Numpy
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

#0 (e) Bitboard Operation: Read Color (i) For Bit Index input
def get_color_at_idx(state, idx):
    mask = 1 << idx
    for color, bits in state["color_bits"].items():
        if bits & mask:
            return color
    return 0

#0 (e) Bitboard Operation: Read Color (ii) For (r,c) input
def get_color_at_rc(state, r, c):
    return get_color_at_idx(state, rc_to_idx(r, c))

#0 (f) Bitboard Operation: Set square/cell to specific color
def set_color_at_idx(state, idx, color):
    mask = 1 << idx

    # remove from all colours first
    for c in state["color_bits"]:
        state["color_bits"][c] &= ~mask

    if color == 0:
        state["occupied_bits"] &= ~mask
    else:
        state["color_bits"][color] |= mask
        state["occupied_bits"] |= mask

#0 (g) Bitboard Operation: Copy State
def copy_state(state):
    return {
        "color_bits": state["color_bits"].copy(),
        "occupied_bits": state["occupied_bits"]
    }


#1 (a) Initialising Empty board [Bitboard Rpresentation]
def empty_board_bits(num_colors=6):
    return {
        "color_bits": {color: 0 for color in range(1, num_colors + 1)},
        "occupied_bits": 0
    }


#2 (a) Next 3 Colors Selector/Preview
def color_selector():
    return [random.randint(1, 6) for _ in range(3)]

#2 (b) Spawn Tile Function
def spawn_tiles_bits(state, colors):
    # Get all empty cells as indices
    empty = list(iter_set_bits(FULL_MASK ^ state["occupied_bits"]))

    num_tiles = min(3, len(empty))
    chosen = random.sample(empty, num_tiles)

    new_state = copy_state(state)
    spawned_coords = []

    for i, idx in enumerate(chosen):
        color = colors[i]
        set_color_at_idx(new_state, idx, color)

        # convert to (row, col) for compatibility
        spawned_coords.append(idx_to_rc(idx))

    return new_state, spawned_coords


#3 (a) Action: Reachability/Legal-Move Checker (i) Edge Masks + Shift Helpers [Necessary to perform Flood Fill, which replaces BFS for reachability check]
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


#3 (a) Action: Reachability/Legal-Move Checker (ii) Flood Fill per source [Returns bitmask of all empty cells of input source]
def reachable_empty_bits(state, source_idx):
    source_bit = 1 << source_idx
    empty_bits = FULL_MASK ^ state["occupied_bits"]

    frontier = source_bit
    visited = source_bit

    while frontier:
        nbrs = (
            shift_left(frontier) |
            shift_right(frontier) |
            shift_up(frontier) |
            shift_down(frontier)
        ) & FULL_MASK

        new_frontier = nbrs & empty_bits & ~visited
        if new_frontier == 0:
            break

        visited |= new_frontier
        frontier = new_frontier

    return visited & empty_bits


#3 (a) Action: Reachability/Legal-Move Checker (iii) Boolean Move Check
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


#3 (b) Action: Move Color Tile
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

#4 Game Mechanic: 5-or-more Check and Clear Function [Checks on selected cell whether 5-or-More was formed around it]
def has_color_at_idx(state, color, idx):
    return ((state["color_bits"][color] >> idx) & 1) == 1

DIRECTIONS = [(0,1), (1,0), (1,1), (1,-1)]

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



# --- SEARCH ALGORITHM CODE --- 

#1 (A) Heuristic: Line Potential 
LINE_POINTS = {
    2: 1,
    3: 4,
    4: 10,
}

def build_ordered_lines():
    horizontal = []
    vertical = []
    diag_dr = []   # down-right
    diag_dl = []   # down-left

    # Horizontal
    for r in range(BOARD_SIZE):
        horizontal.append([rc_to_idx(r, c) for c in range(BOARD_SIZE)])

    # Vertical
    for c in range(BOARD_SIZE):
        vertical.append([rc_to_idx(r, c) for r in range(BOARD_SIZE)])

    # Down-right diagonals
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

    # Down-left diagonals
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

        # find maximal contiguous run of this colour
        j = i
        while j + 1 < n and get_color_at_idx(state, line[j + 1]) == color:
            j += 1

        run_length = j - i + 1

        # only score runs that are in LINE_POINTS
        if run_length not in LINE_POINTS:
            i = j + 1
            continue

        base = LINE_POINTS[run_length]

        # expand to maximal segment containing only {same colour, empty}
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

        left_open = (i - 1 >= 0 and get_color_at_idx(state, line[i - 1]) == 0)
        right_open = (j + 1 < n and get_color_at_idx(state, line[j + 1]) == 0)

        empties_in_segment = 0
        for k in range(L, R + 1):
            if get_color_at_idx(state, line[k]) == 0:
                empties_in_segment += 1

        # baseline: dead runs still count
        score = base

        if segment_length >= 5:
            if left_open and right_open:
                # fully open feasible
                score = 1.6 * base + 0.4 * empties_in_segment
            elif left_open or right_open:
                # semi-open feasible
                score = 1.35 * base + 0.25 * empties_in_segment
            else:
                # feasible but locally boxed
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



#1 (B) Heuristic: Gap Potential
def build_five_cell_windows():
    windows = []
    directions = [(0,1), (1,0), (1,1), (1,-1)]

    for dr, dc in directions:
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

        # longest run along ordered cells
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


#1 (C) Heuristic: Adjacency Path Based Mobility 
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

            # ignore the source tile, since it would move away
            same_color_neighbours = state["color_bits"][color] & neighbour_mask & ~(1 << source_idx)

            if same_color_neighbours != 0:
                total += 1

    return total


#1 (D) Heuristic: Completion Acesss
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

        # Only consider one-color + empty windows
        if len(colors_present) != 1:
            continue

        color = colors_present[0]
        color_bits_in_window = state["color_bits"][color] & mask
        n = popcount(color_bits_in_window)

        # First implementation: only care about 4-of-5 windows
        if n != 4:
            continue

        # Find the empty target square in the window
        empty_targets = []
        for idx in cells:
            if get_color_at_idx(state, idx) == 0:
                empty_targets.append(idx)

        if len(empty_targets) != 1:
            continue

        target_idx = empty_targets[0]

        # Check if some OTHER same-colored piece can reach the target
        reachable_by_same_color = False

        for source_idx in iter_set_bits(state["color_bits"][color]):
            if source_idx in cells:
                continue  # skip pieces already in the 4-line window

            reachable = reachable_empty_bits(state, source_idx)
            if ((reachable >> target_idx) & 1) == 1:
                reachable_by_same_color = True
                break

        if reachable_by_same_color:
            total += 40   # real, actionable 4-of-5
        else:
            total += 8    # structurally good, but currently inaccessible

    return total



#2 Generate all legal moves and boards from current state. I.e. Generate all possible graph nodes transferable from current node + Move associated + Heuristic Score
def all_possible_states_bits(state, alpha, beta, gamma, delta, epsilon, eta, k):
    possible_states = []

    current_gap = gap_potential_score_bits(state)
    current_path = path_mobility_score_bits(state)
    current_access = completion_access_score_bits(state)

    for source_idx in iter_set_bits(state["occupied_bits"]):
        reachable = reachable_empty_bits(state, source_idx)

        for dest_idx in iter_set_bits(reachable):
            temp_state = apply_move_bits(state, source_idx, dest_idx)
            cleared_state, cleared = five_more_clear_bits(temp_state, idx_to_rc(dest_idx))

            cleared_bonus = sum(POINT_SYSTEM.get(L, 0) for L in cleared)
            line_potential = line_potential_score_bits(cleared_state)
            gap_potential = gap_potential_score_bits(cleared_state)
            path_mobility = path_mobility_score_bits(cleared_state)

            access_score = completion_access_score_bits(cleared_state)
            delta_gap = gap_potential - current_gap
            delta_path = path_mobility - current_path
            delta_access = access_score - current_access

            heuristic_score = (
                alpha * line_potential
                + beta * gap_potential
                + gamma * path_mobility
                + delta * delta_gap
                + epsilon* delta_path
                + eta * delta_access
                + k * cleared_bonus
            )

            move = (idx_to_rc(source_idx), idx_to_rc(dest_idx))
            possible_states.append((cleared_state, move, heuristic_score, cleared))

    return possible_states



#3 Monte Carlo Simulator: Note. only initiate if no clear. Note. Input should be post move board
def best_one_step_greedy_score_bits(state, alpha, beta, gamma, delta, epsilon, eta, k):
    states = all_possible_states_bits(
        state,
        alpha, beta, gamma, delta, epsilon, eta, k
    )

    if not states:
        return -1e18

    return max(s[2] for s in states)


def monte_carlo_spawn_then_greedy_bits(
    state,
    next_colors,
    alpha, beta, gamma, delta, epsilon, eta, k,
    num_rollouts=30
):
    scores = []

    for _ in range(num_rollouts):
        sim_state = copy_state(state)

        # simulate random spawn using the actual preview colours
        sim_state, spawned_tiles = spawn_tiles_bits(sim_state, next_colors)

        spawn_score = 0

        # clear any lines formed by spawned tiles
        for new_r, new_c in spawned_tiles:
            if get_color_at_rc(sim_state, new_r, new_c) != 0:
                sim_state, spawned_cleared = five_more_clear_bits(
                    sim_state,
                    (new_r, new_c)
                )

                for L in spawned_cleared:
                    spawn_score += POINT_SYSTEM.get(L, 0)

        # one-step greedy evaluation after the simulated spawn
        future_score = best_one_step_greedy_score_bits(
            sim_state,
            alpha, beta, gamma, delta, epsilon, eta, k
        )

        scores.append(spawn_score + future_score)

    return float(np.mean(scores))


#4 Greedy Algorithm
def greedy(current_node, next_colors, alpha=0.4, beta=0.1, gamma=0.005,
           delta=0.06, epsilon=0.001, eta=0.08, k=9,
           top_k=10, num_rollouts=10, monte_carlo_weight=1.0,
           mc_fill_threshold=0.5):

    states = all_possible_states_bits(
        current_node,
        alpha, beta, gamma, delta, epsilon, eta, k
    )

    if not states:
        return None, None, None, []

    # Beam Top-K pruning: only Monte Carlo evaluate the top candidates
    states.sort(key=lambda x: x[2], reverse=True)
    candidate_states = states[:top_k]

    # Check if theres any clear moves available in candidates
    any_clear = any(len(cleared) > 0 for _, _, _, cleared in candidate_states)
    fill_ratio = current_node["occupied_bits"].bit_count() / NUM_CELLS
    high_stress = fill_ratio >= mc_fill_threshold
    use_monte_carlo = (not any_clear) and high_stress

    reranked_states = []

    for board_state, move, heuristic_score, cleared in candidate_states:

        if use_monte_carlo:
            mc_score = monte_carlo_spawn_then_greedy_bits(
                board_state,
                next_colors,
                alpha, beta, gamma, delta, epsilon, eta, k,
                num_rollouts=num_rollouts
            )

            final_score = heuristic_score + monte_carlo_weight * mc_score
        else:
            final_score = heuristic_score

        reranked_states.append((board_state, move, final_score, cleared))

    best_h_score = max(s[2] for s in reranked_states)
    tied_states = [s for s in reranked_states if s[2] == best_h_score]

    best_board, best_move, best_h_score, cleared = random.choice(tied_states)

    return best_board, best_move, best_h_score, cleared



# --- GREEDY TEST CODE ---
POINT_SYSTEM = {
    5: 10,
    6: 12,
    7: 18,
    8: 28,
    9: 42,
}

def run_single_game(alpha=0.4, beta=0.1, gamma=0.005,
                    delta=0.06, epsilon=0.001, eta=0.08, k=9,
                    print_game=False):

    game_board = empty_board_bits()
    initial_colors = color_selector()
    next_colors = color_selector()
    game_board, spawned_tiles = spawn_tiles_bits(game_board, initial_colors)

    score = 0

    while game_board["occupied_bits"].bit_count() < NUM_CELLS:

        best_board, best_move, _, len_cleared = greedy(
            game_board,
            next_colors,
            alpha, beta, gamma, delta, epsilon, eta, k
        )

        if best_board is None:
            break

        game_board = best_board

        for L in len_cleared:
            score += POINT_SYSTEM.get(L, 0)

        if len_cleared == []:
            game_board, spawned_tiles = spawn_tiles_bits(game_board, next_colors)
            next_colors = color_selector()

            for (new_r, new_c) in spawned_tiles:
                if get_color_at_rc(game_board, new_r, new_c) != 0:
                    game_board, spawned_cleared = five_more_clear_bits(
                        game_board,
                        (new_r, new_c)
                    )

                    for L in spawned_cleared:
                        score += POINT_SYSTEM.get(L, 0)

        if print_game:
            print(bitboards_to_board(game_board))

    print(score)
    return score



# (1) Verification Check: Single Simulation Test Code
run_single_game(print_game=True)

# (2) 20 Runs Test
"""
num_runs = 20
scores = []

for i in range(num_runs):
    score = run_single_game()
    scores.append(score)
    print(f"Game {i+1} | Score = {score}")

avg_score = np.mean(scores)
best_score = np.max(scores)

print("\nSummary over 20 games")
print(f"Average score = {avg_score:.2f}")
print(f"Best score    = {best_score}")
"""


