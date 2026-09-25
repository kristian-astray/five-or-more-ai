# Heuristic: Clustering + [Line + Gap + Clear]
# Utilizes centroid variance distance measuring, favouring board states with more compact color configurations. Common heuristic demonstrated to be effective in many game AIs

#0 Imports
import numpy as np
from collections import deque
import random


# --- GAME IMPLEMENTATION CODE ---

#1 (a) Initialise Empty Board REDUNDANT
#empty_board_np = np.zeros((5, 5), dtype=int)

#1 (b) Next 3 Colors/Tiles Selector
def color_selector():
    num_colors = 6 
    return [np.random.randint(1, num_colors + 1) for i in range(3)]

#1 (c) Spawn Tile Function
def spawn_tiles(board, colors):
    empty_cells = np.argwhere(board==0)                                                     ##Finds empty/avaiable positions to spawn random tiles 
    num_tiles = min(3, len(empty_cells))                                                    ##Simulates EndGame, if number of empty cells is less than 3, then spawn remaining   

    random_index = np.random.choice(len(empty_cells), size=num_tiles, replace=False)  

    spawned_tiles_coord = []                                                                ##Need this for applying 5_or_more check later on to each coordinate each of the three tiles was spawned in 

    for i, index in enumerate(random_index):
        row , column = empty_cells[index]
        board[row, column] = colors[i]
        spawned_tiles_coord.append((row,column))
    
    return board, spawned_tiles_coord

#2 (a) Action: Reachability/Legal-Move Checker
def check_move(board, source, destination):
    (sr, sc) = source
    (dr, dc) = destination

    if board[sr, sc] == 0:                                                                  ##Source has to be a valid occupied tile to move
        return False
    if source == destination:                                                               ##Source can't be the same as destination   
        return False
    if board[dr, dc] != 0:                                                                  ##Destination must be an available square
        return False
    
    rows, cols = board.shape
    queue = deque([source])
    visited = {source}

    while queue:
        r, c = queue.popleft()

        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if (nr, nc) == destination:
                    return True
                if board[nr, nc] == 0:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    return False

#2 (c) Action: Move Tile
def apply_move(board, source, destination):
    sr, sc = source
    dr, dc = destination
    new_board = board.copy()
    new_board[dr, dc] = new_board[sr, sc]
    new_board[sr, sc] = 0
    return new_board

#3 Game Mechanic: 5-or-more Check and Clear Function [Checks on selected cell whether 5-or-More was formed around it]
def five_more_clear(board, cell):
    r, c = cell
    color = board[r, c]

    rows, cols = board.shape
    clear_lines = np.zeros(board.shape, dtype=bool)                                         ##BooLean Mask to clear lines later
    len_cleared = []                                                                        ##Track Lengths Cleared for Scoring

    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]                                          ##Up-Down-DiagRight-DiagLeft

    for d in directions:                                                                    
        dr, dc = d 
        line_form = [(r, c)]
        
        #Positive direction
        adjacent_r, adjacent_c = r + dr, c + dc
        while 0 <= adjacent_r < rows and 0 <= adjacent_c < cols and board[adjacent_r,adjacent_c] == color:
            line_form.append((adjacent_r, adjacent_c))
            adjacent_r += dr
            adjacent_c += dc
  
        #Negative Direction
        adjacent_r, adjacent_c = r - dr, c - dc
        while 0 <= adjacent_r < rows and 0 <= adjacent_c < cols and board[adjacent_r,adjacent_c] == color:
            line_form.append((adjacent_r, adjacent_c))
            adjacent_r -= dr
            adjacent_c -= dc

        #Clear Condition [Note. we only clear after finding the union of lines in all directions, score is awarded by the length of the each line in the set]
        if len(line_form) >= 5:
            len_cleared.append(len(line_form))
            for (rr, cc) in line_form:
                clear_lines[rr, cc] = True

    new_board = board.copy()
    new_board[clear_lines] = 0
    return new_board, len_cleared




# --- SEARCH ALGORITHM CODE ---

#1 (a) Single Directional row or column or diagnoal line potential Calculator
def directional_line_score(node, dr, dc):

    rows, cols = node.shape

    line_points = {
        2: 1,
        3: 4,
        4: 10,
        5: 22,
        6: 48,
    }

    total = 0

    for r in range(rows):
        for c in range(cols):
            color = node[r, c]

            #(1) 1st Logic: if empty square. Skip no line to be formed from here
            if color == 0:
                continue

            #(2) 2nd Logic: if the square behind is of the same color, then we know we already counted this in a previous loop. Skip to avoid double counting
            prev_r, prev_c = r - dr, c - dc
            if 0 <= prev_r < rows and 0 <= prev_c < cols and node[prev_r, prev_c] == color:
                continue

            #(3) 3rd Logic: while next tile in direction dr,dc is in range and of same color, count into line length
            length = 1
            next_r, next_c = r + dr, c + dc
            while 0 <= next_r < rows and 0 <= next_c < cols and node[next_r, next_c] == color:
                length += 1
                next_r += dr
                next_c += dc

            if length in line_points:
                total += line_points[length]   

    return total                                                


#1 (b) Heuristic Score Calculator: Total line Potential Calculator
def line_potential_score(node):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]                                          ##Up-Down-DiagRight-DiagLeft

    total_line_potential = 0

    for direction in directions: 
        total_line_potential += directional_line_score(node, direction[0], direction[1])
    
    return total_line_potential

#1 (c) Directional Gap-Line Potential Calculator: scans all input direction length 5 windows and awards almost formed lines with a gap, e.g. R R R - R (so the line potential doesnt just purely award on consecutive lines)
def directional_gap_potential_score(node, dr, dc):

    rows, cols = node.shape

    five_minus_gap_points = {
        4: 60, 
        3: 12, 
        2: 2
        }
    
    total = 0

    for r in range(rows):
        for c in range(cols):

            #(1) Ensures window is within board boundaries
            end_r = r + 4 * dr
            end_c = c + 4 * dc
            if not (0 <= end_r < rows and 0 <= end_c < cols):
                continue


            #(2) Extract window and it's set to observe which colors exist in this window. E.g. [4, 4, 0, 3, 4] --> (0, 3, 4)
            window = [node[r + i*dr, c + i*dc] for i in range(5)]
            window_set = set(window)

            #(3) If window is just empty, i.e. set only contains 0, then skip
            if window_set == {0}:
                continue

            #(4) If window contains two or more separate colors, then skip 
            non_zero_colors = window_set - {0}
            if len(non_zero_colors) > 1:
                continue
            
            #(5) Iterate through the single color in the window, except empties i.e. value=0, to look for potential consecutive lines with empty gaps in between
            color = list(non_zero_colors)[0]

            #(6) Count how many of that color are in the window
            n = sum(1 for x in window if x == color)
            if n not in five_minus_gap_points:
                continue
            base_score = five_minus_gap_points[n]

            #(7) Contiguity bonus: prefer clustered patterns (RRR__ over R_R_R)
            max_run = run = 0
            for x in window:
                if x == color:
                        run += 1
                        max_run = max(max_run, run)
                else:
                    run = 0

            total += base_score + 2 * max_run
           
    return total

#1 (d) Total Gap-Line Potential Calculator
def gap_potential_score(node):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    total_gap_line_potential = sum(directional_gap_potential_score(node, dr, dc) for dr, dc in directions)

    return total_gap_line_potential


#1 (e) Heuristic Centroid Variance Via Euclidean Distance
def centroid_variance_clustering(node):

    cluster_variance_score = 0 
    
    for color in range(1,7):

        positions = np.argwhere(node == color)

        if len(positions) == 0 or len(positions) == 1:
            continue

        #(1) Find Centroid
        centroid = positions.mean(axis=0)

        #(2) Calculate Squared Euclidean Distance for each tile
        difference = positions - centroid
        square_distances = (difference ** 2).sum(axis=1)

        #(3) Average distance
        variance = square_distances.mean()

        cluster_variance_score += 1.0 / (1.0 + variance)

    return cluster_variance_score
    


#2 Generate all legal moves and boards from current state. I.e. Generate all possible graph nodes transferable from current node + Move associated + Heuristic Score
def all_possible_states(current_node):
    occupied = np.argwhere(current_node != 0)
    empty = np.argwhere(current_node == 0)
    possible_states = []                                                                    ##List of (future_board, (source, destination), heuristic)

    for sr, sc in occupied:
        for dr, dc in empty:
            if check_move(current_node, (sr, sc), (dr, dc)):
                temp_node = apply_move(current_node, (sr, sc), (dr, dc))
                new_node, cleared = five_more_clear(temp_node, (dr,dc))
                line_potential = line_potential_score(new_node)
                gap_potential = gap_potential_score(new_node)
                cluster_score = centroid_variance_clustering(new_node)

                cleared_bonus = 5 * sum(POINT_SYSTEM.get(L, 0) for L in cleared)
                heuristic_score = line_potential + 0.1 * gap_potential + 10 * cluster_score + cleared_bonus                           

                possible_states.append((new_node, ((sr, sc), (dr, dc)), heuristic_score, cleared)) 

    return possible_states


#3 Greedy Algorithm
def greedy(current_node):
    states = all_possible_states(current_node)

    best_h_score = max(s[2] for s in states)
    tied_states = [s for s in states if s[2] == best_h_score]

    best_board, best_move, best_score, cleared = random.choice(tied_states)                   ## Random Tie-Breaking

    return best_board, best_move, best_score, cleared




# --- GREEDY TEST CODE ---
POINT_SYSTEM = {
    5: 10,
    6: 12,
    7: 18,
    8: 28,
    9: 42,
}

def run_single_game(print_game=False):
    game_board = np.zeros((9, 9), dtype=int)
    initial_colors = color_selector()
    next_colors = color_selector()
    game_board, spawned_tiles = spawn_tiles(game_board, initial_colors)

    score = 0

    while np.count_nonzero(game_board == 0) != 0:
        best_board, best_move, _, len_cleared = greedy(game_board)

        if best_board is None:
            break

        source = best_move[0]
        destination = best_move[1]

        #(1a) Apply move (best_board already includes move + clearing from moved tile)
        game_board = best_board

        #(1b) Update Score from Moved Tile
        for L in len_cleared:
            score += POINT_SYSTEM.get(L, 0)

        #(2) Spawn new tiles if no clear made from moved tile
        if len_cleared == []:
            game_board, spawned_tiles = spawn_tiles(game_board, next_colors)
            next_colors = color_selector()

            #(3) For each new spawned tile, iteratively check and clear any 5-or-more lines formed by them respectively
            for (new_r, new_c) in spawned_tiles:
                if game_board[new_r, new_c] != 0:
                    game_board, len_cleared = five_more_clear(game_board, (new_r, new_c))

                    #(3a) Update Score from Spawned Tiles
                    for L in len_cleared:
                        score += POINT_SYSTEM.get(L, 0)

        if print_game:
            print(game_board)

    print(score)
    return score


# (1) Verification Check: Single Simulation Test Code
"""
run_single_game(print_game=True)
"""


# (2) 100 Runs Test
num_runs = 100
scores = []

for i in range(num_runs):
    score = run_single_game()
    scores.append(score)
    print(f"Game {i+1} | Score = {score}")

avg_score = np.mean(scores)
best_score = np.max(scores)

print("\nSummary over 100 games")
print(f"Average score = {avg_score:.2f}")                                                    
print(f"Best score    = {best_score}")

