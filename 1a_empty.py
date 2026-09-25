# Heuristic: Empty Cells
# Counts the number of empty cells on the resulting board state from a move

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

#1 Heuristic Score Calculator: Empty square counter
def h_score(node):
    return np.count_nonzero(node == 0)


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

                heuristic_score = h_score(new_node)                       

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
run_single_game(print_game=True)


# (2) 100 Runs Test
"""
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
"""

