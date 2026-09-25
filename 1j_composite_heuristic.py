# Heuristic: Composite Heuristic
# All the individual heuristics

#0 Imports
import numpy as np
from collections import deque
import random
import pygame


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
                                        
#1 (A) Single Directional row or column or diagnoal line potential Calculator
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

def line_potential_score(node):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]                                          ##Up-Down-DiagRight-DiagLeft

    total_line_potential = 0

    for direction in directions: 
        total_line_potential += directional_line_score(node, direction[0], direction[1])
    
    return total_line_potential


#1 (B) Directional Gap-Line Potential Calculator: scans all input direction length 5 windows and awards almost formed lines with a gap, e.g. R R R - R (so the line potential doesnt just purely award on consecutive lines)
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

def gap_potential_score(node):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    total_gap_line_potential = sum(directional_gap_potential_score(node, dr, dc) for dr, dc in directions)

    return total_gap_line_potential



#1 (C) Heuristic Score Calculator: Adjacency Path Based Mobility Scoring via BFS
def path_mobility_score(node):
    rows, cols = node.shape
    occupied = np.argwhere(node != 0)

    total_mobility = 0

    for sr, sc in occupied:
        color = node[sr, sc]

        # BFS from this piece through empty cells
        queue = deque([(sr, sc)])
        visited = {(sr, sc)}

        while queue:
            r, c = queue.popleft()

            for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    if node[nr, nc] == 0:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        # Count only reachable destinations that would land adjacent to same color
        for (r, c) in visited:
            if (r, c) == (sr, sc):
                continue  

            good = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        # ignore adjacency to itself at the source position
                        if (rr, cc) == (sr, sc):
                            continue
                        if node[rr, cc] == color:
                            good = True
                            break
                if good:
                    break

            if good:
                total_mobility += 1

    return total_mobility


#1 (D) Heuristic: Completion Access
def completion_access_score(node):
    rows, cols = node.shape
    total = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)] 

    for dr, dc in directions:
        for r in range(rows):
            for c in range(cols):
                end_r = r + 4 * dr
                end_c = c + 4 * dc

                if not (0 <= end_r < rows and 0 <= end_c < cols):
                    continue

                cells = [(r + i * dr, c + i * dc) for i in range(5)]
                window = [node[rr, cc] for rr, cc in cells]
                non_zero_colors = set(window) - {0}

                if len(non_zero_colors) != 1:
                    continue

                color = list(non_zero_colors)[0]
                n = sum(1 for x in window if x == color)

                if n != 4:
                    continue

                empty_targets = [(rr, cc) for rr, cc in cells if node[rr, cc] == 0]

                if len(empty_targets) != 1:
                    continue

                target = empty_targets[0]
                reachable_by_same_color = False

                for sr, sc in np.argwhere(node == color):
                    if (sr, sc) in cells:
                        continue

                    if check_move(node, (sr, sc), target):
                        reachable_by_same_color = True
                        break

                if reachable_by_same_color:
                    total += 40
                else:
                    total += 8

    return total







#2 Generate all legal moves and boards from current state. I.e. Generate all possible graph nodes transferable from current node + Move associated + Heuristic Score
def all_possible_states(current_node, alpha, beta, gamma, delta, epsilon, eta, k):
    occupied = np.argwhere(current_node != 0)
    empty = np.argwhere(current_node == 0)
    possible_states = []

    current_gap = gap_potential_score(current_node)
    current_path = path_mobility_score(current_node)
    current_access = completion_access_score(current_node)

    for sr, sc in occupied:
        for dr, dc in empty:
            if check_move(current_node, (sr, sc), (dr, dc)):
                temp_node = apply_move(current_node, (sr, sc), (dr, dc))
                new_node, cleared = five_more_clear(temp_node, (dr, dc))

                cleared_bonus = sum(POINT_SYSTEM.get(L, 0) for L in cleared)

                line_potential = line_potential_score(new_node)
                gap_potential = gap_potential_score(new_node)
                path_mobility = path_mobility_score(new_node)
                access_score = completion_access_score(new_node)

                delta_gap = gap_potential - current_gap
                delta_path = path_mobility - current_path
                delta_access = access_score - current_access

                heuristic_score = (
                    alpha * line_potential
                    + beta * gap_potential
                    + gamma * path_mobility
                    + delta * delta_gap
                    + epsilon * delta_path
                    + eta * delta_access
                    + k * cleared_bonus
                )

                possible_states.append((new_node, ((sr, sc), (dr, dc)), heuristic_score, cleared))

    return possible_states



#3 Greedy Algorithm
def greedy(current_node):
    states = all_possible_states(current_node, 0.4, 0.1, 0.005, 0.06, 0.001, 0.08, 9)

    if not states:
        return None, None, None, []

    best_h_score = max(s[2] for s in states)
    tied_states = [s for s in states if s[2] == best_h_score]

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

### Average: 
### Best: 
"""




# (3) Pygame-Based Visual Simulation for Qualitative Analysis / Interactive Human-AI Simulation Environment
'''
# --- GAME CONFIG --- 
ROWS, COLS = 9,9
CELL_SIZE = 80
HUD_BAR_HEIGHT = 50
COLOR_MAP = {                             ##Color Map: 0 = empty, 1 = white, 2 = red, 3 = green, 4 = blue, 5 = yellow, 6 = purple
    0: (30, 30, 30),     
    1: (255, 255, 255), 
    2: (255, 0, 0),     
    3: (0, 255, 0),     
    4: (0, 0, 255),     
    5: (255, 255, 0),    
    6: (160, 32, 240),
}
POINT_SYSTEM = {
    5: 10,
    6: 12,
    7: 18,
    8: 28,
    9: 42,
}

# --- HELPER FUNCTIONS ---
# (A) Applies a move given by human or AI decision making
def apply_turn(board, next_colors, score, move):

    (sr, sc), (dr, dc) = move

    # (0) Apply Move
    board = apply_move(board, (sr, sc), (dr, dc))

    #(1a) Clear any 5-or-more lines from moved tile
    board, len_cleared = five_more_clear(board, (dr, dc))

    #(1b) Update Score from Moved Tile
    for L in len_cleared:
        score += POINT_SYSTEM.get(L, 0)

    #(2) Spawn new tiles if no clear made from moved tile
    spawned_tiles = []
    if len_cleared == []:
        board, spawned_tiles = spawn_tiles(board, next_colors)
        next_colors = color_selector()
    
    #(3) For each new spawned tile, iteratively check and clear any 5-or-more lines formed by them respectively
        for (new_r, new_c) in spawned_tiles:
            if board[new_r, new_c] != 0:  
                board, len_cleared = five_more_clear(board, (new_r, new_c))

    #(3a) Update Score from Spawned Tiles
                for L in len_cleared:
                    score += POINT_SYSTEM.get(L, 0)

    return board, next_colors, score, spawned_tiles


# --- GAME INITIALIZATION --
# Initialize Pygame
pygame.init()

# Window Display 
screen = pygame.display.set_mode((ROWS*CELL_SIZE, COLS*CELL_SIZE + HUD_BAR_HEIGHT))
pygame.display.set_caption("Five-or-More")

# Initialize Starting Game Board
game_board = np.zeros((ROWS, COLS), dtype=int)
initial_colors = color_selector()
game_board, spawned_tiles = spawn_tiles(game_board, initial_colors)

# Next Three Colors 
next_colors = color_selector()

# Game Features
paused = False
mode = "AI"          # "AI" or "HUMAN"
AI_MOVE_MS = 0     # speed of AI animation 
ai_accum = 0         # AI step count

clock = pygame.time.Clock()

last_move = None              # ((sr, sc), (dr, dc))
last_spawned = []             # [(r,c), (r,c), (r,c)]
highlight_until = 0           # pygame time in ms
HIGHLIGHT_MS = 250            # how long highlights stay visible

#Score + Font
score = 0
font = pygame.font.SysFont(None, 32)

# Select Tile Object
selected = None

# Game loop
running = True
while running:
    dt = clock.tick(60)

    # --- EVENTS ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            # (1) Pause Button [Space Key]
            if event.key == pygame.K_SPACE:
                paused = not paused

            # (2a) Human Mode Toggle [H Key]
            elif event.key == pygame.K_h:
                mode = "HUMAN"
                paused = True     
                selected = None

            # (2b) AI Mode Toggle [A Key]
            elif event.key == pygame.K_a:
                mode = "AI"
                paused = False
                selected = None

            # (3) Force AI Next move Toggle [N Key]
            elif event.key == pygame.K_n:
                states = all_possible_states(game_board, 0.4, 0.1, 0.005, 0.06, 0.001, 0.08, 9)
                best_board, best_move, _ = greedy(game_board)
                game_board, next_colors, score, spawned_tiles = apply_turn(game_board, next_colors, score, best_move)

                last_move = best_move
                last_spawned = spawned_tiles
                highlight_until = pygame.time.get_ticks() + HIGHLIGHT_MS
                selected = None

        # (4) Human Mode Selection
        elif event.type == pygame.MOUSEBUTTONDOWN and mode == "HUMAN":
            x, y = pygame.mouse.get_pos()
            if y < HUD_BAR_HEIGHT:
                continue
            c = x // CELL_SIZE
            r = (y - HUD_BAR_HEIGHT) // CELL_SIZE

            if selected is None:
                if game_board[r, c] != 0:
                    selected = (r, c)
            else:
                sr, sc = selected
                if (sr, sc) == (r, c):
                    pass
                elif game_board[r, c] != 0:
                    selected = (r, c)
                elif check_move(game_board, (sr, sc), (r, c)):
                        #Apply turn with Human Decision Making
                        move = ((sr, sc), (r, c))
                        game_board, next_colors, score, spawned_tiles = apply_turn(game_board, next_colors, score, move)

                        last_move = move
                        last_spawned = spawned_tiles
                        highlight_until = pygame.time.get_ticks() + HIGHLIGHT_MS
                        selected = None

    # (5) AI Mode Steps (Time based)
    if mode == "AI" and (not paused):
        ai_accum += dt
        if ai_accum >= AI_MOVE_MS:
            ai_accum = 0

            if np.count_nonzero(game_board == 0) != 0:
                best_board, best_move, best_h_score, len_cleared = greedy(game_board)
                game_board, next_colors, score, spawned_tiles = apply_turn(game_board, next_colors, score, best_move)

                last_move = best_move
                last_spawned = spawned_tiles
                highlight_until = pygame.time.get_ticks() + HIGHLIGHT_MS


    # --- GRAPHICS ---
    screen.fill((0, 0, 0))

    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE + HUD_BAR_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (80, 80, 80), rect, 1)  
            
            val = game_board[r, c]
            if val != 0:
                padding = 10
                inner_rect = pygame.Rect(
                    c * CELL_SIZE + padding,
                    r * CELL_SIZE + padding + HUD_BAR_HEIGHT,
                    CELL_SIZE - 2*padding,
                    CELL_SIZE - 2*padding
                )
                pygame.draw.rect(screen, COLOR_MAP[val], inner_rect)

            if selected is not None and (r, c) == selected:
                pygame.draw.rect(screen, (255, 255, 255), rect, 3)

    # Draw score in top-right corner
    score_surf = font.render(f"Score: {score}", True, (255,255,255))
    score_x = COLS*CELL_SIZE - score_surf.get_width() - 10   # 10px margin
    score_y = (HUD_BAR_HEIGHT - score_surf.get_height()) // 2
    screen.blit(score_surf, (score_x, score_y))

    # Mode / pause info (left)
    status = f"Mode: {mode}    {'PAUSED' if paused else 'RUNNING'}"
    status_surf = font.render(status, True, (255, 255, 255))
    screen.blit(status_surf, (10, 10))

    # Controls hint
    hint = "SPACE - Pause | H - Human | A - AI | N - Next AI"
    hint_surf = font.render(hint, True, (200, 200, 200))
    screen.blit(hint_surf, (10, 38))

    # Draw "Next:" preview on the left
    label_surf = font.render("Next:", True, (255,255,255))
    screen.blit(label_surf, (10, (HUD_BAR_HEIGHT - label_surf.get_height()) // 2))

    # Small squares for each upcoming colour
    preview_size = 20
    spacing = 5
    start_x = 10 + label_surf.get_width() + 10
    center_y = HUD_BAR_HEIGHT // 2

    for i, col in enumerate(next_colors):
        px = start_x + i * (preview_size + spacing)
        py = center_y - preview_size // 2
        rect = pygame.Rect(px, py, preview_size, preview_size)
        pygame.draw.rect(screen, COLOR_MAP[col], rect)

    # Highlight move and spawned tiles
    # Always highlight last move / spawns
    if last_move is not None:
        (sr, sc), (dr, dc) = last_move
        pygame.draw.rect(screen, (255, 255, 0), pygame.Rect(sc * CELL_SIZE, sr * CELL_SIZE + HUD_BAR_HEIGHT, CELL_SIZE, CELL_SIZE), 5)
        pygame.draw.rect(screen, (255, 255, 0), pygame.Rect(dc * CELL_SIZE, dr * CELL_SIZE + HUD_BAR_HEIGHT, CELL_SIZE, CELL_SIZE), 5)

    for (r, c) in last_spawned:
        pygame.draw.rect(screen, (0, 150, 255), pygame.Rect(c * CELL_SIZE, r * CELL_SIZE + HUD_BAR_HEIGHT, CELL_SIZE, CELL_SIZE), 5)

    pygame.display.flip()


# Quit Pygame
pygame.quit()
'''






