#Five-or-More Abstraction

#0 Imports
import numpy as np
from collections import deque
import pygame

board = [
    [0, 0, 0, 0, 0, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 0, 5, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 4, 0, 0, 0, 0, 0, 6, 0],
    [0, 0, 0, 4, 2, 1, 0, 6, 0],
    [3, 0, 3, 0, 0, 0, 0, 6, 0],
    [0, 0, 0, 5, 0, 0, 0, 0, 0],
]

board_np = np.array(board)
print(board_np)

#1 (a) Initialise Empty Board
empty_board_np = np.zeros((5, 5), dtype=int)

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




# --- TESTING --- [Basic Starting Game Generation & Tile Movement/Action]
#[Initial Board Generation] print(random_tile_generator(empty_board_np))
#board = np.array([ 
#    [1, 1, 0, 0, 0],
#    [0, 0, 0, 0, 0],
#    [0, 4, 0, 3, 6],
#    [0, 0, 0, 4, 0],
#    [0, 0, 0, 5, 0]
#])
#print(legal_move(board, (0,0), (0,4)))
#new_board = apply_move(board, (0,0), (1,4))
#print(new_board)

# --- TESTING --- [Clear Five_More Clear Function]
# board = np.array([ 
#    [1, 1, 1, 1, 1],
#    [0, 0, 1, 0, 0],
#    [0, 4, 1, 3, 6],
#    [0, 0, 1, 4, 0],
#    [0, 0, 1, 5, 0]
# ])
# new_board = five_more_clear(board, (0,2))
# print(new_board)




# --- Game Config --- 
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

#Score + Font
score = 0
font = pygame.font.SysFont(None, 32)

# Select Tile Object
selected = None

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
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
                        #(1a) Clear any 5-or-more lines from moved tile
                        game_board = apply_move(game_board, (sr, sc), (r, c))
                        game_board, len_cleared = five_more_clear(game_board, (r, c))

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

                        selected = None  

    # Graphics
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


    pygame.display.flip()


# Quit Pygame
pygame.quit()

