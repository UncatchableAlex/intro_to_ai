"""
Sliding Puzzle

This program will create a sliding puzzle (like the 15-puzzle), given a user-
supplied .puz file. Assuming that it can find the tkinter module, it will
animate the solution found.

The solution must be provided by a function called solve() inside of solver.py.
This function takes one argument: a row-major 2D list of the layout of the
board. The list must be rectangular, and every value should be a positive int
save the gap, which has a value of 0. It returns a list containing the solution.
Each element in the solution list is a str, that contains a 'U', 'D', 'L', or
'R'.  For example, the solution returned for easy.puz should be 
['L', 'L', 'L', 'U']. If there is no solution (as is the case for half of all 
initial setups), solve() should return None.
"""

import sys

################################################################################
# CONSTANTS
################################################################################

TILE_COLOR = "#FFC080"
NUMBER_COLOR = "#804000"
BACKGROUND_COLOR = "black"
TILE_SIZE = 200
TILE_FONT = ("Arial", 100, "bold")
ERROR_COLOR = "#002080"
ERROR_FONT = ("Arial", 70, "bold italic")

MOVE_TIME = 250
PAUSE_TIME = 125
FRAME_TIME = 10

################################################################################
# GRAPHICAL OBJECTS TO USE
################################################################################

# Does tkinter even exist on this computer? If not, turn off graphics.
try:
    import tkinter as tk
    do_graphics = True
    
    class Board(tk.Frame):
        def __init__(self, window, layout, solution):
            tk.Frame.__init__(self, window)

            # init fields
            self.solution = solution
            self.step = 0
            self.layout = layout
            
            self.canvas = tk.Canvas(width = len(layout[0]) * TILE_SIZE,
                                    height = len(layout) * TILE_SIZE,
                                    background = BACKGROUND_COLOR)
            self.canvas.pack()

            # define the Tile objects (and find the gap)
            self.tiles = [None] * (len(layout)*len(layout[0]))
            for r in range(len(layout)):
                for c in range(len(layout[r])):
                    if layout[r][c] == 0: self.gap = (r,c)
                    else: self.tiles[layout[r][c]] = Tile(self.canvas, layout[r][c], (r, c))

            if solution == None:
                self.canvas.create_text(len(layout[0]) * TILE_SIZE /2, len(layout) * TILE_SIZE /3, text=str("Unsolvable!"), font=ERROR_FONT, fill=ERROR_COLOR)

                    
        def start_animation_logic(self):
            """
            Start the logic up, to handle animation.
            """
            if self.solution != None: self.after(0, self._animate_frame)

        # do a single frame
        def _animate_frame(self):
            if self.step >= len(self.solution): return #stop animation if we've reached the end
            
            delta = float(FRAME_TIME / MOVE_TIME)

            # if we have an int, it's time to set up the next step
            if type(self.step) == int:
                # ID the elements to move
                self.current_move = self.solution[self.step]
                if self.current_move == 'L':
                    self.moving_tile = self.layout[self.gap[0]][self.gap[1]+1]
                    self.deltas = (0, -delta)
                elif self.current_move == 'R':
                    self.moving_tile = self.layout[self.gap[0]][self.gap[1]-1]
                    self.deltas = (0, delta)
                elif self.current_move == 'U':
                    self.moving_tile = self.layout[self.gap[0]+1][self.gap[1]]
                    self.deltas = (-delta, 0)
                elif self.current_move == 'D':
                    self.moving_tile = self.layout[self.gap[0]-1][self.gap[1]]
                    self.deltas = (delta, 0)

            # advance
            new_step = self.step + delta

            # see if we just finished sliding a tile--if so, need to update layout
            if int(new_step) > int(self.step):
                self.step = int(new_step)
                self.tiles[self.moving_tile].jump_to(self.gap)

                # where's the new gap?
                if self.current_move == 'L': new_gap = (self.gap[0], self.gap[1]+1)
                elif self.current_move == 'R': new_gap = (self.gap[0], self.gap[1]-1)
                elif self.current_move == 'U': new_gap = (self.gap[0]+1, self.gap[1])
                elif self.current_move == 'D': new_gap = (self.gap[0]-1, self.gap[1])

                # adjust the layout
                self.layout[self.gap[0]][self.gap[1]] = self.layout[new_gap[0]][new_gap[1]]
                self.layout[new_gap[0]][new_gap[1]] = 0
                self.gap = new_gap

                # pause a bit after each move
                self.after(PAUSE_TIME, self._animate_frame)

            # just continue the moving that was already going
            else:
                self.step = new_step
                self.tiles[self.moving_tile].move(self.deltas)
                self.after(FRAME_TIME, self._animate_frame)

    class Tile:
        """
        Graphical tile within the puzzle.
        """
        def __init__(self, canvas, value, coords):
            """
            Constructor, setting everything up.
            """
            self.value = value
            self.rect = canvas.create_rectangle(1, 1, TILE_SIZE+1, TILE_SIZE+1, fill=TILE_COLOR, width=2)
            self.text = canvas.create_text(TILE_SIZE/2, TILE_SIZE/2, text=str(value), font=TILE_FONT, fill=NUMBER_COLOR)
            self.coords = (0,0)
            self.canvas = canvas
            self.jump_to(coords)

        def jump_to(self, coords):
            """
            Move to some absolute position.
            """
            delta_y = (coords[0] - self.coords[0]) * TILE_SIZE
            delta_x = (coords[1] - self.coords[1]) * TILE_SIZE
            self.coords = coords
            self.canvas.move(self.rect, delta_x, delta_y)
            self.canvas.move(self.text, delta_x, delta_y)

        def move(self, deltas):
            """
            Move to a relative position.
            """
            self.coords = (self.coords[0]+deltas[0], self.coords[1]+deltas[1])
            delta_y = deltas[0] * TILE_SIZE
            delta_x = deltas[1] * TILE_SIZE
            self.canvas.move(self.rect, delta_x, delta_y)
            self.canvas.move(self.text, delta_x, delta_y)

except:
    print("Warning: Could not find the tkinter module. Graphics disabled.", file=sys.stderr)
    do_graphics = False

# given some token from the file, cast it as an int or return 0 if it's a ./_/x
def _parse_token(token):
    if token == '.' or token == "_" or token == "x": return 0
    else: return int(token)
    
################################################################################
# MAIN CODE--INITIAL CHECKS, IMPORTING OF MODULES
################################################################################

# check that there is a valid solve() function
try:
    from solver import solve
except:
    print("You need to supply a file \"solver.py\" with a proper solve() function!", file=sys.stderr)
    sys.exit(1)

# see if the user wants to squelch the graphics
args = sys.argv[1:]
while "--nographics" in args or "-n" in args:
    if "--nographics" in args: index = args.index("--nographics")
    else: index = args.index("-n")

    del args[index]
    do_graphics = False

# check that we have a valid input from the user
if len(args) == 0:
    print("Usage: python3 " +sys.argv[0]+ " (-n/--nographics) <puz file>", file=sys.stderr)
    print("       The -n or --nographics flags will turn off the graphical display.", file=sys.stderr)
    sys.exit(1)

################################################################################
# OKAY--ACTUALLY RUN THE SILLY THING
################################################################################

# parse the input file
with open(args[0]) as f:
    puzzle_data = [[_parse_token(x) for x in line.split()] for line in f]

# solve the puzzle using the student's solve() inside of solver.py
solution = solve(puzzle_data)
print("Solution: ", end="")
if solution == None: print("Impossible!")
else:
    for s in solution: print(s, end="")
    print(" (" +str(len(solution)) + ")")

# actual graphical stuff
if do_graphics:
    # set up the window containing everything
    window = tk.Tk()
    window.wm_title("Sliding Puzzle")

    # init the Board inside the window, and go!
    board = Board(window, puzzle_data, solution)
    board.start_animation_logic()
    board.mainloop()