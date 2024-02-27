"""
This Connect Four player uses Negamax with alpha-beta pruning to win (hopefully).
This file uses the mypy library in python to handle types.
"""
__author__ = "Alex" 
__license__ = "MIT"
__date__ = "February 2024"


import sys
from typing import Tuple, Optional, cast
from negamax import Node, negamax, negamax_alphabeta
from typing_extensions import override


class Connect4Node(Node):
    def __init__(self, rack: Tuple[Tuple[int, ...], ...]):
        self.rack = rack

    @override
    def eval(self) -> int:
        """
        Evaluate the position of the board
        """
        rack = self.rack
        n = len(rack)
        m = len(rack[0])
        total = 0

        def get_new_score(old_score: Optional[int], spot: int) -> int:
            """
            Given an old score and the value of a piece on the board, return a new score
            """
            spot_score = 1 if spot == 2 else -1
            if old_score == None:
                return spot_score
            old_score = cast(int, old_score)
            # if the score and the spot score are different signs, then we set our result to zero. Because we only change each score
            # by multiplying by 10, setting it to zero ensures that it will stay zero
            return old_score * 10 if old_score * spot_score > 0 else 0
        # for each row and column on the board. I am using the physical interpretation of row and column
        # as they would appear on a physical connect four board, (distict from the rows/cols of the rack object)
        for i in range(m): # rows
            for j in range(n): #cols
                # identify every possible connect four. None indicates that we have not found any 
                # lines of contiguous tokens starting at position j, i
                left, down, diag_ur, diag_ul = None, None, None, None
                # traverse in each direction from j,i
                for k in range(0, 4):
                    # if we are far enough to the right to have a connect-four going left, update our left score
                    if j >= 3 and rack[j - k][i]:
                        left = get_new_score(left, rack[j - k][i])
                    if i >= 3 and rack[j][i - k]:
                        down = get_new_score(down, rack[j][i - k])
                    if j >= 3 and i >= 3 and rack[j - k][i - k]: 
                        diag_ur = get_new_score(diag_ur, rack[j - k][i - k])
                    if j <= n - 4 and i >= 3 and rack[j + k][i - k]:
                        diag_ul = get_new_score(diag_ul, rack[j + k][i - k])  
                # sum the scores for each direction              
                dir_sum = (left if left else 0) + (down if down else 0) + (diag_ur if diag_ur else 0) + (diag_ul if diag_ul else 0)
                # determine if we have a connect-four
                if abs(dir_sum) >= 700:
                    return sys.maxsize * (abs(dir_sum)//dir_sum)
                total += dir_sum
        
        return total
    
            
    def get_child(self, move: int, color_to_play: int) -> Optional[Node]:
        """
        Get the child node obtained from dropping a token of color "color_to_play" (either 1 or -1) into slot "move"
        """
        tokenVal = 2 if color_to_play == 1 else 1
        replace_idx = -1
        col = self.rack[move]
        for i in range(len(col)):
            if col[i] == 0:
                replace_idx = i
                break
        if replace_idx == -1:
            return None
        else:
            next_col = tuple(tokenVal if i == replace_idx else col[i] for i in range(len(col)))
        next_rack = tuple(next_col if i == move else self.rack[i] for i in range(len(self.rack)))
        return Connect4Node(next_rack)
    
    
    @override
    def get_children(self, color_to_play: int) -> Tuple[Node,...]:
        """
        Get all valid children nodes
        """
        return tuple(child for i in range(len(self.rack)) if (child := self.get_child(i, color_to_play)))
    
    @override
    def __str__(self) -> str:
        return str(self.rack)

        
    



class ComputerPlayer:
    def __init__(self, id, difficulty_level):
        print(f'difficulty level: {difficulty_level}')
        print(f'computer playing as player {id}')
        """
        Constructor, takes a difficulty level (likely the # of plies to look
        ahead), and a player ID that's either 1 or 2 that tells the player what
        its number is.
        """
        self.color = -1 if id == 1 else 1
        self.difficulty_level = difficulty_level
        

    def pick_move(self, rack: Tuple[Tuple[int,...],...]) -> int:
        """
        Pick the move to make. It will be passed a rack with the current board
        layout, column-major. A 0 indicates no token is there, and 1 or 2
        indicate discs from the two players. Column 0 is on the left, and row 0 
        is on the bottom. It must return an int indicating in which column to 
        drop a disc.
        """
        # make the starting node
        starting_node = Connect4Node(rack)
        do_nega = lambda node: -negamax_alphabeta(node, self.difficulty_level - 1, -sys.maxsize, sys.maxsize, -self.color)


################################################################################# UNCOMMENT FOR NORMAL NEGAMAX :(  #########################################################
        #do_nega = lambda node: -negamax(node, self.difficulty_level - 1, -self.color)


        # enumerate each of the valid children with its corresponding move
        enumerated_children = [(child, i) for i in range(len(rack)) if (child := starting_node.get_child(i, self.color))]

        # figure out if any of the child states are winning on the spot. If so, do that:
        for child, i in enumerated_children:
            if self.color * child.eval() >= sys.maxsize:
                return i
        # sort the moves by score and return the best one depending on our color
        opts = sorted((do_nega(child), i) for child, i in enumerated_children)
        selected = opts[0][1] if self.color == -1 else opts[-1][1]
        return selected




##################################################################################### UNIT TESTS ###############################################################################
cp = ComputerPlayer(4, 2)
rack = ((0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0), (1, 1, 2, 2, 0, 0), (1, 2, 1, 1, 1, 0), (2, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0))
#print(f'computer picks move: {cp.pick_move(rack)}')
#print(f'SHOULD pick 4')

rack = (
    (0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0),
    (1, 1, 2, 2, 1, 1), 
    (2, 1, 1, 1, 2, 2), 
    (2, 1, 0, 0, 0, 0), 
    (0, 0, 0, 0, 0, 0))
#print(f'computer picks move: {cp.pick_move(rack)}')
#print(f'should pick 2 ')

rack = (
    (2, 2, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0),
    (1, 0, 0, 0, 0, 0), 
    (1, 2, 1, 1, 2, 2), 
    (2, 1, 1, 2, 1, 1), 
    (2, 2, 1, 1, 2, 1), 
    (0, 0, 0, 0, 0, 0))

#print(f'computer picks move: {cp.pick_move(rack)}')
#print(f'should NOT pick 2')


rack = (
    (0, 0, 0, 0, 0, 0),
      (0, 0, 0, 0, 0, 0), 
      (2, 0, 0, 0, 0, 0), 
      (1, 2, 1, 1, 1, 0), 
      (1, 2, 2, 0, 0, 0), 
      (0, 0, 0, 0, 0, 0), 
      (0, 0, 0, 0, 0, 0))
node = Connect4Node(rack)
#print(f'computer picks move: {cp.pick_move(rack)}')
#print(f'SHOULD pick 3')

# rack = (
#     (2, 1, 1, 1, 0, 0), 
#     (1, 2, 2, 2, 0, 0))
# node = Connect4Node(rack)
#print(f'computer picks move: {cp.pick_move(rack)}')
#print(f'SHOULD pick 1')

