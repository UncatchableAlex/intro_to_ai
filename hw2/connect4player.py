"""
This Connect Four player just picks a random spot to play. It's pretty dumb.
"""
__author__ = "Alex" # replace my name with yours
__license__ = "MIT"
__date__ = "February 2024"

import random
import time
import sys
from typing import Tuple, Optional, cast
from negamax import Node, negamax
from typing_extensions import override


class Connect4Node(Node):
    def __init__(self, rack: Tuple[Tuple[int, ...], ...], color: int):
        self.rack = rack
        self.color = color

    @override
    def eval(self) -> int:
            rack = self.rack
            n = len(rack)
            m = len(rack[0])
            total = 0
            
            def get_new_score(old_score: Optional[int], spot: int) -> int:
                spot_score = 1 if spot == 2 else -1
                if old_score == None:
                    return spot_score
                old_score = cast(int, old_score)
                return old_score * 10 if old_score * spot_score > 0 else 0
                    
            for i in range(m): # rows
                for j in range(n): #cols
                    left, down, diag_ul, diag_ur = None, None, None, None
                    for k in range(0, 4):
                        # if the current square is far enough to the left to count as the start of a quartet
                        if j >= 3 and rack[j - k][i]:
                            left = get_new_score(left, rack[j - k][i])
                        if i >= 3 and rack[j][i - k]:
                            down = get_new_score(down, rack[j][i - k])
                        if j >= 3 and i >= 3 and rack[j - k][i - k]: 
                            diag_ul = get_new_score(diag_ul, rack[j - k][i - k])
                        if j <= n - 4 and i >= 3 and rack[j + k][i - k]:
                            diag_ur = get_new_score(diag_ur, rack[j + k][i - k])                
                    dir_sum = (left if left else 0) + (down if down else 0) + (diag_ul if diag_ul else 0) + (diag_ur if diag_ur else 0)
                    if abs(dir_sum) >= 700:
                        return sys.maxsize * (abs(dir_sum)//dir_sum)
                    total += dir_sum
            
            return total
    
            
    def get_child(self, move: int) -> Optional[Node]:
        opponent = 2 if self.color == -1 else 1
        replace_idx = -1
        col = self.rack[move]
        for i in range(len(col)):
            if col[i] == 0:
                replace_idx = i
                break
        if replace_idx == -1:
            return None
        else:
            next_col = tuple(opponent if i == replace_idx else col[i] for i in range(len(col)))
        next_rack = tuple(next_col if i == move else self.rack[i] for i in range(len(self.rack)))
        return Connect4Node(next_rack, opponent)
    
    
    @override
    def get_children(self) -> Tuple[Node,...]:
        return tuple(child for i in range(len(self.rack)) if (child := self.get_child(i)))
    
    @override
    def __str__(self) -> str:
        return str(self.rack)
        
    



class ComputerPlayer:
    def __init__(self, id, difficulty_level):
        print(f'difficulty level: {difficulty_level}')
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
        drop a disc. The player current just pauses for half a second (for 
        effect), and then chooses a random valid move.
        """
        print(self.color)
        starting_node = Connect4Node(rack, -1*self.color)
        print(starting_node)
        eval_node = lambda node: negamax(node, self.difficulty_level, -sys.maxsize, sys.maxsize, -self.color) if node else -self.color * sys.maxsize

        opts = sorted((eval_node(starting_node.get_child(i)), i) for i in range(len(rack)))
        print(opts)
        return opts[0][1] if self.color == 1 else opts[-1][1]





cp = ComputerPlayer(1,2)
rack = (
    (2, 0, 0, 0, 0, 0), 
    (1, 2, 2, 1, 0, 0), 
    (2, 0, 2, 0, 0, 0), 
    (2, 2, 0, 1, 0, 0), 
    (0, 0, 2, 0, 0, 0), 
    (0, 0, 0, 2, 0, 0), 
    (0, 0, 0, 0, 2, 0))
node = Connect4Node(rack, 1)
cp.pick_move(rack)
