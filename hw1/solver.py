############################             NOTE: I used a library called mypy to handle typing.              #############################################

from a_star import State, a_star
from typing import Tuple, List, Dict
from typing_extensions import override
from random import randint,seed
from time import time


# the state object that we will feed our a_star algorithm designed to represent the 15 puzzle
class FifteenState(State):
    def __init__(self, board: Tuple[Tuple[int,...],...], gap: Tuple[int,int], 
                 vertical_walking_distances: Dict[Tuple[Tuple[int,...],...], int] = dict(), 
                 horizontal_walking_distances: Dict[Tuple[Tuple[int,...],...], int] = dict()) -> None :
        self.m,self.n = len(board), len(board[0])
        self.board = board
        self.gap = gap
        self.vertical_walking_distances = vertical_walking_distances
        self.horizontal_walking_distances = horizontal_walking_distances
        
    # return a fifteenstate object with the gap swapped with a tile at "pos"
    def swap_tile_with_gap(self, pos: Tuple[int,int]) -> State:
        def get_tile(subpos: Tuple[int,int]) -> int:
            if subpos != self.gap and subpos != pos:
                return self.board[subpos[0]][subpos[1]]
            elif subpos == self.gap:
                return self.board[pos[0]][pos[1]]
            else:
                return self.board[self.gap[0]][self.gap[1]]
        tup = tuple(tuple(get_tile((i,j)) for j in range(self.n)) for i in range(self.m))
        return FifteenState(tup, pos, self.vertical_walking_distances, self.horizontal_walking_distances)
    
    def __repr__(self) -> str:
        return '\n\n' + str(self)
    
    # get the direction to push the gap to transform between two adjacent FifteenStates
    def diff(self, other: 'FifteenState') -> str:
        if self.n != other.n or self.m != other.m:
            raise Exception('Cannot compare boards with different dims')
        pos_a = self.gap
        pos_b = other.gap
        if pos_b[0] < pos_a[0]:
            return 'D'
        elif pos_b[0] > pos_a[0]:
            return 'U'
        elif pos_b[1] < pos_a[1]:
            return 'R'
        elif pos_b[1] > pos_a[1]:
            return 'L'
        return ''
    
    # return a boolean describing if this state is possible to solve
    def has_sol(self) -> bool:
        flattened_board = [val for row in self.board for val in row if val != 0]
        inversions = 0
        for i in range(len(flattened_board)):
            for j in range(i):
                if flattened_board[j] > flattened_board[i]:
                    inversions += 1
        if self.n % 2 == 1:
            return inversions % 2 == 0
        
        adj_inversions = inversions + (self.m - self.gap[0] - 1)
        return adj_inversions % 2 == 0
    
    # return a list of neighboring states
    @override
    def get_neighbors(self) -> List[Tuple[State, int]]:
        neighbors: List[Tuple[State, int]] = []
        if self.gap[0] > 0:
            neighbor = self.swap_tile_with_gap((self.gap[0] - 1, self.gap[1]))
            neighbors.append((neighbor, 1))
        if self.gap[0] < len(self.board) - 1:
            neighbor = self.swap_tile_with_gap((self.gap[0] + 1, self.gap[1]))
            neighbors.append((neighbor, 1))  
        if self.gap[1] > 0:
            neighbor = self.swap_tile_with_gap((self.gap[0], self.gap[1] - 1))
            neighbors.append((neighbor, 1))  
        if self.gap[1] < len(self.board[0]) - 1:
            neighbor = self.swap_tile_with_gap((self.gap[0], self.gap[1] + 1))
            neighbors.append((neighbor, 1))            
        return neighbors
    
    # get the MINIMUM number of moves needed to solve the puzzle from this state
    @override
    def h(self, _) -> int:
        m, n = len(self.board), len(self.board[0])
        # if walking distance isn't available then use manhattan distance
        if not self.vertical_walking_distances or not self.horizontal_walking_distances:
            total = 0
            for i in range(m):
                for j in range(n):
                    if self.board[i][j] != 0:
                        val = self.board[i][j] - 1
                        total += abs((val // n) - i) + abs((val % n) - j)
            return total
        # if we DO have walking distance available, use those:
        vert_walking_state: Tuple[Tuple[int,...], ...] = tuple(tuple(sorted(tuple((self.board[i][j] - 1) // n if self.board[i][j] else -1 for j in range(n)))) for i in range(m))
        horiz_walking_state: Tuple[Tuple[int,...], ...] = tuple(tuple(sorted(tuple((self.board[j][i] - 1) % n if self.board[j][i] else -1 for j in range(m)))) for i in range(n))
        return self.vertical_walking_distances[vert_walking_state] + self.horizontal_walking_distances[horiz_walking_state]
    
    @override
    def __str__(self) -> str:
        return '\n'.join(' '.join(str(num) if num > 9 else str(num) + ' ' for num in row) for row in self.board)
    
    @override
    def __hash__(self) -> int:
        return hash(self.board)   
    
    @override
    def __eq__(self, other) -> bool:
        if isinstance(other, FifteenState):
            return self.board == other.board
        return False

# get the coordinates of a "zero" in a 2d list of integers 
def find_zero_of_2d_list(ls):
    for i in range(len(ls)):
        for j in range(len(ls[i])):
            if ls[i][j] == 0:
                return (i,j)
    return None

    
# set random.puz to a random (possible) configuration
def make_random(rows, cols):
    seed(time())
    f = open('random.puz', 'r+')
    f.truncate(0)

    solvable = False
    while not solvable:
        nums = [i for i in range(rows*cols)]
        def next_num():
            num = nums.pop(randint(0, len(nums) - 1))
            return str(num) if num != 0 else '.'
        
        rand_board_str = '\n'.join(' '.join(next_num() for _ in  range(cols)) for _ in range(rows))
        tile_positions = tuple(tuple(int(tile) if tile != '.' else 0 for tile in line.split(' ')) for line in rand_board_str.split('\n'))
        rand_board = FifteenState(tile_positions, find_zero_of_2d_list(tile_positions))
        solvable = rand_board.has_sol()
    f.write(rand_board_str)
    f.close

# calculate a dictionary of walking distances. I wrote this code, but the idea came from Ken'ichiro Takahashi
# https://web.archive.org/web/20141224035932/http://juropollo.xe0.ru:80/stp_wd_translation_en.htm
# https://web.archive.org/web/20210402062612/https://www.ic-net.or.jp/home/takaken/e/15pz/wd.gif
def calc_vertical_walking_distances(m:int, n:int) -> Dict[Tuple[Tuple[int,...],...], int]:
    # a list of tuples containing the element bag and the row of the gap
    starting: Tuple[Tuple[int,...],...] = tuple(tuple(sorted(i if i < m - 1 or j < n - 1 else -1 for j in range(n))) for i in range(m))
    vertical_walking_distances: Dict[Tuple[Tuple[int,...],...], int] = {starting: 0}
    que: List[Tuple[Tuple[Tuple[int,...],...], int]] = [(starting, m - 1)]
    dist = 0
    while que:
        dist += 1
        size = len(que)
        # search the next ply:
        for _ in range(size):
            curr, gap_row = que.pop(0)
            # for each row that we can swap the gap to
            for swap_row in [row for row in (gap_row - 1, gap_row + 1) if 0 <= row < m]:
                # let i be the index of the element in the swap row getting swapped with the gap
                for i in range(n):
                    # let j and k be the indices of and element in the next state
                    def get_ele_of_next_state(j, k):
                        # if the element is the one getting swapped, return the gap:
                        if j == swap_row and k == i:
                            return -1 # we're letting -1 represent the gap
                        # if the element is the gap, return the element getting swapped from the swap row:
                        elif curr[j][k] == -1:
                            return curr[swap_row][i]
                        # otherwise, just return the same element that at position i,j in the current state:
                        else:
                            return curr[j][k]
                    # get the neighbor state for this state/swap_row/index combo:
                    next = tuple(tuple(sorted(tuple(get_ele_of_next_state(i,j) for j in range(n)))) for i in range(m))
                    # if we haven't found this neighbor state yet, then log it and toss it on the queue!
                    if next not in vertical_walking_distances:
                        vertical_walking_distances[next] = dist
                        que.append((next, swap_row))
    return vertical_walking_distances


# get a series of moves to solve a state represented by a 2d list
def solve(ls):
    # ls = [[1,2,3,0],[5,6,7,8],[9,10,11,12],[13,4,14,15]] # the example from Takahashi-san's website. walking distance: 17
    m,n = len(ls), len(ls[0])
    gap = find_zero_of_2d_list(ls)
    # find the gap!
    if not gap:
        print("Your puzzle doesn't have a gap!! Find a valid puzzle and try again")
        return []
    # precompute our walking distance dictionaries:
    vwds = calc_vertical_walking_distances(m, n)
    # only bother computing a table for horizontal walking distance if the puzzle isn't square
    hwds = calc_vertical_walking_distances(n, m) if m != n else vwds
    start = FifteenState(tuple(tuple(row) for row in ls), gap, vertical_walking_distances=vwds, horizontal_walking_distances=hwds)
    if not start.has_sol():
        return None
    pos_to_num = lambda i,j : len(ls[0]) * i + j + 1 if i < m - 1 or j < n - 1 else 0
    end = FifteenState(tuple(tuple(pos_to_num(i,j) for j in range(n)) for i in range(m)), (m - 1, n - 1))
    _, states = a_star(start, end)
    moves = [states[i].diff(states[i + 1]) for i in range(0, len(states) - 1)]
    return moves
