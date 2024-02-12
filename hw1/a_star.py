from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
import heapq

class State(ABC):
    @abstractmethod
    def get_neighbors(self) -> List[Tuple['State', int]]: pass

    @abstractmethod
    def h(self, goal_state: 'State') -> int: pass

    @abstractmethod
    def __hash__(self) -> int: pass

    @abstractmethod
    def __eq__(self, other) -> bool: pass

    def __lt__(self, other) -> bool:
        return True
    
    def __gt__(self, other) -> bool:
        return False
    
def get_path(goal_state: State, starting_state: State, closed: Dict[State, State]) -> List[State]:
    path = [goal_state]
    while path[-1] != starting_state:
        path.append(closed[path[-1]])
    path.reverse()
    return path

# A generalized Python implementation of the famous A* algorithm shameless copied from Wikipedia pseudocode:
# https://en.wikipedia.org/wiki/A*_search_algorithm#Pseudocode
def a_star(starting_state: State, goal_state: State) -> Tuple[int, List[State]]:
    open: List[Tuple[int, State]] = [(0, starting_state)]
    came_from: Dict[State, State] = {}
    g_score: Dict[State, int] = {starting_state: 0}
    while open:
        _, current = heapq.heappop(open)
        current_score = g_score[current]
        if current == goal_state:
            return current_score, get_path(goal_state, starting_state, came_from)
        for neighbor, dist in current.get_neighbors():
            tentative_g_score = current_score + dist
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + neighbor.h(goal_state)
                came_from[neighbor] = current
                heapq.heappush(open, (f_score, neighbor))
    # :(
    return -1, []

