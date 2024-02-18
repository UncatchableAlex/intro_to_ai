from abc import ABC, abstractmethod
from typing import Tuple
import sys

class Node(ABC):
    @abstractmethod
    def get_children(self) -> Tuple['Node',...]: pass

    @abstractmethod
    def eval(self) -> int: pass

    @abstractmethod
    def __str__(self) -> str: pass

# https://en.wikipedia.org/wiki/Negamax#Negamax_with_alpha_beta_pruning
def negamax(node: Node, depth: int, alpha: int, beta: int, color: int) -> int:
    #print(node, depth)
    children = node.get_children()
    if depth == 0 or not children:
        return color * node.eval()
    val = -sys.maxsize
    for child in children:
       val = max(val, -negamax(child, depth - 1, -beta, -alpha, -color))
       alpha = max(alpha, val)
       if alpha >= beta:
           break
    return val 