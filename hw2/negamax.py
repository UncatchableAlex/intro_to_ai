from abc import ABC, abstractmethod
from typing import Tuple
import sys

class Node(ABC):
    """
    An abstract node class with a few basic methods.
    """
    @abstractmethod
    def get_children(self, color_to_play: int) -> Tuple['Node',...]: pass

    @abstractmethod
    def eval(self) -> int: pass

    @abstractmethod
    def __str__(self) -> str: pass


# https://en.wikipedia.org/wiki/Negamax#Negamax_with_alpha_beta_pruning
def negamax_alphabeta(node: Node, depth: int, alpha: int, beta: int, color: int) -> int:
    if depth == 0 or not (children := node.get_children(color)):
        return node.eval()
    val = -sys.maxsize
    for child in children:
       val = max(val, -negamax_alphabeta(child, depth - 1, -beta, -alpha, -color))
       alpha = max(alpha, val)
       if alpha >= beta:
           break
    return val 


# https://en.wikipedia.org/wiki/Negamax#Negamax_base_algorithm
def negamax(node: Node, depth: int, color: int) -> int:
    if depth == 0 or not (children := node.get_children(color)):
        return node.eval()
    val = -sys.maxsize
    for child in children:
       val = max(val, -negamax(child, depth - 1, -color))
    return val 