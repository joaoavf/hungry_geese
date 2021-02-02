from random import shuffle, choice
import math
from time import time
from collections import defaultdict
from itertools import product


def ucb1(child_score, child_count, parent_count, exploration_parameter=math.sqrt(2)):
    e1 = math.log(parent_count) / child_count
    return (child_score / child_count) + exploration_parameter * math.sqrt(e1)


class Node:
    def __init__(self, geese, food, played, depth=0, max_depth=10):
        self.geese = geese
        self.food = food
        self.played = played
        self.depth = depth
        self.max_depth = max_depth

        self.children = {}
        self.plays = [[]]

        [shuffle(p) for p in self.plays]

        self.count = 0
        self.score = 0
        self.value = [0] * len(geese) if max_depth > depth else [len(goose) + depth for goose in geese]

    def explore_or_exploit(self):
        plays = []
        for i, play in enumerate(self.plays):
            if play:
                plays.append(play.pop())
            else:
                value, count = defaultdict(), defaultdict()
                for key in self.children.keys():
                    value[key[i]] += self.children[key].value
                    count[key[i]] += self.children[key].score
                scores = [ucb1(value[key], count[key], self.count) for key in value.keys()]
                plays.append(self.plays[i][scores.index(max(scores))])

        return tuple(plays)

    def birth_children(self, plays):
        geese, food = process(self.geese, self.food, plays)
        self.children[_tuple] = Node(geese=geese, food=food, played=_tuple, depth=self.depth + 1)
