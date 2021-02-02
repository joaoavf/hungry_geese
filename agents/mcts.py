from random import shuffle, choice
import math
from time import time
from collections import defaultdict
import itertools

RANGE = set(range(11 * 7))


def process_play(geese: list, food: list, moves: tuple):
    for goose, move in zip(geese, moves):

        goose.insert(0, move)
        if move in food:
            food.pop(food.index(move))
        else:
            goose.pop()

    geese_occupied = [item for sublist in geese for item in sublist]

    doubles = []

    for a, b in itertools.combinations(geese, r=2):
        doubles.extend([e for e in a if e in b])

    t = False
    for goose in geese:
        for e in goose:
            if e in doubles:
                t = True
        if t:
            goose.insert(0, -len(goose))
        t = False

    while len(food) < 2:
        food.append(choice(tuple(RANGE.difference(geese_occupied + food))))

    return geese, food


def possible_moves(goose):
    if len(goose) > 0:
        player_head = goose[0]
    else:
        return []

    left = ((player_head % 11 - 1) + 11) % 11 + (player_head // 11) * 11
    right = ((player_head % 11 + 1) + 11) % 11 + (player_head // 11) * 11
    down = ((player_head // 11 + 8) % 7) * 11 + player_head % 11
    up = ((player_head // 11 + 6) % 7) * 11 + player_head % 11
    return [left, right, up, down]


def manager(current_node, max_time):
    t0 = time()

    while time() - t0 < max_time:
        tree_search(current_node)

    scores = [child.score for child in current_node.children]

    return current_node.children[scores.index(max(scores))].play


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
        self.plays = self.initialize_plays()

        [shuffle(p) for p in self.plays]

        self.count = 0
        self.score = 0
        self.value = [0] * len(geese) if max_depth > depth else [len(goose) + depth for goose in geese]

    def initialize_plays(self):
        occupied_geese = [item for sublist in self.geese for item in sublist]
        moves = [possible_moves(goose) for goose in self.geese]
        return [[e for e in move if e not in occupied_geese] for move in moves]

    def explore_or_exploit(self):
        plays = []
        for i, play in enumerate(self.plays):
            if play:
                plays.append(play.pop())
            else:
                value, count = defaultdict(int), defaultdict(int)
                for key in self.children.keys():
                    value[key[i]] += self.children[key].value
                    count[key[i]] += self.children[key].score
                scores = [ucb1(value[key], count[key], self.count) for key in value.keys()]
                plays.append(self.plays[i][scores.index(max(scores))])

        plays = tuple(plays)

        if plays not in self.children.keys:
            geese, food = process_play(self.geese, self.food, plays)
            self.children[plays] = Node(geese=geese, food=food, played=plays, depth=self.depth + 1)

        return self.children[plays]


def tree_search(node):
    if min([value for value in node.value]) > 0:  # Find terminal nodes
        node.score = [node.score[i] + node.value[i] for i, _ in enumerate(node.score)]
        node.count = [node.count[i] + 1 for i, _ in enumerate(node.score)]
        return node.value

    child = node.explore_or_exploit()
    result = tree_search(node=child)

    node.score = [node.score[i] + result[i] for i, _ in enumerate(node.score)]
    node.count = [node.count[i] + 1 for i, _ in enumerate(node.score)]

    return result