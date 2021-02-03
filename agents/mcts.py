from random import shuffle, choice, choices
import math
from time import time
from collections import defaultdict
import itertools
from copy import deepcopy
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Action

RANGE = set(range(11 * 7))
ACTIONS = [Action.WEST.name, Action.EAST.name, Action.NORTH.name, Action.SOUTH.name]


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

    geese = [goose if len(set(goose).intersection(doubles)) == 0 else [] for goose in geese]

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


def manager(current_node, max_time, player_index):
    t0 = time()

    while time() - t0 < max_time:
        tree_search(current_node)

    scores = defaultdict(int)
    count = defaultdict(int)
    for key, child in current_node.children.items():
        scores[key[player_index]] += child.score[player_index]
        count[key[player_index]] += child.count[player_index]

    inverted_dict = {v: k for k, v in scores.items()}

    index = possible_moves(current_node.geese).index(inverted_dict[max(list(scores.values()))])

    return ACTIONS[index]


def ucb1(child_score, child_count, parent_count, exploration_parameter=math.sqrt(2)):
    e1 = math.log(parent_count) / child_count
    return (child_score / child_count) + exploration_parameter * math.sqrt(e1)


class Node:
    def __init__(self, geese, food, played=None, reward=None, depth=0, max_depth=10):
        self.geese = geese
        self.food = food
        self.played = played
        self.depth = depth
        self.max_depth = max_depth

        self.children = {}
        self.plays = self.initialize_plays()

        [shuffle(p) for p in self.plays]

        self.count = [0] * len(geese)
        self.score = [0] * len(geese)

        if reward is None:
            self.value = [0 for _ in self.geese]
        else:
            self.value = reward

    def initialize_plays(self):
        occupied_geese = [item for sublist in self.geese for item in sublist]
        moves = [possible_moves(goose) for goose in self.geese]
        available_moves = [[e for e in move if e not in occupied_geese] for move in moves]
        return [a if len(a) else m for a, m in zip(available_moves, moves)]

    def explore_or_exploit(self):
        plays = []
        keys = list(self.children.keys())
        for i, play in enumerate(self.plays):
            if play:
                plays.append(play.pop())
            else:
                score, count = defaultdict(int), defaultdict(int)
                for key in keys:
                    count[key[i]] += self.children[key].count[i]
                    score[key[i]] += self.children[key].score[i]

                scores = [ucb1(child_score=score[key[i]], child_count=count[key[i]], parent_count=self.count[i]) for key
                          in keys]

                plays.append(choices(keys, weights=scores)[0][i])  # Choices returns a list with 1 element

        plays = tuple(plays)

        if plays not in self.children.keys():
            geese, food = process_play(deepcopy(self.geese), self.food[:], plays)

            reward = [len(g1) + self.depth if self.max_depth == self.depth or len(g2) == 0 else 0 for g1, g2 in
                      zip(self.geese, geese)]

            self.children[plays] = Node(geese=geese,
                                        food=food,
                                        played=plays,
                                        reward=reward,
                                        depth=self.depth + 1,
                                        max_depth=self.max_depth)

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


def agent(obs_dict, config_dict):
    observation = Observation(obs_dict)
    player_index = observation.index
    geese = observation.geese
    food = observation.food

    current_node = Node(geese=geese, food=food)

    return manager(current_node=current_node, max_time=5, player_index=player_index)
