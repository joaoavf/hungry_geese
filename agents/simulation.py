from random import shuffle, choice
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action
import itertools

MOVE_LIST = [Action.WEST.name, Action.EAST.name, Action.NORTH.name, Action.SOUTH.name]
RANGE = set(range(11 * 7))


def distance(a, b):
    v = abs(a // 11 - b // 11)
    v = min(v, 7 - v)
    h = abs(a % 11 - b % 11)
    h = min(h, 11 - h)
    return v + h


def moves(player_head):
    left = ((player_head % 11 - 1) + 11) % 11 + (player_head // 11) * 11
    right = ((player_head % 11 + 1) + 11) % 11 + (player_head // 11) * 11
    down = ((player_head // 11 + 8) % 7) * 11 + player_head % 11
    up = ((player_head // 11 + 6) % 7) * 11 + player_head % 11
    return [left, right, up, down]


def direction(player, food, valid_moves):
    v = player // 11 - food // 11
    h = player % 11 - food % 11

    if v != 0:
        move = ([-1, 1][(v + 7) % 7 > 3] + 7 + player // 11) % 7 * 11 + (player % 11)
        if move in valid_moves:
            return move

    if h != 0:
        move = ([-1, 1][(h + 11) % 11 > 5] + 11 + player % 11) % 11 + (11 * (player // 11))
        if move in valid_moves:
            return move

    if valid_moves:
        return choice(valid_moves)

    if v != 0:
        move = ([-1, 1][(v + 7) % 7 > 3] + 7 + player // 11) % 7 * 11 + (player % 11)
        return move

    if h != 0:
        move = ([-1, 1][(h + 11) % 11 > 5] + 11 + player % 11) % 11 + (11 * (player // 11))
        return move


def move(geese: list, moves: list, food: list):
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


def simulate_round(geese: list, foods: list) -> tuple:
    geese_occupied = [item for sublist in geese for item in sublist]
    mv = []
    for goose in geese:
        head = goose[0]
        mvs = moves(head)
        valid_moves = [v for v in mvs if v not in geese_occupied]

        ds = [distance(food, head) for food in foods]

        food_selected = ds.index(min(ds))

        mv_choose = direction(head, foods[food_selected], valid_moves)
        mv.append(mv_choose)

    geese, foods = move(geese, mv, foods)

    return geese, foods


def init_state(n_geese=2):
    _mv_list = list(RANGE)[:]
    shuffle(_mv_list)

    geese = [[_mv_list.pop()] for _ in range(n_geese)]

    food = [_mv_list.pop(), _mv_list.pop()]

    return geese, food


def simulation(n_geese=2):
    geese, food = init_state(n_geese=n_geese)

    counter = 0
    while sum([goose[0] >= 0 for goose in geese]):
        geese, food = simulate_round(geese, food)

        counter += 1

        reward = counter

    return geese, food
