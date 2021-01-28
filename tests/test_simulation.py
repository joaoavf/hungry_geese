from agents.simulation import moves, direction


def test_moves():
    assert moves(player_head=0) == [10, 1, 66, 11]
    assert moves(player_head=37) == [36, 38, 26, 48]
    assert moves(player_head=76) == [75, 66, 65, 10]


def test_direction():
    assert direction(player=0, food=1, valid_moves=[10, 1, 66, 11]) == 1
    assert direction(player=0, food=1, valid_moves=[10, 66, 11]) in [10, 66, 11]
    assert direction(player=0, food=2, valid_moves=[10, 1, 66, 11]) == 1
    assert direction(player=0, food=8, valid_moves=[10, 1, 66, 11]) == 10
