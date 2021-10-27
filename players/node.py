from jass.game.game_state import GameState


class Node:

    def __init__(self, parent=None, state=None, card=None):
        self._children = []
        self._parent = parent
        self._state = state
        self._card = card

        self._win_score = 0.0
        self._simulation_cnt = 0
        self._player_nr = -1

    def get_parent(self):
        return self._parent

    def set_parent(self, parent):
        self._parent = parent

    def get_children(self) -> []:
        return self._children

    def add_child(self, child):
        self._children.append(child)

    def get_simulation_cnt(self) -> int:
        return self._simulation_cnt

    def increment_simulation_cnt(self):
        self._simulation_cnt += 1

    def get_player_nr(self) -> int:
        return self._player_nr

    def get_state(self) -> GameState:
        return self._state

    def get_win_score(self) -> float:
        return self._win_score

    def set_win_score(self, score: float):
        self._win_score += score

    def get_card(self):
        return self._card
