import numpy as np
from jass.game.game_observation import GameObservation


class HandSampler2:
    def __init__(self,game_obs:GameObservation,):
        self._game_obs = game_obs

        self._available_cards = np.ones(shape=36, dtype=int)
        played_cards = np.copy(game_obs.tricks)
        played_cards = np.reshape(played_cards, 36)
        played_cards = played_cards[played_cards != -1]
        available_cards = np.ma.masked_where(game_obs.hand == 1, self._available_cards).filled(0)

        for played in played_cards:
            available_cards[played] = 0

    def sample(self):
        # required as multiprocessing messes with random numbers
        np.random.seed()

        available_cards = np.copy(self._available_cards)
        hands = np.zeros(shape=[4, 36], dtype=np.int)
        hands[self._game_obs.player] = self._game_obs.hand

        player = (self._game_obs.player - 1) % 4

        while len(np.flatnonzero(available_cards)) > 0:
            if player == self._game_obs.player:
                player = (player - 1) % 4
                continue
            card = np.random.choice(np.flatnonzero(available_cards))
            available_cards[card] = 0
            hands[player][card] = 1
            player = (player - 1) % 4
        return hands
