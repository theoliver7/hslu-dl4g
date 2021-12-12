import time

import numpy as np
from jass.game.game_observation import GameObservation


class HandSampler2:

    def sample(self, game_obs: GameObservation, available_cards: np.array):
        # required as multiprocessing messes with random numbers
        np.random.seed()
        # TODO move out of sampler

        hands = np.zeros(shape=[4, 36], dtype=np.int)
        hands[game_obs.player] = game_obs.hand

        player = (game_obs.player - 1) % 4

        while len(np.flatnonzero(available_cards)) > 0:
            if player == game_obs.player:
                player = (player - 1) % 4
                continue
            card = np.random.choice(np.flatnonzero(available_cards))
            available_cards[card] = 0
            hands[player][card] = 1
            player = (player - 1) % 4
        return hands
