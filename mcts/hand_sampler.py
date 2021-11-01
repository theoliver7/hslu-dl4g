import numpy as np
from jass.game.game_observation import GameObservation


class HandSampler:

    def sample(self, game_obs: GameObservation):
        available_cards = np.ones(shape=36, dtype=int)
        available_cards = np.ma.masked_where(game_obs.hand == 1, available_cards).filled(0)
        hands = np.zeros(shape=[4, 36], dtype=np.int)

        for i in range(0, 4):
            if i == game_obs.player:
                hands[i] = game_obs.hand
            else:
                new_hands = self.__sample_one_hand(available_cards)
                hands[i] = np.copy(new_hands)
        return hands

    def __sample_one_hand(self, available_cards: np.array):
        one_hand = np.zeros(shape=36, dtype=int)
        for i in range(0, 9):
            card = np.random.choice(np.flatnonzero(available_cards))
            available_cards[card] = 0
            one_hand[card] = 1

        return one_hand
