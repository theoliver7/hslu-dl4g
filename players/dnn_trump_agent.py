import numpy as np
from jass.agents.agent import Agent
from jass.game.const import PUSH, color_of_card, offset_of_card
from jass.game.game_observation import GameObservation
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list, \
    convert_one_hot_encoded_cards_to_str_encoded_list
from jass.game.rule_schieber import RuleSchieber
from tensorflow import keras

class DNNBasedTrumpSelectionAgent(Agent):
    trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
    # score if the color is not trump
    no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
    # score if obenabe is selected (all colors)
    obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0, ]
    # score if uneufe is selected (all colors)
    uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]

    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()

        self._model = keras.models.load_model('../notebooks/models/v1')


    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action for the given observation
        Args:
            obs: the game observation, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        # add your code here using the function above
        input = np.append(obs.hand,obs.forehand)
        input = input[:,np.newaxis]

        output = self._model.predict(input.T)
        trump = np.argmax(output)
        if trump ==6:
            if obs.forehand:
                trump = 10
            else:
                trump = np.argsort(np.max(output, axis=0))[-2]
        return trump


    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play.

        Args:
            obs: the game observation

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """
        valid_actions = np.flatnonzero(self._rule.get_valid_cards_from_obs(obs))
        action = np.random.choice(valid_actions)
        return action
