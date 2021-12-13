import os

import numpy as np
from jass.agents.agent import Agent
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from tensorflow import keras


class DNNBasedTrumpSelectionAgent(Agent):

    def __init__(self,model_location=""):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()
        if os.path.exists(model_location):
            self._model = keras.models.load_model(model_location)
        else:
            print("invalid file location")

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

    def action_play_card(self, game_obs: GameObservation) -> int:
        """
        Determine the card to play for this trick with the minmax algorithm

        Args:
            state: the game state where all hands of all players can be seen

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """
        valid_actions = np.flatnonzero(self._rule.get_valid_cards_from_obs(game_obs))
        return np.random.choice(valid_actions)
