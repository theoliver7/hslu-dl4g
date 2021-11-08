import numpy as np
from jass.agents.agent import Agent
from jass.game.const import PUSH, color_of_card, offset_of_card, trump_ints, OBE_ABE, UNE_UFE
from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_state_util import observation_from_state, state_from_observation

from mcts.mcts import MonteCarloTreeSearch
from mcts.hand_sampler import HandSampler


class InformationSetMCTSAgent(Agent):
    trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
    # score if the color is not trump
    no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
    # score if obenabe is selected (all colors)
    obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0, ]
    # score if uneufe is selected (all colors)
    uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]

    def __init__(self, iterations=100):
        super().__init__()
        self._rule = RuleSchieber()
        self._iterations = iterations

    def action_trump(self, obs: GameObservation) -> int:
        """
            Determine trump action for the given observation
            Args:
                obs: the game observation, it must be in a state for trump selection

            Returns:
                selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        my_hand = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
        trump_score = 0
        selected_color = -1
        for color in trump_ints:
            trump_tmp = self.__calculate_trump_selection_score(my_hand, color)
            if trump_tmp > trump_score:
                trump_score = trump_tmp
                selected_color = color

        if trump_score <= 68:
            if obs.forehand == -1:
                return PUSH

        return selected_color

    def action_play_card(self, obs: GameObservation) -> int:
        game_sim = self.__create_game_sim_from_obs(obs)

        to_play = MonteCarloTreeSearch().information_set_search(game_sim.state, iterations=self._iterations)

        return to_play

    def __calculate_trump_selection_score(self, cards, trump: int) -> int:
        result = 0
        for card in cards:
            color = color_of_card[card]
            if trump == color:
                result += self.trump_score[offset_of_card[card]]
            elif trump == OBE_ABE:
                result += self.obenabe_score[offset_of_card[card]]
            elif trump == UNE_UFE:
                result += self.uneufe_score[offset_of_card[card]]
            else:
                result += self.no_trump_score[offset_of_card[card]]

        return result

    def __create_game_sim_from_obs(self, game_obs: GameObservation) -> GameSim:
        game_sim = GameSim(rule=self._rule)
        game_sim.init_from_state(state_from_observation(game_obs, HandSampler().sample(game_obs)))
        return game_sim
