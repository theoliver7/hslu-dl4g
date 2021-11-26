import multiprocessing

import jasscpp
import numpy as np
from jass.agents.agent import Agent
from jass.game.const import PUSH, color_of_card, offset_of_card, UNE_UFE, OBE_ABE, trump_ints
from jass.game.game_state_util import state_from_observation
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list

from mcts.hand_sampler import HandSampler
from mcts.mcts import MonteCarloTreeSearch


# Trump selection:  by assigning a value to each card, depending on whether the color is trump or not.
#                   This table is from the Maturawork of Daniel Graf from 2009: "Jassen auf Basis der Spieltheorie".
#
# Play card:        Plays highest value card


class DeterminizationMCTSAgentCpp(Agent):
    trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
    # score if the color is not trump
    no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
    # score if obenabe is selected (all colors)
    obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0, ]
    # score if uneufe is selected (all colors)
    uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]

    def __init__(self, threads=100, cutoff_time=1.0):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = jasscpp.RuleSchieberCpp()
        self._threads = threads
        self._cutoff_time = cutoff_time

    def action_trump(self, obs: jasscpp.GameObservationCpp) -> int:
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

    def action_play_card(self, game_obs: jasscpp.GameObservationCpp) -> int:
        """
        Determine the card to play for this trick with the minmax algorithm

        Args:
            state: the game state where all hands of all players can be seen

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """

        sampler = HandSampler()

        # instantly return if only one card is valid to play
        print(np.count_nonzero(self._rule.get_valid_cards_from_obs(game_obs)))
        if np.count_nonzero(self._rule.get_valid_cards_from_obs(game_obs)) == 1:
            print("instant return")
            return int(np.argmax(self._rule.get_valid_cards_from_obs(game_obs)))

        # Investigate multiprocessing and thread safety of this return_dict
        manager = multiprocessing.Manager()
        mcts_results = manager.dict()
        jobs = []
        for i in range(self._threads):
            p = multiprocessing.Process(target=self.determinization_and_search,
                                        args=(sampler, game_obs, mcts_results, self._cutoff_time))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        print(mcts_results)

        max_card = max(mcts_results, key=mcts_results.get)
        return max_card

    @staticmethod
    def determinization_and_search(sampler, game_obs, mcts_results, cutoff_time):
        hands_sample = sampler.sample(game_obs)
        game_sim = DeterminizationMCTSAgentCpp.__create_game_sim_from_obs(game_obs, hands_sample)
        to_play, simulation_cnt = MonteCarloTreeSearch().search(game_sim.state, 0, cutoff_time)

        if to_play in mcts_results:
            mcts_results[to_play] = simulation_cnt + mcts_results[to_play]
        else:
            mcts_results[to_play] = simulation_cnt

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

    @staticmethod
    def __create_game_sim_from_obs(game_obs: jasscpp.GameObservationCpp, hands: np.array) -> jasscpp.GameSimCpp:
        state = jasscpp.state_from_observation(game_obs, hands)
        game_sim = jasscpp.GameSimCpp(state,jasscpp.RuleSchieberCpp())
        return game_sim