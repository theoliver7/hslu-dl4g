import multiprocessing
import os

import numpy as np
from jass.agents.agent import Agent
from jass.game.const import PUSH, color_of_card, offset_of_card
from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.game_state_util import state_from_observation
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list, \
    convert_one_hot_encoded_cards_to_str_encoded_list
from jass.game.rule_schieber import RuleSchieber
# from tensorflow import keras

from mcts.hand_sampler import HandSampler
from mcts.mcts import MonteCarloTreeSearch
from players.determinization_mcts_agent import DeterminizationMCTSAgent


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
        # sampler = HandSampler()
        #
        # # instantly return if only one card is valid to play
        # print(np.count_nonzero(self._rule.get_valid_cards_from_obs(game_obs)))
        # if np.count_nonzero(self._rule.get_valid_cards_from_obs(game_obs)) == 1:
        #     print("instant return")
        #     return int(np.argmax(self._rule.get_valid_cards_from_obs(game_obs)))
        # threads = 1
        # cutoff_time = 0.01
        # # Investigate multiprocessing and thread safety of this return_dict
        # manager = multiprocessing.Manager()
        # mcts_results = manager.dict()
        # jobs = []
        # for i in range(threads):
        #     p = multiprocessing.Process(target=self.determinization_and_search,
        #                                 args=(sampler, game_obs, mcts_results, cutoff_time))
        #     jobs.append(p)
        #     p.start()
        #
        # for proc in jobs:
        #     proc.join()
        # print(mcts_results)
        #
        # max_card = max(mcts_results, key=mcts_results.get)
        # return max_card

    @staticmethod
    def determinization_and_search(sampler, game_obs, mcts_results, cutoff_time):
        hands_sample = sampler.sample(game_obs)
        game_sim = DNNBasedTrumpSelectionAgent.__create_game_sim_from_obs(game_obs, hands_sample)
        to_play, simulation_cnt = MonteCarloTreeSearch().search(game_sim.state, 0, cutoff_time)

        if to_play in mcts_results:
            mcts_results[to_play] = simulation_cnt + mcts_results[to_play]
        else:
            mcts_results[to_play] = simulation_cnt

    @staticmethod
    def __create_game_sim_from_obs(game_obs: GameObservation, hands: np.array) -> GameSim:
        game_sim = GameSim(rule=RuleSchieber())
        game_sim.init_from_state(state_from_observation(game_obs, hands))
        return game_sim
