import multiprocessing
import os
import queue

from mcts.hand_sampler_2 import HandSampler2
from mcts.mcts_cpp import MonteCarloTreeSearchCpp

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # don't know whats happening

import numpy as np
import jasscpp
from jass.agents.agent import Agent
from jass.game.const import PUSH, color_of_card, offset_of_card, UNE_UFE, OBE_ABE, trump_ints
from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from tensorflow import keras
import tensorflow as tf


# Trump selection:  by assigning a value to each card, depending on whether the color is trump or not.
#                   This table is from the Maturawork of Daniel Graf from 2009: "Jassen auf Basis der Spieltheorie".
#
# Play card:        Plays highest value card


class DeterminizationCppMCTSAgent(Agent):
    trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
    # score if the color is not trump
    no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
    # score if obenabe is selected (all colors)
    obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0, ]
    # score if uneufe is selected (all colors)
    uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]

    def __init__(self, determinizations=100, cutoff_time=1.0, model_location="", iterations=200):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = jasscpp.RuleSchieberCpp()
        self.determinizations = determinizations
        self._cutoff_time = cutoff_time
        self._iterations = iterations

        if os.path.exists(model_location):
            self._model = keras.models.load_model(model_location)
        else:
            self._model = None
            print("invalid file location")

    def action_trump(self, obs: GameObservation) -> int:
        """
            Determine trump action for the given observation
            Args:
                obs: the game observation, it must be in a state for trump selection

            Returns:
                selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        if self._model is not None:
            input = np.append(obs.hand, obs.forehand)
            input = input[:, np.newaxis]
            output = None
            with tf.device('/cpu:0'):
                output = self._model.predict(input.T)
            trump = np.argmax(output)
            if trump == 6:
                if obs.forehand:
                    trump = 10
                else:
                    trump = np.argsort(np.max(output, axis=0))[2]
            return trump
        else:
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

    def action_play_card(self, game_obs: GameObservation) -> int:
        """
        Determine the card to play for this trick with the minmax algorithm

        Args:
            state: the game state where all hands of all players can be seen

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """

        # instantly return if only one card is valid to play

        cpp_obs = DeterminizationCppMCTSAgent.__to_cpp_obs(game_obs)

        valid_cards = self._rule.get_valid_cards_from_obs(cpp_obs)
        if np.count_nonzero(valid_cards) == 1:
            print("instant return")
            return int(np.argmax(valid_cards))

        manager = multiprocessing.Manager()

        hand_sampler = HandSampler2(game_obs)
        samples = multiprocessing.Queue(self.determinizations)

        for i in range(self.determinizations):
            samples.put(hand_sampler.sample())

        mcts_results = manager.dict()
        jobs = []
        for i in range(os.cpu_count()+1):
            p = multiprocessing.Process(target=self.determinization_and_search,
                                        args=(samples, cpp_obs, mcts_results, self._cutoff_time,self._iterations))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        print(mcts_results)
        max_card = max(mcts_results, key=mcts_results.get)
        return max_card

    # This method will be executed in a separate process
    @staticmethod
    def determinization_and_search(samplers: multiprocessing.Queue, game_obs: jasscpp.GameObservationCpp,
                                   mcts_results, cutoff_time,iterations):
        while not samplers.empty():
            try:
                hands_sample = samplers.get_nowait()
            except queue.Empty:
                continue

            game_sim = DeterminizationCppMCTSAgent.__create_game_sim_from_obs(game_obs, hands_sample)
            to_play, simulation_cnt = MonteCarloTreeSearchCpp().search(game_sim.state, iterations=iterations,
                                                                       seconds_limit=cutoff_time)

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
    def __to_cpp_obs(game_obs: GameObservation):
        cpp_obs = jasscpp.GameObservationCpp()
        cpp_obs.dealer = game_obs.dealer
        cpp_obs.player = game_obs.player
        cpp_obs.trump = game_obs.trump
        cpp_obs.declared_trump_player = game_obs.declared_trump
        cpp_obs.forehand = game_obs.forehand
        cpp_obs.hand = game_obs.hand
        cpp_obs.tricks = game_obs.tricks
        cpp_obs.trick_winner = game_obs.trick_winner
        cpp_obs.trick_first_player = game_obs.trick_first_player
        cpp_obs.trick_points = game_obs.trick_points
        cpp_obs.nr_cards_in_trick = game_obs.nr_cards_in_trick
        cpp_obs.nr_played_cards = game_obs.nr_played_cards
        cpp_obs.points = game_obs.points
        # TODO weird not sure if this is the same
        cpp_obs.current_trick = game_obs.nr_tricks
        return cpp_obs

    @staticmethod
    def __create_game_sim_from_obs(game_obs: GameObservation, hands: np.array) -> GameSim:
        game_sim = jasscpp.GameSimCpp()
        state = jasscpp.state_from_observation(game_obs, hands)
        game_sim.state = state
        return game_sim
