import multiprocessing

import numpy as np
from jass.agents.agent import Agent
from jass.game.const import PUSH, color_of_card, offset_of_card
from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.rule_schieber import RuleSchieber

from mcts.hand_sampler import HandSampler
from mcts.mcts import MonteCarloTreeSearch


# Trump selection:  by assigning a value to each card, depending on whether the color is trump or not.
#                   This table is from the Maturawork of Daniel Graf from 2009: "Jassen auf Basis der Spieltheorie".
#
# Play card:        Plays highest value card


class DeterminizationMCTSAgent(Agent):
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

    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action for the given observation
        Args:
            obs: the game observation, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        # add your code here using the function above
        push_threshold = 68
        hand_cards = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
        color_trump_values = [0, 0, 0, 0]
        for color in range(4):
            color_trump_values[color] = self.__calculate_trump_selection_score(hand_cards, color)
        max_value = max(color_trump_values)
        best_color = color_trump_values.index(max_value)
        if max_value < push_threshold and obs.player < 1:
            return PUSH
        else:
            return best_color

    def action_play_card(self, game_obs: GameObservation) -> int:
        """
        Determine the card to play for this trick with the minmax algorithm

        Args:
            state: the game state where all hands of all players can be seen

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """

        # make a monte carlo tree search with perfect information to determine the best move

        sampler = HandSampler()

        # Investigate multiprocessing and thread safety of this return_dict
        manager = multiprocessing.Manager()
        mcts_results = manager.dict()
        jobs = []
        for i in range(100):
            p = multiprocessing.Process(target=self.determinization_and_search,
                                        args=(sampler, game_obs, mcts_results))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        print(mcts_results)

        max_card = max(mcts_results, key=mcts_results.get)
        return max_card

    @staticmethod
    def determinization_and_search(sampler, game_obs, mcts_results):
        hands_sample = sampler.sample(game_obs)
        game_sim = DeterminizationMCTSAgent.__create_game_sim_from_obs(game_obs, hands_sample)
        to_play, simulation_cnt = MonteCarloTreeSearch().search(game_sim.state, 0, 1)

        if to_play in mcts_results:
            mcts_results[to_play] = simulation_cnt + mcts_results[to_play]
        else:
            mcts_results[to_play] = simulation_cnt

    def __calculate_trump_selection_score(self, cards, trump: int) -> int:
        # add your code here
        trump_selection_score = 0
        for card in cards:
            if color_of_card[card] == trump:
                trump_selection_score += self.trump_score[offset_of_card[card]]
            else:
                trump_selection_score += self.no_trump_score[offset_of_card[card]]

        return trump_selection_score

    @staticmethod
    def __create_game_sim_from_obs(game_obs: GameObservation, hands: np.array) -> GameSim:
        game_sim = GameSim(rule=RuleSchieber())
        game_sim.init_from_cards(hands, game_obs.dealer)
        game_sim.state.player = game_obs.player
        game_sim.state.trump = game_obs.trump
        game_sim.state.forehand = game_obs.forehand
        game_sim.state.declared_trump = game_obs.declared_trump
        game_sim.state.tricks = game_obs.tricks
        game_sim.state.trick_winner = game_obs.trick_winner
        game_sim.state.trick_points = game_obs.trick_points
        game_sim.state.trick_first_player = game_obs.trick_first_player
        game_sim.state.current_trick = game_obs.current_trick
        game_sim.state.nr_tricks = game_obs.nr_tricks
        game_sim.state.nr_cards_in_trick = game_obs.nr_cards_in_trick
        game_sim.state.nr_played_cards = game_obs.nr_played_cards
        game_sim.state.points = game_obs.points
        return game_sim
