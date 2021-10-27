import copy

import numpy as np
from jass.agents.agent import Agent
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.game.const import PUSH, color_of_card, offset_of_card, DJ, HJ, SJ, CJ, card_ids
from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list, \
    convert_one_hot_encoded_cards_to_str_encoded_list, count_colors
from jass.game.rule_schieber import RuleSchieber
from players.node import Node

# Trump selection:  by assigning a value to each card, depending on whether the color is trump or not.
#                   This table is from the Maturawork of Daniel Graf from 2009: "Jassen auf Basis der Spieltheorie".
#
# Play card:        Plays highest value card
from players.ucb import UCB


class CheatingMCTSAgent(Agent):
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

    def action_play_card(self, state: GameState) -> int:
        """
        Determine the card to play for this trick with the minmax algorithm

        Args:
            state: the game state where all hands of all players can be seen

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """

        # out of the valid cards find the best one to play when looking into cards of the other players
        # call minmax to find out the best card (child node) to play
        return self.__mcts(state)

    def __calculate_trump_selection_score(self, cards, trump: int) -> int:
        # add your code here
        trump_selection_score = 0
        for card in cards:
            if color_of_card[card] == trump:
                trump_selection_score += self.trump_score[offset_of_card[card]]
            else:
                trump_selection_score += self.no_trump_score[offset_of_card[card]]

        return trump_selection_score

    def __mcts(self, state: GameState):
        """
        Determine the card to play after analysis of all hands of the players

        :param node:
        :return:
        """
        #
        game_sim = GameSim(rule=RuleSchieber())
        game_sim.init_from_state(state)

        sim_state = game_sim.state

        print(game_sim)
        tree = Node(state=sim_state)
        print(sim_state.hands)

        # Replace with timer
        for i in range(100):
            # Selection
            selected_node = self.__select_node(tree)
            if len(selected_node.get_children()) < 8:
                # Expansion
                self.__expand_node(selected_node, game_sim)
            # Simulation
            score = self.__simulate_play(selected_node)
            # Backpropagation
            self.__back_propagation(selected_node, score, sim_state.player)
        node = self.__get_most_simulated_node(tree)
        print(node)
        print(node.get_card())

        return card_ids.get(node.get_card())


    def __select_node(self, tree: Node) -> Node:
        node = tree
        if len(node.get_children()) != 0:
            ucb = UCB()
            node = ucb.find_node_to_explore(node)
        return node

    def __expand_node(self, node: Node, sim: GameSim):
        state = sim.state
        player_hand = self._rule.get_valid_cards(state.hands[state.player],
                                                 state.current_trick,
                                                 state.nr_cards_in_trick,
                                                 state.trump)
        print(player_hand)

        available_cards = convert_one_hot_encoded_cards_to_str_encoded_list(player_hand)
        sim_copy = copy.deepcopy(sim)
        for card in available_cards:
            print("_________________________________________",card_ids.get(card))
            sim_copy = copy.deepcopy(sim)
            sim_copy.action_play_card(card_ids.get(card))
            new_node = Node(node, sim_copy.state, card)
            new_node.set_parent(node)
            node.add_child(new_node)




    def __simulate_play(self, node: Node):
        print("simulating game for card", node)
        game_sim = GameSim(rule=RuleSchieber())
        game_sim.init_from_state(node.get_state())

        sim_state = game_sim.state
        random_player = AgentRandomSchieber()
        cards_cnt = sim_state.nr_played_cards
        while cards_cnt < 36:
            game_sim.action_play_card(random_player.action_play_card(game_sim.get_observation()))
            cards_cnt += 1

        print(game_sim.state.points)
        print(game_sim.state.points[0] / sum(game_sim.state.points))
        # normalize points
        return game_sim.state.points[0] / sum(game_sim.state.points)

    def __back_propagation(self, node: Node, score: int, player: int):
        tmp = node
        while tmp is not None:
            tmp.increment_simulation_cnt()
            if tmp.get_state().player == player:
                tmp.set_win_score(score)
            tmp = tmp.get_parent()

    def __get_most_simulated_node(self, tree: Node):
        most_simulated = Node()
        child: Node
        for child in tree.get_children():
            if child.get_simulation_cnt() > most_simulated.get_simulation_cnt():
                most_simulated = child
        return most_simulated
