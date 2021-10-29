from jass.agents.agent import Agent
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.game.const import PUSH, color_of_card, offset_of_card, card_ids
from jass.game.game_observation import GameObservation
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.rule_schieber import RuleSchieber

from players.mcts_node import MCTSNode


# Trump selection:  by assigning a value to each card, depending on whether the color is trump or not.
#                   This table is from the Maturawork of Daniel Graf from 2009: "Jassen auf Basis der Spieltheorie".
#
# Play card:        Plays highest value card


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

        # make a monte carlo tree search with perfect information to determine the best move
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

    def __mcts(self, game_state: GameState):
        """
        Determine the card to play after analysis of all hands of the players

        :param game_state: current game state
        :return: Best card to play according to mcts
        """

        tree = MCTSNode(state=game_state)
        # Replace with timer
        for i in range(300):
            # Selection & Expansion
            node_to_simulate = self._tree_policy(tree)
            # Simulation
            score = self.__simulate_play(node_to_simulate, game_state.player)
            # Backpropagation
            self.__back_propagation(node_to_simulate, score)
        node = self.__get_most_simulated_node(tree)
        return card_ids.get(node.get_card())

    def _tree_policy(self, node: MCTSNode):
        """
        :param node: root node from which to find best next best one
        :return: most promising node to explore
        -------
        """
        current_node = node
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child_ubc()
        return current_node

    def __simulate_play(self, node: MCTSNode, simulating_player: int):
        """
        :param node: selected node which simulated
        :param simulating_player: player which started mcts simulation
        :return: outcome of play
        -------
        """
        game_sim = GameSim(rule=RuleSchieber())
        game_sim.init_from_state(node.get_state())

        sim_state = game_sim.state
        random_player = AgentRandomSchieber()
        cards_cnt = sim_state.nr_played_cards
        while cards_cnt < 36:
            game_sim.action_play_card(random_player.action_play_card(game_sim.get_observation()))
            cards_cnt += 1

        if simulating_player % 2 == 0:
            # normalize points
            return game_sim.state.points[0] / sum(game_sim.state.points)
        else:
            # normalize points
            return game_sim.state.points[1] / sum(game_sim.state.points)

    def __back_propagation(self, node: MCTSNode, score: int):
        """
        :param node: node which payoff was calculated
        :param score: the score that was achieved in simulation
        -------
        """
        tmp_node = node
        while tmp_node is not None:
            tmp_node.increment_simulation_cnt()
            tmp_node.set_win_score(score)
            tmp_node = tmp_node.get_parent()

    def __get_most_simulated_node(self, tree: MCTSNode):
        most_simulated = MCTSNode()
        child: MCTSNode
        for child in tree.get_children():
            if child.get_simulation_cnt() > most_simulated.get_simulation_cnt():
                most_simulated = child
        return most_simulated
