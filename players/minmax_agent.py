import numpy as np
from jass.agents.agent import Agent
from jass.game.const import PUSH, color_of_card, offset_of_card, DJ, HJ, SJ, CJ
from jass.game.game_observation import GameObservation
from jass.game.game_state import GameState
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list, \
    convert_one_hot_encoded_cards_to_str_encoded_list, count_colors
from jass.game.rule_schieber import RuleSchieber


# Trump selection:  by assigning a value to each card, depending on whether the color is trump or not.
#                   This table is from the Maturawork of Daniel Graf from 2009: "Jassen auf Basis der Spieltheorie".
#
# Play card:        Plays highest value card
class MinMaxAgent(Agent):
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

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play.

        Args:
            obs: the game observation

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        hand_cards = convert_one_hot_encoded_cards_to_int_encoded_list(valid_cards)
        string_cards = convert_one_hot_encoded_cards_to_str_encoded_list(valid_cards)

        card_scores = []
        for card in hand_cards:
            if color_of_card[card] == obs.trump:
                card_scores.append(self.trump_score[offset_of_card[card]])
            else:
                card_scores.append(self.no_trump_score[offset_of_card[card]])
        card = np.argmax(card_scores)
        return hand_cards[card]

    def __calculate_trump_selection_score(self, cards, trump: int) -> int:
        # add your code here
        trump_selection_score = 0
        for card in cards:
            if color_of_card[card] == trump:
                trump_selection_score += self.trump_score[offset_of_card[card]]
            else:
                trump_selection_score += self.no_trump_score[offset_of_card[card]]

        return trump_selection_score