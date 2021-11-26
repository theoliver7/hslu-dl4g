from jass.game.const import *
from jass.game.game_util import deal_random_hand

import jasscpp

from players.determinization_mcts_cpp_agent import DeterminizationMCTSAgentCpp

rule = jasscpp.RuleSchieberCpp()
game = jasscpp.GameSimCpp()
hands = deal_random_hand()
game.init_from_cards(hands, SOUTH)
game.perform_action_trump(OBE_ABE)
print(game.state)
print(game.state.dealer)
agent = DeterminizationMCTSAgentCpp(threads=4,cutoff_time=1)
while not game.is_done():
    obs = jasscpp.observation_from_state(game.state, -1 )
    print(obs)
    # check both methods to obtain valid cards to test additional functions
    valid_cards = rule.get_valid_cards(obs.hand, obs.tricks[obs.current_trick], obs.nr_cards_in_trick, obs.trump)
    valid_cards2 = game.get_valid_cards()
    assert (valid_cards == valid_cards2).all()
    player = game.state.player
    card = agent.action_play_card(obs)
    print(card)
    game.perform_action_play_card(card)

    # check that card removed from original hands, and hands from the game are the same
    hands[player, card] = 0
    assert (hands == game.state.hands).all()

    # test access to points, which is a vector
    print(game.state.points)