from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.game.const import NORTH, PUSH
from jass.game.game_sim import GameSim
from jass.game.game_util import deal_random_hand
from jass.game.rule_schieber import RuleSchieber
from players.minmax_agent import MinMaxAgent

def main():
    rule = RuleSchieber()
    game = GameSim(rule=rule)
    agent = MinMaxAgent()

    game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)

    # start game with pushing for trump selection
    game.action_trump(PUSH)
    # use agent to select trump
    game.action_trump(agent.action_trump(game.get_observation()))

    while not game.is_done():
        game.action_play_card(agent.action_play_card(game.state))



if __name__ == '__main__':
    main()