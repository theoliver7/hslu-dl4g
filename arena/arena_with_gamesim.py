import numpy as np
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.game.const import NORTH
from jass.game.game_sim import GameSim
from jass.game.game_util import deal_random_hand
from jass.game.rule_schieber import RuleSchieber

from players.cheating_mcts_agent import CheatingMCTSAgent
from players.determinization_mcts_agent import DeterminizationMCTSAgent

def main():
    nr_games_to_play = 3
    points_team_0 = np.zeros(nr_games_to_play)
    points_team_1 = np.zeros(nr_games_to_play)

    for i in range(nr_games_to_play):
        rule = RuleSchieber()
        game = GameSim(rule=rule)
        agent = CheatingMCTSAgent()
        opponent = DeterminizationMCTSAgent()
        game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)

        # Select random Trump
        game.action_trump(opponent.action_trump(game.get_observation()))

        while not game.is_done():
            if game.state.player % 2 == 0:
                game.action_play_card(agent.action_play_card(game.state))
            else:
                game.action_play_card(opponent.action_play_card(game.get_observation()))
        print("GAMESIM ", i, " WITH POINTS:", game.state.points)
        points_team_0[i] = game.state.points[0]
        points_team_1[i] = game.state.points[1]

    print(points_team_0.mean())
    print(points_team_1.mean())


if __name__ == '__main__':
    main()
