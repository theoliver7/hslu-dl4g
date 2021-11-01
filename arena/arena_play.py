# HSLU
#
# Created by Thomas Koller on 7/28/2020
#

import logging

from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena

from players.determinization_mcts_agent import DeterminizationMCTSAgent
from players.minmax_agent import MinMaxAgent
from players.rule_based_agent import RuleBasedAgent, RuleBasedAgentPatrik


def main():
    # Set the global logging level (Set to debug or info to see more messages)
    logging.basicConfig(level=logging.WARNING)

    # setup the arena
    arena = Arena(nr_games_to_play=1, save_filename='arena_games')
    team0 = AgentRandomSchieber()
    team1 = DeterminizationMCTSAgent()

    arena.set_players(team0, team1, team0, team1)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()
