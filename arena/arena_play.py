# HSLU
#
# Created by Thomas Koller on 7/28/2020
#

import logging

from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena

from players.determinization_mcts_agent import DeterminizationMCTSAgent
from players.determinization_mcts_cpp_agent import DeterminizationCppMCTSAgent
from players.information_set_mcts_agent import InformationSetMCTSAgent


def main():
    # Set the global logging level (Set to debug or info to see more messages)
    logging.basicConfig(level=logging.WARNING)

    # setup the arena
    arena = Arena(nr_games_to_play=20)

    # team0 = InformationSetMCTSAgent(700)
    team0 = DeterminizationCppMCTSAgent(threads=1, cutoff_time=0.0)
    team1 = DeterminizationCppMCTSAgent(threads=1, cutoff_time=0.0, model_location='../notebooks/models/v7')
    # team1 = AgentRandomSchieber()

    arena.set_players(team0, team1, team0, team1)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()
