# HSLU
#
# Created by Thomas Koller on 7/28/2020
#

import logging

from jass.arena.arena import Arena

from players.determinization_mcts_agent import DeterminizationMCTSAgent
from players.determinization_mcts_cpp_agent import DeterminizationCppMCTSAgent


def main():
    # Set the global logging level (Set to debug or info to see more messages)
    logging.basicConfig(level=logging.WARNING)

    # setup the arena
    arena = Arena(nr_games_to_play=10)

    team0 = DeterminizationCppMCTSAgent(determinizations=100, cutoff_time=0.9)
    team1 = DeterminizationCppMCTSAgent(determinizations=200, cutoff_time=0.35)
    DeterminizationCppMCTSAgent(determinizations=300,
                                cutoff_time=0.18,
                                model_location="/home/localadmin/dl4g/notebooks/models/v7")
    DeterminizationCppMCTSAgent(determinizations=390,
                                cutoff_time=0.1,
                                model_location="/home/localadmin/dl4g/notebooks/models/v7")

    # team0 = InformationSetMCTSAgent(700)
    # team0 = DeterminizationMCTSAgent(threads=1, cutoff_time=0.2, )
    # team0 = DeterminizationCppMCTSAgent(determinizations=10, cutoff_time=0.2, )
    # # team1 = DeterminizationCppMCTSAgent(determinizations=1, cutoff_time=0.2, )
    # team1 = DeterminizationMCTSAgent(threads=10, cutoff_time=0.2, )
    # team1 = AgentRandomSchieber()

    arena.set_players(team0, team1, team0, team1)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()
