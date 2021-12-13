# HSLU
#
# Created by Thomas Koller on 7/28/2020
#

import logging

from jass.arena.arena import Arena

from players.determinization_mcts_agent import DeterminizationMCTSAgent


def main():
    # Set the global logging level (Set to debug or info to see more messages)
    logging.basicConfig(level=logging.WARNING)
    # Set seed for np random that plays are always the same -> better to compare trump selections
    # np.random.seed(77)
    # setup the arena

    arena = Arena(nr_games_to_play=100)
    # team0 = DNNBasedTrumpSelectionAgent('../notebooks/models/v6')
    # team1 = RuleBasedTrumpSelectionAgent()
    team0 = DeterminizationMCTSAgent(4, 0.2, '../notebooks/models/v4')
    team1 = DeterminizationMCTSAgent(4, 0.2, '../notebooks/models/v4')

    arena.set_players(team0, team1, team0, team1)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()
