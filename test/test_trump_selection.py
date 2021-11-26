# HSLU
#
# Created by Thomas Koller on 7/28/2020
#

import logging
import numpy as np
from jass.arena.arena import Arena

from players.dnn_trump_agent import DNNBasedTrumpSelectionAgent
from players.rule_trump_agent import RuleBasedTrumpSelectionAgent


def main():
    # Set the global logging level (Set to debug or info to see more messages)
    logging.basicConfig(level=logging.WARNING)
    #Set seed for np random that plays are always the same -> better to compare trump selections
    np.random.seed(77)
    # setup the arena
    arena = Arena(nr_games_to_play=1000)
    team0 = DNNBasedTrumpSelectionAgent()
    team1 = RuleBasedTrumpSelectionAgent()

    arena.set_players(team0, team1, team0, team1)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


if __name__ == '__main__':
    main()
