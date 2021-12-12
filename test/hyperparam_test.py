import pickle

import optuna
import logging
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena
from players.rule_based_agent import RuleBasedAgentPatrik
from players.determinization_mcts_agent import DeterminizationMCTSAgent
from players.determinization_mcts_cpp_agent import DeterminizationCppMCTSAgent
from players.information_set_mcts_agent import InformationSetMCTSAgent

def params(trial):
    return {
        'threads': trial.suggest_int('threads', 1, 500),
        'cutoff_time': trial.suggest_float('cutoff_time', 0.01, 3),
    }


def optimize_agent(trial):
    trial_params = params(trial)

    logging.basicConfig(level=logging.WARNING)

    # setup the arena
    arena = Arena(nr_games_to_play=2)

    team0 = DeterminizationMCTSAgent(threads=trial_params['threads'],cutoff_time=trial_params['cutoff_time'], model_location='../notebooks/models/v7')
    team1 = RuleBasedAgentPatrik()


    arena.set_players(team0, team1, team0, team1)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))


    return -1 * arena.points_team_0.mean()


def main():
    study = optuna.create_study()
    try:
        study.optimize(optimize_agent, n_trials=5, n_jobs=1)
        fileObj = open('study.pkl', 'wb')
        pickle.dump(study, fileObj)
        fileObj.close()
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')

if __name__ == '__main__':
    main()