# HSLU
#
# Created by Thomas Koller on 7/30/2020
#
"""
Example how to use flask to create a service for one or more players
"""
import logging

from jass.service.player_service_app import PlayerServiceApp
from jass.agents.agent_random_schieber import AgentRandomSchieber

from players.determinization_mcts_agent import DeterminizationMCTSAgent
from players.information_set_mcts_agent import InformationSetMCTSAgent
from players.rule_based_agent import RuleBasedAgentPatrik


def create_app():
    """
    This is the factory method for flask. It is automatically detected when flask is run, but we must tell flask
    what python file to use:

        export FLASK_APP=player_service.py
        export FLASK_ENV=development
        flask run --host=0.0.0.0 --port=8888
    """
    logging.basicConfig(level=logging.CRITICAL)

    # create and configure the app
    app = PlayerServiceApp('player_service')

    # you could use a configuration file to load additional variables
    # app.config.from_pyfile('my_player_service.cfg', silent=False)

    # add some players
    app.add_player('random', AgentRandomSchieber())
    app.add_player('patrik_rule', RuleBasedAgentPatrik())

    app.add_player('dmcts0', DeterminizationMCTSAgent(threads=12, cutoff_time=9.5))
    app.add_player('dmcts1', DeterminizationMCTSAgent(threads=8, cutoff_time=5.0))
    app.add_player('dmcts2', DeterminizationMCTSAgent(threads=100, cutoff_time=5.0))
    app.add_player('dmcts3', DeterminizationMCTSAgent(threads=200, cutoff_time=5.0))

    app.add_player('imcts0', InformationSetMCTSAgent(iterations=1000))
    app.add_player('imcts1', InformationSetMCTSAgent(iterations=2000))
    app.add_player('imcts2', InformationSetMCTSAgent(iterations=3000))
    app.add_player('imcts3', InformationSetMCTSAgent(iterations=4000))
    return app


if __name__ == '__main__':
    app = create_app()
    app.run()
