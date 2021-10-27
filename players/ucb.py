import sys

import numpy as np

from players.node import Node


class UCB:

    def calculate_ucb(self, totalvisits: int, nodewinscore: float, nodevisit: float) -> float:
        hyperparameter = 1
        if nodevisit == 0:
            return sys.maxsize
        return (nodewinscore / nodevisit) + hyperparameter * np.sqrt(np.log(totalvisits) / nodevisit)

    def find_node_to_explore(self, node: Node):
        parent_visit_count = node.get_simulation_cnt()

        bestscore = 0.0
        bestchildren = []

        child: Node
        for child in node.get_children():
            score = self.calculate_ucb(parent_visit_count, child.get_win_score(),
                                       node.get_simulation_cnt())
            if score > bestscore:
                bestchildren = [child]
                bestscore = score
            elif score == bestscore:
                bestchildren.append(child)

        if len(bestchildren) == 0:
            print("No best child found")

        return np.random.choice(bestchildren)