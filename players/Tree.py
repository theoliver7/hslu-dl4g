from players.Node import Node


class Tree:

    def __init__(self) -> None:
        self._rootNode = Node()

    def get_root_node(self) -> Node:
        return self._rootNode
