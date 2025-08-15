import json
# import pydantic


class Node:
    def __init__(self, id, text, parent_id=None, representation=None, score=None, consistency=None, diversity=None):
        self.id = id
        self.text = text
        self.parent_id = parent_id
        self.children_ids = []
        self.representation = representation # Last token's hidden state
        self.score = score # Only for leaf nodes, or calculated upwards
        self.consistency = consistency # New attribute for consistency score
        # self.diversity = diversity # New attribute for diversity score

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "representation_shape": self.representation.shape if self.representation is not None else None,
            "score": self.score,
            "consistency": self.consistency,
            # "diversity": self.diversity
        }

# class StepCandidates: 
#     pass
    

class ThoughtTree:
    def __init__(self):
        self.nodes = {}
        self.root_id = None

    def add_node(self, node):
        self.nodes[node.id] = node
        if node.parent_id:
            parent_node = self.nodes.get(node.parent_id)
            if parent_node and node.id not in parent_node.children_ids:
                parent_node.children_ids.append(node.id)
        if self.root_id is None:
            self.root_id = node.id

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def to_json(self):
        nodes_data = {node_id: node.to_dict() for node_id, node in self.nodes.items()}
        return json.dumps(nodes_data, indent=4)
