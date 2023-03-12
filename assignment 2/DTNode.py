class DTNode:
    """
    Node for a decision tree used as both
        1. decision, and
        2. leaf nodes.
    """

    def __init__(self, decision, children=None):
        """
        A DTNode object must be initialisable with a decision,
        :param decision:
            1.  either a function that takes an object (typically a feature vector) and
                indicates which child should be followed (when the object is node is a decision node);
            2.  or a value which represents the classification or regression result (when the object is a leaf node).
        :param children:
            set to a data structure that maps the output of the decision function to a specific child.
            We assume the output of the decision function is an index into a list.
        """
        self.decision = decision
        self.children = children

    def predict(self, input_object):
        """
        recursive method
        :param input_object:
            (e.g. a feature vector).
            If it's a leaf node, the input can be anything. It's simply ignored
        :return:
            result of the decision tree for that input.
        """
        return self.decision

# The following (leaf) node will always predict True
node = DTNode(True)

# Prediction for the input (1, 2, 3):
x = (1, 2, 3)
print(node.predict(x))

# Sine it's a leaf node, the input can be anything. It's simply ignored.
print(node.predict(None))