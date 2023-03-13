class DTNode:
    """
    Node for a decision tree used as both
        1. decision, and
        2. leaf nodes.
    """

    def __init__(self, decision, children=[]):
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

    def leaves(self):
        if len(self.children) == 0:
            return 1

        else:
            return sum([child_node.leaves() for child_node in self.children])

    def predict(self, input_object):
        """
        recursive method
        :param input_object:
            (e.g. a feature vector).
            If it's a leaf node, the input can be anything. It's simply ignored
        :return:
            result of the decision tree for that input.
        """

        if not callable(self.decision):
            return self.decision

        else:
            i = self.decision(input_object)  # function that indicates index of which child to followed
            child_node = self.children[i]  # maps the output of the decision function to a specific child.
            return child_node.predict(input_object)


# The following (leaf) node will always predict True
node = DTNode(True)
x = (1, 2, 3)
print(node.predict(x))

# Sine it's a leaf node, the input can be anything. It's simply ignored.
print(node.predict(None))

yes_node = DTNode("Yes")
no_node = DTNode("No")
tree_root = DTNode(lambda x: 0 if x[2] < 4 else 1)
tree_root.children = [yes_node, no_node]
print(tree_root.predict((False, 'Red', 3.5)))
print(tree_root.predict((False, 'Green', 6.1)))

tt = DTNode(False)
tf = DTNode(True)
ft = DTNode(True)
ff = DTNode(False)
t = DTNode(lambda v: 0 if v[1] else 1)
f = DTNode(lambda v: 0 if v[1] else 1)
t.children = [tt, tf]
f.children = [ft, ff]
n = DTNode(lambda v: 0 if v[0] else 1)
n.children = [t, f]
print(n.predict((True, True)))
print(n.predict((True, False)))
print(n.predict((False, True)))
print(n.predict((False, False)))

n = DTNode(True)
print(n.leaves())

t = DTNode(True)
f = DTNode(False)
n = DTNode(lambda v: 0 if not v else 1)
n.children = [t, f]
print(n.leaves())

tt = DTNode(False)
tf = DTNode(True)
ft = DTNode(True)
ff = DTNode(False)
t = DTNode(lambda v: 0 if v[1] else 1)
f = DTNode(lambda v: 0 if v[1] else 1)
t.children = [tt, tf]
f.children = [ft, ff]
n = DTNode(lambda v: 0 if v[0] else 1)
n.children = [t, f]
print(n.leaves())

tt = DTNode(False)
tf = DTNode(True)
ft = DTNode(True)
ff = DTNode(False)
t = DTNode(lambda v: 0 if v[1] else 1)
f = DTNode(lambda v: 0 if v[1] else 1)
t.children = [tt, tf]
f.children = [ft, ff]
n = DTNode(lambda v: 0 if v[0] else 1)
n.children = [t, f]
z = DTNode(lambda v: 0 if v[0] else 1)
z.children = [n, n, tt]
print(z.leaves())
