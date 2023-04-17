class DTNode:

    def __init__(self, decision):
        self.decision = decision
        self.children = []

    def leaves(self):
        if callable(self.decision):
            return sum([child.leaves() for child in self.children])

        else:
            return 1

    def predict(self, input_x):

        if callable(self.decision):
            i = self.decision(input_x)
            child = self.children[i]
            return child.predict(input_x)

        else:
            return self.decision

n = DTNode(True)
print(n.leaves())