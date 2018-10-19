"""
This code contains functions that convert tree-like conversation into branches
or timeline

"""
from copy import deepcopy


def tree2branches(root):
    node = root
    parent_tracker = []
    parent_tracker.append(root)
    branch = []
    branches = []
    i = 0
    siblings = None
    while True:
        node_name = list(node.keys())[i]
        branch.append(node_name)
        # get children of the node
        # actually all chldren, all tree left under this node
        first_child = list(node.values())[i]
        if first_child != []:  # if node has children
            node = first_child  # walk down
            parent_tracker.append(node)
            siblings = list(first_child.keys())
            i = 0  # index of a current node
        else:
            branches.append(deepcopy(branch))
            if siblings is not None:
                i = siblings.index(node_name)  # index of a current node
                # if node doesnt have next siblings
                while i+1 >= len(siblings):
                    if node is parent_tracker[0]:  # if it is a root node
                        return branches
                    del parent_tracker[-1]
                    del branch[-1]
                    node = parent_tracker[-1]  # walk up ... one step
                    node_name = branch[-1]
                    siblings = list(node.keys())
                    i = siblings.index(node_name)
                i = i+1    # ... walk right
    #            node =  parent_tracker[-1].values()[i]
                del branch[-1]
            else:
                return branches