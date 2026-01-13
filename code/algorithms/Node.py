'''

Author: Ada Jacyna

Object Node defines current state of a specified branch of the decision tree. It describes:
    --- attribute_pos --- the attribute position currently being considered
    --- split_condition_group --- set of the letters that was obtained after the split
    --- left_node --- further node on left side (meaning based of the split_condition_group letters)
    --- right_node --- further node on right side (meaning base of the negation split_condition_group letters)
'''
class Node:
    def __init__(self, attribute_pos=None, split_condition_group=None, left_node=None, right_node=None, class_value=None):
        self.attribute_pos = attribute_pos
        self.split_condition_group = split_condition_group
        self.left_node = left_node
        self.right_node = right_node
        self.class_value = class_value
