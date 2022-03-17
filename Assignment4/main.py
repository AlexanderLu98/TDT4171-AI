import numpy as np
import pandas as pd
from graphviz import Digraph
import random
import string

def plurality_values(object):
    pass

def learn_decision_tree(examples, attributes, parent_examples=()):
    #psudocode
    if examples.empty:
        return plurality_values(parent_examples)

    #elif all examples have the same classification then return classification
    #elif attributes is empty then return plurality-value(example)
    #else
    
    elif attributes.empty:
        return plurality_values(examples)
    #else blablabla

def main():
    print("test")

if __name__ == "__main__":
    main()