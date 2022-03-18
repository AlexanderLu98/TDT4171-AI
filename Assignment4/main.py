
import numpy as np
import pandas as pd
from graphviz import Digraph
import uuid

target_value = '1.5'

def B(q):
    #Returns entropy of a Boolean random variable that is true with probability 
    if q ==1 or q == 0:
        return 0
    else:
        return -(q*np.log2(q) + (1-q)*np.log2(1-q))

def remainder(target_value, A, p, n, examples):

    distinct_vals = examples[A].unique()
    splits = []
    for val in distinct_vals:
        splits.append(examples[examples[A] == val])

    sum = 0
    for split in splits:
        try:
            pk = split[target_value].value_counts()[1]
        except:
            pk = 0
        try:
            nk = split[target_value].value_counts()[2]
        except:
            nk = 0
        sum += ((pk + nk)/(p + n)) * B(pk/(pk + nk))
    return sum

def importance(target_value, A, examples):
    """calculates the expected reduction in entropy from the test on A
    Args:
        tar: target attribute
        A: the attribute we are testing on
        examples: our dataset
        rand (bool, optional): whether or not to use random importance. Defaults to False.
    Returns:
        the expected reduction in entropy
    """
    #p. 1223 in the book
    #the information gain from the attribute test on A is the expected reduction in entropy
    #positive
    p = examples[target_value].value_counts()[1]
    n = examples[target_value].value_counts()[2]
    return B(p/(p+n)) - remainder(target_value, A, p, n, examples)

def get_id():
    #return a uniqiue ID
    return str(uuid.uuid4())

def plurality_values(e, tree):
    #return the most common value
    value = e[target_value].mode()[0]
    id = get_id()
    tree.node(id, label = str(value))
    return [id, str(value)]

def same_classification(e: pd.Series, tree: Digraph):
    #return the classification
    classificaion = e.unique()[0]
    id = get_id()
    tree.node(id, label = str(classificaion))
    return [id, str(classificaion)]

def learn_decision_tree(examples, attributes, tree, parent_examples=()):

    if examples.empty:
        return plurality_values(parent_examples,tree)

    elif len(examples[target_value].unique()) == 1:
        return same_classification(examples[target_value], tree)

    elif attributes.empty:
        return plurality_values(examples,tree)
    
    #find the most important attribute
    else:
        model = {} #store the model
        gain = {} #information gain
        for col in attributes:
            gain[col] = importance(target_value, col, examples)
        
        max_list = sorted(gain.items(), key=lambda x: x[1], reverse=True)
        max_list = list(filter(lambda x: x[0] != '1.5', max_list))
        A = max_list[0][0]
        id = get_id()
        tree.node(id, label=A)
        subtree_dict = {} #used to the subtree under each v
        for v in examples[A].unique():
            exs = examples[examples[A]==v]
            new_attrs = attributes.copy().drop(A)
            subtree = learn_decision_tree(exs, new_attrs, tree, examples)
            tree.edge(id, subtree[0], label=str(v))
            subtree_dict[v] = subtree[1] #add the subtree under v
        
        model[A] = subtree_dict
        return [id, model]

#Helping method for tester, runs recursevily
def traverse(dict, row):
    classes = ['1', '2']
    if dict in classes:
        return dict
    for key in dict:
        next = row.loc[key]
    return traverse(dict[key][next], row)

def tester(test,res):
    correct = 0 
    wrong = 0
    for _, row in test.iterrows():
        prediction = traverse(res, row)

        if prediction == str(row.loc['1.4']):
            correct += 1
        else:
            wrong += 1
    print(f"\nModel predicted {correct} correct and {wrong} wrong. \nAccuracy = {round(correct/(correct+wrong),3)}")

def main():
    train = pd.read_csv("Assignment4/train.csv")
    test = pd.read_csv("Assignment4/test.csv")

    #initialize the tree
    tree = Digraph(filename="Decision_tree_learning.dot")

    #learn the model
    res = learn_decision_tree(train, train.columns, tree)[1]

    #draw the tree
    tree.render(view=True)
    tester(test, res)
    


if __name__ == "__main__":
    main()