
import random
import numpy as np
import pandas as pd
from graphviz import Digraph
import uuid

target_value = '1.5'

#Returns entropy of a Boolean random variable that is true with probability
def B(q): 
    if q ==1 or q == 0:
        return 0
    else:
        return -(q*np.log2(q) + (1-q)*np.log2(1-q))

#Remainder as shown in the curriculum book
def remainder(target_value, A, examples):
    #Finding p and n
    _, counts = np.unique(examples, return_counts=True)
    p = counts[0]
    n = counts[1]

    #Finding p_k and n_k
    distinct_vals = examples[A].unique()
    temp = []
    for val in distinct_vals:
        temp.append(examples[examples[A] == val])

    sum = 0
    for element in temp:
        try:
            p_k = element[target_value].value_counts()[1]
        except:
            p_k = 0
        try:
            n_k = element[target_value].value_counts()[2]
        except:
            n_k = 0
        sum += ((p_k + n_k)/(p + n)) * B(p_k/(p_k + n_k))
    return sum, p, n

#Measure importance, using the notion of information gain, which is defined in terms of entropy
def importance(target_value, A, examples, randomized=False):
    #If randomized this will return a random value between 0 and 1
    if randomized: 
        return random.uniform(0,1) 
    temp = remainder(target_value, A, examples)

    #If not randomized it will return B(p/p+n) − Remainder(A)
    return B(temp[1]/(temp[1]+temp[2])) - temp[0]

def get_id():
    #return a uniqiue ID
    return str(uuid.uuid4())

#Helper method to find the pluarlity values
def plurality_values(e, tree):
    value = e[target_value].mode()[0]
    id = get_id()
    tree.node(id, label = str(value))
    return [id, str(value)]

#Helper method for ldt function
def same_classification(e: pd.Series, tree: Digraph):
    classificaion = e.unique()[0]
    id = get_id()
    tree.node(id, label = str(classificaion))
    return [id, str(classificaion)]

#Learns the decision tree. This is the main function
def learn_decision_tree(examples, attributes, tree, parent_examples=(), randomized = False):
    #If examples is empty then return plurality-values(parent_examples)
    if examples.empty:
        return plurality_values(parent_examples,tree)
    #Else if all examples have the same classification then return the classification
    elif len(examples[target_value].unique()) == 1:
        return same_classification(examples[target_value], tree)
    #Else if attributes is empty then return plurality-value(examples)
    elif attributes.empty:
        return plurality_values(examples,tree)
    
    #find the most important attribute
    else:
        model = {}
        information_gain = {}
        for col in attributes:
            information_gain[col] = importance(target_value, col, examples, randomized)
        
        #Finds the A with the highest information gain
        max_list = sorted(information_gain.items(), key=lambda x: x[1], reverse=True)
        max_list = list(filter(lambda x: x[0] != '1.5', max_list))
        A = max_list[0][0]
        id = get_id()
        #tree<- a new decision tree with root test A
        tree.node(id, label=A)
        subtree_dict = {}
        for v in examples[A].unique():
            exs = examples[examples[A]==v]
            new_attributes = attributes.copy().drop(A)
            #subtree<-Learn_decision_tree(exs, new attributes-A,examples)
            subtree = learn_decision_tree(exs, new_attributes, tree, examples)
            #Adds a branch to tree with label (A = v) and subtree
            tree.edge(id, subtree[0], label=str(v))
            subtree_dict[v] = subtree[1]
        
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

def tester(test,model):
    y = 0 
    n = 0
    for _, row in test.iterrows():
        prediction = traverse(model, row)

        if prediction == str(row.loc['1.4']):
            y += 1
        else:
            n += 1
    print(f"\nThis model predicted {y} correct and {n} wrong. \nThis equals an accuracy = {round(y/(y+n),3)}")

def main():
    train = pd.read_csv("Assignment4/train.csv")
    test = pd.read_csv("Assignment4/test.csv")

    #initialize the tree
    tree = Digraph(filename="Decision_tree_learning.dot")

    #learn the model
    model = learn_decision_tree(train, train.columns, tree, randomized=True)[1]

    #draw the tree
    tree.render(view=True)
    tester(test, model)
    


if __name__ == "__main__":
    main()