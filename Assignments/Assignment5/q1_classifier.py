import argparse
import sys, os
import pickle as pkl
import pandas
import numpy
import csv
from scipy.stats import chisquare

sys.setrecursionlimit(4000)
node_count=0
'''
Read all the file names and p value from the command line arguments.
'''
parser = argparse.ArgumentParser()
parser.add_argument('-p','--pvalue', help='Description', required=True)
parser.add_argument('-f1','--train_dataset', help='Description', required=True)
parser.add_argument('-f2','--test_dataset', help='Description', required=True)
parser.add_argument('-o','--output_file', help='Description', required=True)
parser.add_argument('-t','--decision_tree', help='Description', required=True)

args = parser.parse_args()

'''
Decision tree will have nodes with some value and 
it will have 5 children as every feature can have only 5 values.
Each leaf node will mean that we have reached to conclusion of True or False and will have it's value as True or False.
Each intermediate node will split the data (i.e branches) into it's 5 children and will have it's value as feature name.
''' 
class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data


    def save_tree(self,filename):
        obj = open(filename,'w')
        pkl.dump(self,obj)

'''
Load all the files into dataframes using pandas
'''
train_dataset = pandas.read_csv(args.train_dataset, header=None, delim_whitespace= True)
train_labels = pandas.read_csv(args.train_dataset.split('.')[0] + '_label.csv', header=None)
test_dataset = pandas.read_csv(args.test_dataset, header=None, delim_whitespace= True)
test_dataset = test_dataset.as_matrix()
pvalue = float(args.pvalue)
#print((train_labels[0] == 0).sum()== (train_labels.shape[0] - 7807))
# print train_labels.shape[0]
# print train_labels.shape[1]

'''Maximizing information gain is equivalent to minimizing average
entropy, because the first term in information gain is constant for all features.
We will iterate through all the features and for every feature we will iterate
through all the unique values for that feature in the training data(i.e. 1 to 5 or less)
For each feature we will find its entropy using the entropy of each of it's unique values and
their probibilty. We will find the feature with minimum entropy and return the index value
of the feature with minimum entropy.'''
def get_best_feature_index(dataset, labels, feature_list):

    min_entropy = sys.maxint
    best_feature_index = 0
    # From the list of current features, find the feature with the minimum entropy value
    for feature in (feature_list):
        # Find total number of examples
        total_count = dataset.shape[0]
        # Unique Values for current feature
        unique_values = dataset[feature].unique()
        entropy = 0
        
        #Iterate through all the unique values
        for value in unique_values:
            count = (dataset[feature] == value).sum()
            probability_feature_value = float(count)/total_count

            # Finding label values (i.e. 0 or 1) for each of the unique feature values
            # by getting their indices from the feature dataframe and count the nos. of 0 and 1
            indices = dataset[dataset[feature] == value].index.values.astype(int)
            labels_subset = labels.ix[indices]
            positive_count = (labels_subset[0] == 1).sum()
            negative_count = (labels_subset[0] == 0).sum()

            positive_probability = float(positive_count)/len(indices)
            negative_probability = float(negative_count)/len(indices)

            if positive_probability !=0:
                positive_term = positive_probability * (numpy.log2(positive_probability))
            else:
                positive_term =0

            if negative_probability !=0:
                negative_term = negative_probability * (numpy.log2(negative_probability))
            else:
                negative_term =0
            
            feature_value_entropy = -(positive_term) - (negative_term)
            entropy += probability_feature_value * float(feature_value_entropy)

        #Find the feature with minimum entropy
        if entropy < min_entropy:
            best_feature_index = feature
            min_entropy = entropy
    return best_feature_index

'''
 chiSquare calculation for the best feature. We make two lists which are needed by the scipy library.
 One list for the observed frequencies and one for the expected frequencies.
 Observed frequencies is basically a list which contains the count of positive(1) values and negative(0) value
 for each of the unique values of the best feature. Whereas, expected frequency is a list of each feature's unique values'
 expected positive and expected negative both of which are calculated by multiplying the positive and negative probability
 of the whole feature with the number of positive and negative occurences for each of the unique values in the feature.
 If the p_value returned by scipy is less that threshold value then we return True meaning we can explore the nodes further.
 '''   

def stopping_criterion(dataset, labels, best_feature_index, threshold_value):

    observed_frequencies = list()
    expected_frequencies = list()

    positive_count_total = (labels[0] == 1).sum()
    negative_count_total = (labels[0] == 0).sum()
    total_count = labels.shape[0]

    positive_probability = float(positive_count_total)/total_count
    negative_probability = float(negative_count_total)/total_count

    unique_values = dataset[best_feature_index].unique()
    for value in unique_values:
        indices = dataset[dataset[best_feature_index] == value].index.values.astype(int)

        positive_count_unique_value =0
        negative_count_unique_value =0

        labels_subset = labels.ix[indices]
        positive_count_unique_value = (labels_subset[0] == 1).sum()
        negative_count_unique_value = (labels_subset[0] == 0).sum()

        expected_positive = float(positive_probability)*len(indices)
        expected_negative = float(negative_probability)*len(indices)

        observed_frequencies.append(positive_count_unique_value)
        observed_frequencies.append(negative_count_unique_value)

        expected_frequencies.append(expected_positive)
        expected_frequencies.append(expected_negative)

    chiSquare, p_value = chisquare(observed_frequencies, expected_frequencies)

    if p_value < threshold_value:
        return True
    else:
        return False

'''
This method basically builds the decision tree. It is called recursively with three termination condition:
1. If all the values are either True (1) or False (0). We return the node with the corresponding value i.e 1 or 0
2. If we have exhausted all the features. We check whether the number of 0's or 1's is more and accordingly return a node.
3. If the chiSquare calculation return False i.e. we cannot split further. We check whether the number of 0's or 1's is more and accordingly return a node.
In each call we find the index of the best feature i.e. least entropy values amongst the remaining features and
then check for chi value and if it allows further exploration of the attribute (returns true) then
form a node with the best feature index value and explore all its five children recursively
'''
def ID3(dataset, feature_list, labels, pvalue):
    global node_count
    # If all the values are 1 then True
    # If all the values are 0 then False
    if(labels[0] == 1).sum() == labels.shape[0]:
        root = TreeNode('T')
        node_count += 1
        return root
    if(labels[0] == 0).sum() == labels.shape[0]:
        root = TreeNode('F')
        node_count +=1
        return root
    
    #If all the features are checked stop splitting.
    if(len(feature_list) ==0):
        return stop_splitting(labels)

    # Choose the best feature with the minimum entropy
    best_feature_index = get_best_feature_index(dataset, labels, feature_list)

    updated_feature_list = list(feature_list)
    updated_feature_list.remove(best_feature_index)

    if stopping_criterion(dataset, labels, best_feature_index, pvalue):
        root = TreeNode(best_feature_index + 1)
        node_count +=1
        for itr in range(5):
            dataset_subset = dataset[dataset[best_feature_index] == itr+1]
            indices = dataset_subset.index.values.astype(int)
            #print indices
            train_labels_subset = labels.ix[indices]
            #print train_labels_subset
            root.nodes[itr] = ID3(dataset_subset, updated_feature_list, train_labels_subset, pvalue)
    else:
        return stop_splitting(labels)
    return root

'''
A helper method which count the nos. of positives (1) and negatives (0) and form a node
with value equal to whichever has greater count. T if 1; F if 0 has maximum.
'''

def stop_splitting(labels):
    global node_count
    positive_count = (labels[0] == 1).sum()
    negative_count = (labels[0] == 0).sum()
    node_count += 1
    if positive_count > negative_count:
        root = TreeNode('T')
        return root
    else:
        root = TreeNode('F')
        return root
    print node_count

print("Training...")

# A list of contain indicex number of all the features
features = list()
for column in train_dataset.columns:
	features.append(column)

root = ID3(train_dataset, features, train_labels, pvalue)

root.save_tree(args.decision_tree)

print("Testing...")

# Evaluate data points in test dataset
def evaluate_datapoint(root, datapoint):
    if root.data == 'T': return 1
    if root.data == 'F': return 0
    return evaluate_datapoint(root.nodes[datapoint[int(root.data)-1]-1], datapoint)

Ypredict = []
for i in range(0,len(test_dataset)):
    Ypredict.append(evaluate_datapoint(root,test_dataset[i]))

with open(args.output_file, "wb") as f:
    writer = csv.writer(f,delimiter=',')
    for y in Ypredict:
        writer.writerow([y])

print("Output files generated")
print node_count