import argparse
import sys, os
import pickle as pkl
import pandas
import numpy
import csv
from scipy.stats import chisquare

sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser()
parser.add_argument('-p','--pvalue', help='Description', required=True)
parser.add_argument('-f1','--train_dataset', help='Description', required=True)
parser.add_argument('-f2','--test_dataset', help='Description', required=True)
parser.add_argument('-o','--output_file', help='Description', required=True)
parser.add_argument('-t','--decision_tree', help='Description', required=True)

args = parser.parse_args()

# Decision tree will have nodes with some value and 
# it will have 5 children as every feature can have only 5 values.
# Each leaf node will mean that we have reached to conclusion of True or False and will have it's value as True or False.
# Each intermediate node will split the data (i.e branches) into it's 5 children and will have it's value as feature name.

class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data


    def save_tree(self,filename):
        obj = open(filename,'w')
        pkl.dump(self,obj)

train_dataset = pandas.read_csv(args.train_dataset, header=None, delim_whitespace= True)
train_labels = pandas.read_csv(args.train_dataset.split('.')[0] + '_label.csv', header=None)
test_dataset = pandas.read_csv(args.test_dataset, header=None, delim_whitespace= True)
feature_names = pandas.read_csv('featnames.csv', header=None, delim_whitespace= True)

test_dataset = test_dataset.as_matrix()
#print((train_labels[0] == 0).sum()== (train_labels.shape[0] - 7807))
# print train_labels.shape[0]
# print train_labels.shape[1]

def calculate_entropy(dataset, labels, current_feature_index):
    # Find total number of examples
    total_count = dataset.shape[0]
    # Unique Values for current Attribute
    unique_values = dataset[current_feature_index].unique()
    entropy = 0
    
    for value in unique_values:
        count = (dataset[current_feature_index] == value).sum()
        probability_feature_value = float(count)/total_count

        indices = dataset[dataset[current_feature_index] == value].index.values.astype(int)

        positive_count =0
        negative_count =0
        for i in indices:
            if labels[0][i] == 1:
                positive_count +=1
            else:
                negative_count +=1

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

    return entropy


def get_best_feature_index(dataset, labels, features_list):
    min_entropy = 9999999999
    best_feature_index = 0
    # print feature_names.shape[0]
    # Find the feature with the minimum entropy value
    for i in range(feature_names.shape[0]):
        newEntropy = calculate_entropy(dataset, labels, i)
        if newEntropy < min_entropy:
            best_feature_index = i
            min_entropy = newEntropy

    return best_feature_index

def stopping_criterion(dataset, labels, best_feature_index, pValue):

    observed_frequencies = list()
    expected_frequencies = list()

    positive_count = (labels[0] == 1).sum()
    negative_count = (labels[0] == 0).sum()
    total_count = labels.shape[0]

    positive_probability = float(positive_count)/total_count
    negative_probability = float(negative_count)/total_count

    unique_values = dataset[best_feature_index].unique()
    for value in unique_values:
        indices = dataset[dataset[best_feature_index] == value].index.values.astype(int)

        positive_count =0
        negative_count =0
        for i in indices:
            if labels[0][i] == 1:
                positive_count +=1
            else:
                negative_count +=1

        expected_positive = float(positive_probability)*len(indices)
        expected_negative = float(negative_probability)*len(indices)

        observed_frequencies.append(positive_count)
        observed_frequencies.append(negative_count)

        expected_frequencies.append(expected_positive)
        expected_frequencies.append(expected_negative)

    chiSquare, p_value = chisquare(observed_frequencies, expected_frequencies)

    if p_value <= pValue:
        return True
    else:
        return False


def ID3(dataset, labels, pvalue):
    # If all the values are 1 then True
    # If all the values are 0 then False
    if(labels[0] == 1).sum() == labels.shape[0]:
        root = TreeNode('T')
        return root
    if(labels[0] == 0).sum() == labels.shape[0]:
        root = TreeNode('F')
        return root
    
    #If all the features are checked stop splitting.
    if(feature_names.shape[0] ==0):
        return stop_splitting(labels)

    # Choose the best attribute with the minimum entropy
    best_feature_index = get_best_feature_index(dataset, labels, feature_names)
    #print best_feature_index
    if stopping_criterion(dataset, labels, best_feature_index, pvalue):
        root = TreeNode(best_feature_index+1)

        feature_names.drop(feature_names.index[best_feature_index])
        i =0
        
        for value in range(1,6):
            dataset_subset = dataset[dataset[best_feature_index] == value]
            indices = dataset[dataset[best_feature_index] == value].index.values.astype(int)
            #print indices
            train_labels_subset = labels.ix[indices]
            #print train_labels_subset

            if dataset_subset.empty:
                positive_count = (labels[0] == 1).sum()
                negative_count = (labels[0] == 0).sum()
                if positive_count >= negative_count:
                    root.nodes[i] = TreeNode('T')
                else:
                    root.nodes[i] = TreeNode('F')
            else:
                root.nodes[i] = ID3(dataset_subset, train_labels_subset, pvalue)
            i += 1
    else:
        return stop_splitting(labels)
    return root


def stop_splitting(labels):
    positive_count = (labels[0] == 1).sum()
    negative_count = (labels[0] == 0).sum()
    if positive_count >= negative_count:
        root = TreeNode('T')
        return root
    else:
        root = TreeNode('F')
        return root

pvalue = float(args.pvalue)


print("Training...")

root = ID3(train_dataset, train_labels, pvalue)

root.save_tree(args.decision_tree)

print("Testing...")

# Evaluate data points in test dataset
def evaluate_datapoint(root, datapoint):
    if root.data == 'T': return 1
    if root.data == 'F': return 0
    return evaluate_datapoint(root.nodes[datapoint[int(root.data)-1]-1], datapoint)

Ytree = []
for i in range(0,len(test_dataset)):
    Ytree.append(evaluate_datapoint(root,test_dataset[i]))

with open(args.output_file, "wb") as f:
    writer = csv.writer(f,delimiter=',')
    for y in Ytree:
        writer.writerow([y])

print("Output files generated")