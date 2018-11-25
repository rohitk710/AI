import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p','--pvalue', help='Description', required=True)
parser.add_argument('-f1','--train_dataset', help='Description', required=True)
parser.add_argument('-f2','--test_dataset', help='Description', required=True)
parser.add_argument('-o','--output_file', help='Description', required=True)
parser.add_argument('-t','--decision_tree', help='Description', required=True)

args = parser.parse_args()

