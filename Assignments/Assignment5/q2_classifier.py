import sys
import pandas
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('-f1','--train_file', help='Description', required=True)
parser.add_argument('-f2','--test_file', help='Description', required=True)
parser.add_argument('-o','--output_file', help='Description', required=True)

args = parser.parse_args()

# Prepare the train data by reading all the lines from file
# Break each line into 3 columns delimited by space.
# First column is Email Id, second is Label, third is Content

#Get train corpus from file into list
train_corpus_info = []
with open(args.train_file) as train_file:
    for line in train_file:
        line = line.strip()
        train_corpus_info.append(line.split(" ",2))

#Get test corpus from file into list
test_corpus_info = []
with open(args.test_file) as test_file:
    for line in test_file:
        line = line.strip()
        test_corpus_info.append(line.split(" ",2))

#print train_corpus_info

train_dataset = pandas.DataFrame(train_corpus_info, columns = ['email_id','label_info','content'])
test_dataset = pandas.DataFrame(test_corpus_info, columns = ['email_id','label_info','content'])

# Calculate the spam and ham probability of the train dataset
spam_prob = float(train_dataset.label_info.value_counts()[0])/(train_dataset.shape[0])
ham_prob = float(train_dataset.label_info.value_counts()[1])/(train_dataset.shape[0])

# Separate out the ham and spam dataset using label_info column info
ham_dataset = train_dataset.ix[train_dataset['label_info'] == 'ham']
spam_dataset = train_dataset.ix[train_dataset['label_info'] == 'spam']

# Make the list of ham and spam content
ham_email_list = list(ham_dataset.content)
spam_email_list = list(spam_dataset.content)

# Form the dictionary of ham words with their frequency count
ham_words = {}
for ham_email in ham_email_list:
    ham_content = ham_email.split(' ')
    words = ham_content[0::2]
    frequency = ham_content[1::2]

    for itr in range(len(words)):
        ham_words.setdefault(words[itr], 0)
        ham_words[words[itr]] += int(frequency[itr])

# Form the dictionary of spam words with their frequency count
spam_words = {}
for spam_email in spam_email_list:
    spam_content = spam_email.split(' ')
    words = spam_content[0::2] 
    frequency = spam_content[1::2]

    for itr in range(len(words)):
        spam_words.setdefault(words[itr], 0)
        spam_words[words[itr]] += int(frequency[itr])

# Calculating all the distinct words in the training dataset.
distinct = len(set(list(spam_words) + list(ham_words)))

# Total number of words in Spam and Ham emails.
total_spam_words = sum(spam_words.values())
total_ham_words = sum(ham_words.values())

# List containing only the string of words and their frequency content of every email.
test_email_list = list(test_dataset.content)

prediction = []
for test_email in test_email_list:
    test_email_pairs = {}
    test_email_content = test_email.split(' ')
    test_words = test_email_content[0::2]
    frequency = test_email_content[1::2]

    for itr in range(len(test_words)):
        test_email_pairs.setdefault(test_words[itr], 0)
        test_email_pairs[test_words[itr]] += int(frequency[itr])

    probspam = math.log(spam_prob)
    spam_denominator = float((total_spam_words) + distinct*10)
    for word in test_words:
        for i in range(test_email_pairs[word]):
            probspam += math.log((spam_words.get(word,0) + 10)/ spam_denominator)

    probham = math.log(ham_prob)
    ham_denominator = float((total_ham_words) + distinct*10)
    for word in test_words:
        for i in range(test_email_pairs[word]):
            probham += math.log((ham_words.get(word, 0) + 10)/ham_denominator)

    # Classifying prediction as Spam or Ham.
    if probspam > probham:
        prediction.append('spam')
    else:
        prediction.append('ham')

# Writing the predicted output to 'output.csv' file.
prediction_collect = pandas.Series(prediction)
id = pandas.Series(test_dataset.email_id)
output = pandas.concat([id, prediction_collect], axis=1)
output.to_csv(args.output_file, index=False, header=False)

