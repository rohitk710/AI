import sys
import pandas
import argparse
import math


parser = argparse.ArgumentParser()
parser.add_argument('-f1','--train_file', help='Description', required=True)
parser.add_argument('-f2','--test_file', help='Description', required=True)
parser.add_argument('-o','--output_file', help='Description', required=True)

args = parser.parse_args()
laplace_value = 7

'''Prepare the train data by reading all the lines from file
Break each line into 3 columns delimited by space.
First column is Email Id, second is Label, third is Content'''

# Get train corpus from file into list
train_corpus_info = list()
with open(args.train_file) as train_file:
    for line in train_file:
        line = line.strip()
        train_corpus_info.append(line.split(" ",2))

#Get test corpus from file into list
test_corpus_info = list()
with open(args.test_file) as test_file:
    for line in test_file:
        line = line.strip()
        test_corpus_info.append(line.split(" ",2))

# Form 3 column pandas data frame from the corpus
train_dataset = pandas.DataFrame(train_corpus_info, columns = ['email_id','label_info','content'])
test_dataset = pandas.DataFrame(test_corpus_info, columns = ['email_id','label_info','content'])

# Separate out the ham and spam dataset using label_info column info
ham_dataset = train_dataset.ix[train_dataset['label_info'] == 'ham']
spam_dataset = train_dataset.ix[train_dataset['label_info'] == 'spam']

# Calculate the spam and ham probability of the train dataset
spam_prob = float(spam_dataset.shape[0])/(train_dataset.shape[0])
ham_prob = float(ham_dataset.shape[0])/(train_dataset.shape[0])

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

# Count all the distinct words in the training dataset.
count_distinct_words = len(set(list(spam_words) + list(ham_words)))

# Total number of words in Spam and Ham emails.
total_spam_words = sum(spam_words.values())
total_ham_words = sum(ham_words.values())


'''Calculate the conditional probability of ham words and spam words
Also, calculate the default probability if the word doesn't exist in the training set.'''
conditional_prob_ham ={}
ham_denominator = float((total_ham_words) + count_distinct_words*laplace_value)
for word, frequency in ham_words.iteritems():
    conditional_prob_ham[word] = ((frequency + laplace_value)/ ham_denominator)
default_ham = ((laplace_value)/ ham_denominator)

conditional_prob_spam ={}
spam_denominator = float((total_spam_words) + count_distinct_words*laplace_value)
for word, frequency in spam_words.iteritems():
    conditional_prob_spam[word] = ((frequency + laplace_value)/ spam_denominator)
default_spam = ((laplace_value)/ spam_denominator)

# List to store predictions with the ids and a variable to keep counts of correct predictions
prediction = list()
correct = 0

# List containing only the string of words and their frequency content of every email.
test_email_list = list(test_dataset.content)

# Iterate through all the test emails
for test_email in test_email_list:
    test_email_pairs = {}
    test_email_content = test_email.split(' ')
    test_words = test_email_content[0::2]
    frequency = test_email_content[1::2]

    # Dictionay of each word with its frequency in a test mail
    for itr in range(len(test_words)):
        test_email_pairs.setdefault(test_words[itr], 0)
        test_email_pairs[test_words[itr]] += int(frequency[itr])

    '''Applying the multinomial Naive Baye's formula to caculate P(spam/ham | word) using 
    conditional probability of the word and its frequency, for all the words in the mail.
    We are adding proabilites instead of multiplying them since we have taken log 
    because when we multiply all the probabilites they start approaching zero
    therefore we take log of it and add all of them.'''
    prob_spam = math.log(spam_prob)
    prob_ham = math.log(ham_prob)
    for word in test_words:
        frequency = test_email_pairs.get(word, 0)
        prob_spam += math.log(conditional_prob_spam.get(word, default_spam)) * frequency
        prob_ham += math.log(conditional_prob_ham.get(word, default_ham)) * frequency

    # Classifying prediction as Spam or Ham.
    i =list(test_dataset.content).index(test_email)
    if prob_spam > prob_ham:
        prediction.append([test_dataset['email_id'][i],'spam'])
        if test_dataset['label_info'][i]== 'spam':
            correct += 1
    else:
        prediction.append([test_dataset['email_id'][i],'ham'])
        if test_dataset['label_info'][i] == 'ham':
            correct += 1

accuracy = float(correct) / len(test_dataset.index)
print accuracy
# Writing the predicted output to output_file argument passed.
prediction_data_frame = pandas.DataFrame(prediction)
prediction_data_frame.to_csv(args.output_file, index=False, header=False, sep =' ')
