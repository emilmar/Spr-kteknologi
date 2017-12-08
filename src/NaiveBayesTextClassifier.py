
import math
import numpy as np
import argparse
from Dataset import Dataset
import json
import requests
import sys

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""

"""
This class performs text classification using the Naive Bayes method.
"""
accuracy_global = []
class NaiveBayesTextClassifier(object):

    def __init__(self, size, simple=False, withfilter=True):
        """
        Constructor. Read the training file and, possibly, the test
        file. If test file is null, read input from the keyboard.
        """

        # The vocabulary (= the unique words in the training set).
        self.vocabulary = set()

        """
        Model parameters: P(word|cat)
        The index in the arraylist corresponds to the identifier
        of the category (i.e. the first element contains the
        probabilities for category 1, the second for category 2,
        etc.
        """
        self.likelihood = []

        # Prior probabilities P(cat) for all the categories.
        self.prior_prob = {}

        # The resulting classified categories
        self.classified_categories = []
        self.cat_index = {'NEGATIVE': 0, 'NEUTRAL':1, 'POSITIVE':2}
        # The current training set.
        self.training_set = Dataset(size, True, 'data/tweets_random.txt', 'data/percentage_random.txt', 'data/cat_random.txt', simple, withfilter)
        self.test_set = Dataset(1-size, False,'data/tweets_random.txt', 'data/percentage_random.txt', 'data/cat_random.txt', simple, withfilter)
        self.build_model()
        self.classify_testset(self.test_set)


    # ---------------------------------------------------------- #


    def classify_datapoint(self, d):
        """
        Computes the posterior probability P(cat|d) = P(cat|w1 ... wn) =
        = P(cat) * P(w1|cat) * ... *vP(wn|cat), for all categories cat.

        :param d: The datapoint to be classified.
        :return: The name of the winning category (i.e. argmax P(cat|d) ).
        """

        #  REPLACE THE STATEMENT BELOW WITH YOUR CODE
        result = []

        # the category
        for category_index in range(len(self.training_set.no_of_datapoints.keys())):
            local_result = 0

            # enbart kategorin
            local_result += self.prior_prob[category_index]

            # gå igenom alla ord i datapunkten d
            for token in d.word.keys():
                if token in self.vocabulary:
                    local_result += self.likelihood[category_index][token]

            result.append(local_result)

        # hitta den mest troliga kategorin
        most_prob = 0
        most_index = None

        for i in range(len(result)):
            if most_index is None or result[i] > most_prob:
                most_prob = result[i]
                most_index = i

        return self.training_set.cat_name[most_index]

    # ---------------------------------------------------------- #


    def build_model(self):
        """
        Computes the prior probabilities P(cat) and likelihoods P(word|cat),
        for all words and categories. To avoid underflow, log-probabilities
        are used.
        """

        # Prior probabilities P(cat)
        for category_index in range(len(self.training_set.no_of_datapoints.keys())):
            log_prob = math.log(self.training_set.no_of_datapoints[category_index] / self.training_set.totno_of_datapoints)
            self.prior_prob[category_index] = log_prob

        # populera ordförrådet
        for point in self.training_set.point:
            for token in point.word.keys():
                self.vocabulary.add(token)

        # P(word|cat)
        for category_index in range(len(self.training_set.no_of_datapoints.keys())):
            category_name = self.training_set.cat_name[category_index]

            word_occurrences = {}
            words_in_category = self.training_set.no_of_words[category_index]

            for token in self.vocabulary:
                occurrences = 1

                # gå igenom alla dokument (datapoints) i kategorin
                # kolla hur många gånger ordet word förekommer, addera till occurrences
                for point in self.training_set.point:
                    if point.cat == category_name:
                        occurrences += point.word[token]

                word_occurrences[token] = math.log(occurrences / (words_in_category + len(self.vocabulary)))

            self.likelihood.append(word_occurrences)

    # ---------------------------------------------------------- #



    def classify_testset(self, testset):

        for data_point in testset.point:
            self.classified_categories.append(self.classify_datapoint(data_point))

        # --------------------- TASK 2 ----------------------------- #

        # Time to compute accuracy, precision and recall
        # Accuracy: TP + TN / TP + TN + FP + FN (correct / all)
        # Precision = TP / TP + FP
        # Recall = TP / TP + FN

        # Matrix with 3 categories and TP, TN, FP, FN in that order

        results = []
        for category in range(self.training_set.no_of_categories):
            results.append([])
            for i in range(4):
              results[category].append(0)

        for idx, data_point in enumerate(testset.point):    # For each datapoint
            true_ans = self.training_set.cat_index[data_point.cat]
            ans = self.training_set.cat_index[self.classified_categories[idx]]
            if ans == true_ans:
              results[ans][0] += 1    # TP
              results[(ans + 1) % 3][1] += 1  # TN
              results[(ans + 2) % 3][1] += 1  # TN
            else:   # If classified wrong
              s = set([0, 1, 2]).difference(set([ans, true_ans]))
              results[true_ans][3] += 1   # False negative for this class
              results[ans][2] += 1        # False positive for this class
              results[s.pop()][1] += 1          # True negative for this class

            # For each category, print precision, recall and accuracy
        for category in range(self.training_set.no_of_categories):
            accuracy = (results[category][0] + results[category][1])/float(sum(results[category]))
            precision = results[category][0]/ (float(results[category][0] + results[category][2]))
            recall = results[category][0] / float((results[category][0] + results[category][3]))
            print("For category '%s' Accuracy: %s, Precision: %s, Recall: %s" % (self.training_set.cat_name[category], str(round(accuracy,4)), str(round(precision, 4)), str(round(recall, 4))))
        accuracy = (results[0][0] + results[1][0] + results[2][0] + results[0][1] + results[1][1] + results[2][1])
        accuracy = accuracy / float(sum(results[0])+sum(results[1]) + sum(results[2]))
        print("The Accuracy for the entire testset is %s " % str(accuracy))
        accuracy_global.append(accuracy)
        # Print the most 50 most impactful words in each category
        #print("Class %s" % self.training_set.cat_name[0])
        #res = sorted(self.likelihood[0].items(), key=lambda x: x[1], reverse=True)
        #print(res)
        #Kappa statistisk


        # ---------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description='NaiveBayesTextClassifier')
    parser.add_argument('--all', '-a', action="store_true", default=False, help='Test all files (default False)')
    parser.add_argument('--single', '-s', action="store_true", default = False, help='Test one file')
    parser.add_argument('--simple', '-si', action="store_false", default=True, help='Test with simple model i.e one letter at a time (default True)')
    parser.add_argument('--partition', '-p', type=float, default=0.5, help="Partition size for training set (default 0.5)")
    parser.add_argument('--nofilter', '-nf', action="store_true", default=False, help='Without filter')
    
    arguments = parser.parse_args()

    if arguments.all:
        for i in np.arange(0.5,1.0,0.05):
            nbtc = NaiveBayesTextClassifier(size=i, simple=arguments.simple, withfilter=arguments.nofilter)
        for i in np.arange(0.5,1.0,0.05):
            nbtc = NaiveBayesTextClassifier(size=i, simple=arguments.simple, withfilter=arguments.nofilter)
        #for x in nbtc.classified_categories: print(x)
        for row in accuracy_global:
            print(row)

    elif arguments.single:
        nbtc = NaiveBayesTextClassifier(size=arguments.partition, simple=arguments.simple, withfilter=arguments.nofilter)

if __name__ == "__main__":
    main()
