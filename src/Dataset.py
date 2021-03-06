import codecs
from Datapoint import Datapoint
from DatapointSimple import DatapointSimple
from collections import defaultdict
import csv

import sys
#filename  = open('global_warming.csv', "r")

class Dataset(object):

    no_of_categories = 3
    cat_name = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']

    def __init__(self, percentage, test_or_training, simple=False, tweet_file="tweets_random.txt", existance_file="percentage_random.txt", category_file="cat_random.txt", withfilter=True):
        # The datapoints.
        self.trainingfile = test_or_training
        self.point = []
        self.percentage_of_file = percentage
        self.cat_index = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}

        # The number of word occurrences per category.
        self.no_of_words = defaultdict(int)

        # The number of data points per category.
        self.no_of_datapoints = defaultdict(int)

        with codecs.open(tweet_file, 'r', 'utf-8',errors='replace') as tweets:
            with codecs.open(existance_file, 'r', 'utf-8', errors='replace') as existence:
                with codecs.open(category_file, 'r', 'utf-8', errors='replace') as cat:

                    #lines = csv.reader(f, delimiter = "'", quotechar='"')
                    self.totno_of_datapoints = 0

                    tweet_lines = tweets.readlines()

                    number_of_tweets = (len(tweet_lines))
                    lines_to_get = number_of_tweets * self.percentage_of_file
                    lines_to_get = int(lines_to_get)
                    print("Lines read: ", lines_to_get)

                    existence_lines = existence.readlines()
                    cat_lines =cat.readlines()


                    if self.trainingfile:
                        self.totno_of_datapoints +=1
                        tweet = tweet_lines[:lines_to_get]
                        cat = cat_lines[:lines_to_get]
                        percentage = existence_lines[:lines_to_get]

                    else:
                        self.totno_of_datapoints += 1
                        tweet = tweet_lines[lines_to_get:number_of_tweets]
                        cat = cat_lines[lines_to_get:number_of_tweets]
                        percentage = existence_lines[lines_to_get:number_of_tweets]

                    for index, line in enumerate(tweet):
                        #dp  = Datapoint(tweet[index], cat[index],percentage[index], withfilter)
                        if simple:
                            dp = DatapointSimple(text=tweet[index], cat=cat[index], withfilter=withfilter)
                        else:
                            dp = Datapoint(tweet[index], cat[index], withfilter)
                        self.point.append(dp)
                        c_index = self.cat_name.index(dp.cat)
                        self.no_of_words[c_index] += dp.no_of_words
                        self.no_of_datapoints[c_index] += 1

if __name__ == '__main__':
    ds = Dataset()