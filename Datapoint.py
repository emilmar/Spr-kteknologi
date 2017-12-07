from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
import nltk

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""

class Datapoint(object):
    """
    This class represents a text as a bag-of-words + a category.
    """
    def __init__(self, text, cat, percentage, withfilter):
        # The category.
        self.cat = cat
        self.prep_cat()
        # The text represented as a map from the word to the number of occurrences.
        self.word = defaultdict(float)

        self.beginning = text[0:min(len(text),19)].replace('\n',' ')

        # Remove punctuation (!, ?)
        # Remove most common words
        # Remove undecodeable characters u"\uFFFD"
        tokenizer = RegexpTokenizer('\s+', gaps=True)
        tokens = tokenizer.tokenize(text.lower())

        unwanted_signs = ['?','.','_','-','!',u"\uFFFD","'",":",";","|",'""','\"', ')','(', '[',']','&']
        extra_split_signs = ['=','+']
        common_words = ["for", "to", "is", "of", "has", "will", "the", "rt"]

        for token in tokens:
            if withfilter:
                for sign in unwanted_signs:
                    token = token.replace(sign,"")

                for word in common_words:
                    if word==token:
                        #print(token)
                        pass

                if "http" in token:
                    pass

                for splitsign in extra_split_signs:
                    if splitsign in token:
                        token = token.split(splitsign)
                        for t in token:
                            self.word[t] += 1*percentage

            if token and type(token) is not list: #if not empty string
                self.word[token] =+1*percentage
        self.no_of_words = 0
        for value in self.word.values():
            self.no_of_words += value

    def get_words(self):
        return self.word.keys()

    def prep_cat(self):
        self.cat = self.cat.strip().upper()
        if self.cat in ['NO','N']:
            self.cat = 'NEGATIVE'
        elif self.cat in ['N/A','NA']:
            self.cat = 'NEUTRAL'
        elif self.cat in ['Y','YES']:
            self.cat = 'POSITIVE'