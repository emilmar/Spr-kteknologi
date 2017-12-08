from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
import nltk

""" ORIGINAL
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""

"""
This file has been further modified for a project of students in DD1418. 
"""
class DatapointSimple(object):
    """
    This class represents a text as a bag-of-letters + a category.
    """
    def __init__(self, text, cat, percentage=1, withfilter=True):
        # The category.
        self.cat = cat
        self.prep_cat()
        # The text represented as a map from the letter to the number of occurrences.
        self.word = defaultdict(float)

        # Remove undecodeable characters u"\uFFFD"

        unwanted_signs = ['?','.','_','-','!',"'",":",";","|",'""','\"', ')','(', '[',']','&','=','+']

        for letter in text:
        	if letter == u"\uFFFD":
        		pass
        	if withfilter:
        		if letter in unwanted_signs:
        			pass
        	if letter != " ": #if not blankspace
        		print(letter)
        		self.word[token] += 1


        # Number of letters in tweet
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