"""
Final Project - Natural Language Processing - S18
AXYZ group
Xinyang Zhu / 2021409044 / XXZ170001 / xyzhu@utdallas.edu / (469)-974-7431
Alvin Zhang / 2021352902 / YXZ165231 / yxz165231@utdallas.edu / (469)-500-2197

This document is the property of Xinyang Zhu & Alvin Zhang
This document may not be reproduced or transmitted in any form,
in whole or in part, without the express written permission of
Xinyang Zhu & Alvin Zhang

corpus.py
This file is a module of corpus, which is FAQ data in our project
"""

from collections import Counter                     # Count the bag of words
from cosine import get_cosine                       # Get the cos similarity
import nltk                                         # NLTK - For downloading only
from nltk.corpus import stopwords                   # NLTK - Use to identify stop words
from nltk.tokenize import sent_tokenize             # NLTK - Use to tokenize text to sentences
from nltk.tokenize import word_tokenize             # NLTK - Use to tokenize text to words
# from nltk.stem.porter import PorterStemmer        # NLTK - A type of Stemmer
from nltk.stem.lancaster import LancasterStemmer    # NLTK - A type of Stemmer (Selected)
# from nltk.stem import SnowballStemmer             # NLTK - A type of Stemmer
from nltk.stem import WordNetLemmatizer             # NLTK - WordNet Lemmatizer
from nltk.corpus import wordnet as wn               # NLTK - WOrdNet interface
from nltk.parse.stanford import StanfordParser      # Stanford - syntactic tree parser

# NLTK data downloading
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')

# Tools implement
stop_words = set(stopwords.words('english'))
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
parser = StanfordParser('/Users/henry/stanfordNLP/stanford-parser-full-2018-02-27/stanford-parser.jar',
                        '/Users/henry/stanfordNLP/stanford-english-corenlp-2018-02-27-models.jar')


# Class of entry of corpus
class FAQ:
    def __init__(self, str_q, str_a):

        # Clean the input (delete unnecessary space)
        self.str_q = str_q.strip()
        self.str_a = str_a.strip()

        # Tokenize
        self.bag_q = word_tokenize(self.str_q)
        self.bag_a = word_tokenize(self.str_a)

        # POS Tagging (As feature)
        self.bag_q_pos = nltk.pos_tag(self.bag_q)     # Tuple (word, POS)
        self.bag_a_pos = nltk.pos_tag(self.bag_a)     # Tuple (word, POS)

        # Remove stop words
        self.bag_q_sw_removed = [w for w in self.bag_q if w.lower() not in stop_words]
        self.bag_a_sw_removed = [w for w in self.bag_a if w.lower() not in stop_words]

        # Stem (As feature)
        self.bag_q_stemmed = []
        self.bag_a_stemmed = []
        for word in self.bag_q:
            self.bag_q_stemmed.append(stemmer.stem(word))
        for word in self.bag_a:
            self.bag_a_stemmed.append(stemmer.stem(word))

        # Lemmatize (As feature)
        self.bag_q_lemmatized = []
        self.bag_a_lemmatized = []
        for word in self.bag_q:
            self.bag_q_lemmatized.append(lemmatizer.lemmatize(word))
        for word in self.bag_a:
            self.bag_a_lemmatized.append(lemmatizer.lemmatize(word))

        # Tree Parse (As feature)
        self.sent_q = sent_tokenize(self.str_q)
        self.sent_a = sent_tokenize(self.str_a)
        self.parse_tree_q = parser.raw_parse_sents(self.sent_q)
        self.parse_tree_a = parser.raw_parse_sents(self.sent_a)

        # WordNet hypernymns, hyponyms, meronyms, AND holonyms (As feature)
        self.hypernymns = []
        self.hyponyms = []
        self.meronyms = []
        self.holonyms = []
        self.bag_counter = Counter(self.bag_q_sw_removed) + Counter(self.bag_a_sw_removed)
        for word in self.bag_counter.keys():
            synsets = wn.synsets(word)
            if synsets:
                max_cos = 0.0
                target_synset = None
                for synset in synsets:
                    definition = synset.definition()
                    cos = get_cosine(Counter(self.bag_q + self.bag_a), Counter(definition))
                    if cos > max_cos:
                        max_cos = cos
                        target_synset = synset
                if target_synset is None:
                    target_synset = synsets[0]
                if target_synset.hypernyms():
                    self.hypernymns += target_synset.hypernyms()
                if target_synset.hyponyms():
                    self.hyponyms += target_synset.hyponyms()
                if target_synset.part_meronyms():
                    self.meronyms += target_synset.part_meronyms()
                if target_synset.part_holonyms():
                    self.holonyms += target_synset.part_holonyms()

    def print(self):
        print("Q:", self.str_q)
        print("A:", self.str_a)

    def print_bag(self):
        print("Q:", self.bag_q)
        print("A:", self.bag_a)

    def print_counter(self):
        print("Q:", self.counter_q)
        print("A:", self.counter_a)

    def concat(self):
        return self.bag_q+self.bag_a

    def print_all_features(self):
        print(self.str_q)
        print(self.str_a)
        print("\tSTOP_WORDS REMOVED")
        print('\t\t', self.bag_q_sw_removed)
        print('\t\t', self.bag_a_sw_removed)
        print("\tSTEMMED")
        print('\t\t', self.bag_q_stemmed)
        print('\t\t', self.bag_a_stemmed)
        print("\tLEMMATIZED")
        print('\t\t', self.bag_q_lemmatized)
        print('\t\t', self.bag_a_lemmatized)
        print("\tPOS")
        print('\t\t', self.bag_q_pos)
        print('\t\t', self.bag_a_pos)
        print("\tPARSE TREE")
        for i in self.parse_tree_q:
            for j in i:
                print(j)
        for i in self.parse_tree_a:
            for j in i:
                print(j)
        print("\tWORD NET")
        print('\t\t', self.hypernymns)
        print('\t\t', self.hyponyms)
        print('\t\t', self.meronyms)
        print('\t\t', self.holonyms)
