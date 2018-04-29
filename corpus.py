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
import pandas as pd                                 # Pandas - create a dataset for train
from nltk.corpus import stopwords                   # NLTK - Use to identify stop words
from nltk.tokenize import sent_tokenize             # NLTK - Use to tokenize text to sentences
from nltk.tokenize import word_tokenize             # NLTK - Use to tokenize text to words
# from nltk.stem.porter import PorterStemmer        # NLTK - A type of Stemmer
from nltk.stem.lancaster import LancasterStemmer    # NLTK - A type of Stemmer (Selected)
# from nltk.stem import SnowballStemmer             # NLTK - A type of Stemmer
from nltk.stem import WordNetLemmatizer             # NLTK - WordNet Lemmatizer
from nltk.corpus import wordnet as wn               # NLTK - WOrdNet interface
from nltk.parse.stanford import StanfordParser      # Stanford - syntactic tree parser
import re

# NLTK data downloading
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')

# Tools implement
stop_words = set(stopwords.words('english'))
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
parser = StanfordParser('stanford-parser-full-2018-02-27/stanford-parser.jar',
                        'stanford-english-corenlp-2018-02-27-models.jar')
TOKENIZER = r"\d+\.\d+|'\w+|[\w\/\-]+"  # Tokenizer

def data_cleaning(string):
    string = string.lower()
    return string

# Class of entry of corpus
class FAQ:
    def __init__(self, str_q, str_a, index):
        # Clean the input (delete unnecessary space)
        self.str_q = str_q.strip()
        self.str_a = str_a.strip()
        # Tokenize
        self.bag_q = re.findall(TOKENIZER, data_cleaning(str_q))
        self.bag_a = re.findall(TOKENIZER, data_cleaning(str_a))
        # Tokenize
        # self.bag_q = word_tokenize(self.str_q)
        # self.bag_a = word_tokenize(self.str_a)

        # Remove stop words
        self.bag_q_sw_removed = [w for w in self.bag_q if w.lower() not in stop_words]
        self.bag_a_sw_removed = [w for w in self.bag_a if w.lower() not in stop_words]

        # POS Tagging (As feature)
        self.bag_q_pos = nltk.pos_tag(self.bag_q_sw_removed)     # Tuple (word, POS)
        self.bag_a_pos = nltk.pos_tag(self.bag_a_sw_removed)     # Tuple (word, POS)
        
         # Create dataframe to generate the train data
        self.trainData = pd.DataFrame()
        trainData_q = pd.DataFrame()
        trainData_a = pd.DataFrame()
        
        tokens_q = []
        wordTags_q = []
        values_q = []
        tokens_a = []
        wordTags_a = []
        values_a = []
        for tag in self.bag_q_pos:
            tokens_q.append(tag[0])
            wordTags_q.append(tag[1])
            values_q.append(index)
        trainData_q['token'] = tokens_q
        trainData_q['wordTag'] = wordTags_q
        trainData_q['value'] = values_q
        for tag in self.bag_a_pos:
            tokens_a.append(tag[0])
            wordTags_a.append(tag[1])
            values_a.append(index)
        trainData_a['token'] = tokens_a
        trainData_a['wordTag'] = wordTags_a
        trainData_a['value'] = values_a
        
        # Stem (As feature)
        self.bag_q_stemmed = []
        self.bag_a_stemmed = []
        # Lemmatize (As feature)
        self.bag_q_lemmatized = []
        self.bag_a_lemmatized = []
        stems_q = pd.DataFrame()
        stems_a = pd.DataFrame()
        stemWords_q = []
        stemArray_q = []
        stemWords_a = []
        stemArray_a = []
        lemmatizes_q = []
        lemmatizes_a = []
        for word in self.bag_q_sw_removed:
            self.bag_q_stemmed.append(stemmer.stem(word))
            self.bag_q_lemmatized.append(lemmatizer.lemmatize(word))
            stemWords_q.append(word)
            stemArray_q.append(stemmer.stem(word))
            lemmatizes_q.append(lemmatizer.lemmatize(word))
        for word in self.bag_a_sw_removed:
            self.bag_a_stemmed.append(stemmer.stem(word))
            self.bag_a_lemmatized.append(lemmatizer.lemmatize(word))
            stemWords_a.append(word)
            stemArray_a.append(stemmer.stem(word))
            lemmatizes_a.append(lemmatizer.lemmatize(word))
        stems_q['token'] = stemWords_q
        stems_q['stem'] = stemArray_q
        stems_q['lemmas'] = lemmatizes_q
        stems_a['token'] = stemWords_a
        stems_a['stem'] = stemArray_a
        stems_a['lemmas'] = lemmatizes_a
            
        # Tree Parse (As feature)
        self.sent_q = sent_tokenize(self.str_q)
        self.sent_a = sent_tokenize(self.str_a)
        self.parse_tree_q = parser.raw_parse_sents(self.sent_q)
        self.parse_tree_a = parser.raw_parse_sents(self.sent_a)

        # WordNet hypernymns, hyponyms, meronyms, AND holonyms (As feature)
        synsets_q = pd.DataFrame()
        synsets_a = pd.DataFrame()
        synsets_q_word = []
        synsets_a_word = []
        hypernymns_q = []
        hyponyms_q = []
        meronyms_q = []
        holonyms_q = []
        hypernymns_a = []
        hyponyms_a = []
        meronyms_a = []
        holonyms_a = []
        self.bag_counter = Counter(self.bag_q_sw_removed) + Counter(self.bag_a_sw_removed)
        for word in self.bag_q_sw_removed:
            synsets_q_word.append(word)
            syn_hypernyms = ''
            syn_hyponyms = ''
            syn_meronyms = ''
            syn_holonyms = ''
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
                    for hypernym in target_synset.hypernyms():
                        syn_hypernyms += str(hypernym) + '|'
                    hypernymns_q.append(syn_hypernyms)
                else:
                    hypernymns_q.append('null')
                if target_synset.hyponyms():
                    for hyponym in target_synset.hyponyms():
                        syn_hyponyms += str(hyponym) + '|'
                    hyponyms_q.append(syn_hyponyms)
                else:
                    hyponyms_q.append('null')
                if target_synset.part_meronyms():
                    for meronym in target_synset.part_meronyms():
                        syn_meronyms += str(meronym) + '|'
                    meronyms_q.append(syn_meronyms)
                else:
                    meronyms_q.append('null')
                if target_synset.part_holonyms():
                    for holonym in target_synset.part_holonyms():
                        syn_holonyms += str(holonym) + '|'
                    holonyms_q.append(syn_holonyms)
                else:
                    holonyms_q.append('null')
            else:
                hypernymns_q.append('null')
                hyponyms_q.append('null')
                meronyms_q.append('null')
                holonyms_q.append('null')
        synsets_q['token'] = synsets_q_word
        synsets_q['hypernyms'] = hypernymns_q
        synsets_q['hyponyms'] = hyponyms_q
        synsets_q['meronyms'] = meronyms_q
        synsets_q['holonyms'] = holonyms_q
    
        for word in self.bag_a_sw_removed:
            synsets_a_word.append(word)
            syn_hypernyms = ''
            syn_hyponyms = ''
            syn_meronyms = ''
            syn_holonyms = ''
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
                    for hypernym in target_synset.hypernyms():
                        syn_hypernyms += str(hypernym) + '|'
                    hypernymns_a.append(syn_hypernyms)
                else:
                    hypernymns_a.append('null')
                if target_synset.hyponyms():
                    for hyponym in target_synset.hyponyms():
                        syn_hyponyms += str(hyponym) + '|'
                    hyponyms_a.append(syn_hyponyms)
                else:
                    hyponyms_a.append('null')
                if target_synset.part_meronyms():
                    for meronym in target_synset.part_meronyms():
                        syn_meronyms += str(meronym) + '|'
                    meronyms_a.append(syn_meronyms)
                else:
                    meronyms_a.append('null')
                if target_synset.part_holonyms():
                    for holonym in target_synset.part_holonyms():
                        syn_holonyms += str(holonym) + '|'
                    holonyms_a.append(syn_holonyms)
                else:
                    holonyms_a.append('null')
            else:
                hypernymns_a.append('null')
                hyponyms_a.append('null')
                meronyms_a.append('null')
                holonyms_a.append('null')
        synsets_a['token'] = synsets_a_word
        synsets_a['hypernyms'] = hypernymns_a
        synsets_a['hyponyms'] = hyponyms_a
        synsets_a['meronyms'] = meronyms_a
        synsets_a['holonyms'] = holonyms_a

        self.hypernymns = hypernymns_q + hypernymns_a
        self.hyponyms = hyponyms_q + hyponyms_a
        self.meronyms = meronyms_q + meronyms_a
        self.holonyms = holonyms_q + holonyms_a
        trainData_q = trainData_q.merge(stems_q, on ='token', how ='left')
        trainData_a = trainData_a.merge(stems_a, on ='token', how ='left')
        trainData_q = trainData_q.merge(synsets_q, on ='token', how ='left')
        trainData_a = trainData_a.merge(synsets_a, on ='token', how ='left')
        self.trainData = trainData_q
        # self.trainData = trainData_q.append(trainData_a, ignore_index = True)
        # self.trainData['token'] = self.trainData['token'].astype(str).astype('category')
        # self.trainData['wordTag'] = self.trainData['wordTag'].astype(str).astype('category')
        # self.trainData['stem'] = self.trainData['stem'].astype(str).astype('category')
        # self.trainData['lemmas'] = self.trainData['lemmas'].astype(str).astype('category')
        # self.trainData['hypernyms'] = self.trainData['hypernyms'].astype(str).astype('category')
        # self.trainData['hyponyms'] = self.trainData['hyponyms'].astype(str).astype('category')
        # self.trainData['meronyms'] = self.trainData['meronyms'].astype(str).astype('category')
        # self.trainData['holonyms'] = self.trainData['holonyms'].astype(str).astype('category')

    def print_all(self):
        print("Q:", self.str_q)
        print("A:", self.str_a)

    def print_bag(self):
        print("Q:", self.bag_q)
        print("A:", self.bag_a)

    # def print_counter(self):
    #     print("Q:", self.counter_q)
    #     print("A:", self.counter_a)

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
