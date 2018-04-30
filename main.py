"""
Final Project - Natural Language Processing - S18
AXYZ group
Xinyang Zhu / 2021409044 / XXZ170001 / xyzhu@utdallas.edu / (469)-974-7431
Alvin Zhang / 2021352902 / YXZ165231 / yxz165231@utdallas.edu / (469)-500-2197

This document is the property of Xinyang Zhu & Alvin Zhang
This document may not be reproduced or transmitted in any form,
in whole or in part, without the express written permission of
Xinyang Zhu & Alvin Zhang

main.py
This is the main entrance of this project, contain main loop and UI
"""

import xml.etree.ElementTree as ET      # XML Parser
from corpus import FAQ                # Our corpus model class
from collections import Counter                     # Count the bag of words
from cosine import get_cosine                       # Get the cos similarity
import nltk                                         # NLTK - For downloading only
import numpy as np
from nltk.corpus import stopwords                   # NLTK - Use to identify stop words
from nltk.tokenize import sent_tokenize             # NLTK - Use to tokenize text to sentences
from nltk.tokenize import word_tokenize             # NLTK - Use to tokenize text to words
# from nltk.stem.porter import PorterStemmer        # NLTK - A type of Stemmer
from nltk.stem.lancaster import LancasterStemmer    # NLTK - A type of Stemmer (Selected)
# from nltk.stem import SnowballStemmer             # NLTK - A type of Stemmer
from nltk.stem import WordNetLemmatizer             # NLTK - WordNet Lemmatizer
from nltk.corpus import wordnet as wn               # NLTK - WOrdNet interface
from nltk.parse.stanford import StanfordParser      # Stanford - syntactic tree parser
from sklearn.model_selection import train_test_split
import re
TOKENIZER = r"\d+\.\d+|'\w+|[\w\/\-]+"  # Tokenizer

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

MAX_SHOW = 10                           # Amount of best result will be shown

print("CS6320 Natural Language Processing Final Project by AXZY")

# Read corpus (FAQs)
faq_tree, faqs = None, None
try:
    faq_tree = ET.parse('FAQs.xml')
    faqs = faq_tree.getroot()
# except FileNotFoundError:
#     print("😰 Sorry, cannot find FAQ file, "
#           "please put 'FAQs.xml' file at the same dir as this python file and try again. "
#           "Email to xyzhu@utdallas.edu if necessary.")
#     exit(0)
except ET.ParseError:
    print("😰 Sorry, we find this 'FAQs.xml' cannot be parsed properly. "
          "Please check file integrity or re-download the project. "
          "Email to xyzhu@utdallas.edu if necessary.")
    exit(0)
print("✌️ Parse FAQ file successfully.")

# Establish corpus and extract features
print("🤔️ Processing FAQ data...")
import pandas as pd
corpus = []
trainData = pd.DataFrame()

def data_cleaning(string):
    string = string.lower()
    return string

for faq in faqs:
    current = FAQ(faq[0].text, faq[1].text, len(corpus))
    corpus.append(current)
    # print current.trainData
    trainData = trainData.append(current.trainData, ignore_index = True)
    print(len(corpus), "\tProcessed.")
# faq = faqs[0]
# current = FAQ(faq[0].text, faq[1].text, len(corpus))
# trainData = trainData.append(current.trainData, ignore_index = True)

# trainData.to_csv('text.csv')
valueClass = trainData['value']
trainData = trainData.drop('value', axis=1)
trainData['token'] = trainData['token'].astype(str).astype('category')
trainData['wordTag'] = trainData['wordTag'].astype(str).astype('category')
trainData['stem'] = trainData['stem'].astype(str).astype('category')
trainData['parseTree'] = trainData['parseTree'].astype(str).astype('category')
trainData['lemmas'] = trainData['lemmas'].astype(str).astype('category')
trainData['hypernyms'] = trainData['hypernyms'].astype(str).astype('category')
trainData['hyponyms'] = trainData['hyponyms'].astype(str).astype('category')
trainData['meronyms'] = trainData['meronyms'].astype(str).astype('category')
trainData['holonyms'] = trainData['holonyms'].astype(str).astype('category')

X_tr, X_val, y_tr, y_val = train_test_split(trainData, valueClass, test_size=0.2, random_state=0)
# a = np.asarray(corpusTraining)
# # print corpusTraining
# np.savetxt("test.txt", a, delimiter=",")
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
# clf.fit(trainData, valueClass)

# import lightgbm as lgb
# lgb_train = lgb.Dataset(X_tr, y_tr)
# lgb_val = lgb.Dataset(X_val, y_val)
# params = {
#         'task': 'train',
#         'boosting_type': 'gbdt',
#         'objective': 'regression',
#         'metric': {'l2', 'auc'},
#         'num_leaves': 100,
#         'learning_rate': 0.05,
#         'feature_fraction': 0.9,
#         'bagging_fraction': 0.8,
#         'bagging_freq': 5,
#         'verbose': 0,
#         'min_data': 1
#     }

# lgbm_model = lgb.train(params, train_set = lgb_train, valid_sets = lgb_val, verbose_eval=5)

print("✌️ FAQ data processed.")

# Listen user input
while True:
    u_input = raw_input("Please input your question (Input \"help\" to see manual):\n")

    # input & help & debug
    if u_input == "help":
        print("Manual: ")
    elif u_input == "show raw":
        ET.dump(faqs)
    elif u_input == "show":
        for faq in corpus:
            print(corpus.index(faq))
            # faq.print()
    elif u_input == "show bag":
        for faq in corpus:
            print(corpus.index(faq))
            faq.print_bag()
    elif u_input == "show features":
        for faq in corpus:
            print(corpus.index(faq))
            faq.print_all_features()
    else:
        # Input process

        # Data Cleaning
        input_str = u_input.strip()

        # Tokenize
        input_sents = sent_tokenize(input_str)
        # input_bag = word_tokenize(input_str)
        input_bag = re.findall(TOKENIZER, data_cleaning(input_str))

        # parse bag to counter
        input_counter = Counter(input_bag)

        # Tree Parse (As feature)
        input_tree = parser.raw_parse_sents(input_sents)
        input_parse_tree = []
        for i in input_tree:
            input_parse_tree += i
        parse_tree_str = ''
        for i in input_tree:
            for j in i:
                parse_tree_str += str(j)

        # POS Tagging (As feature)
        testData = pd.DataFrame()
        input_pos = nltk.pos_tag(input_bag)  # Tuple (word, POS)
        tokens = []
        wordTags = []
        values = []
        trees = []
        for tag in input_pos:
            tokens.append(tag[0])
            wordTags.append(tag[1])
            trees.append(parse_tree_str)
        testData['token'] = tokens
        testData['wordTag'] = wordTags
        testData['parseTree'] = trees
        # Remove stop words
        input_sw_removed = [w for w in input_bag if w.lower() not in stop_words]

        # Stem (As feature) & Lemmatize (As feature)
        stems = pd.DataFrame()
        stemWords = []
        input_stemmed = []
        input_lemmatized = []
        for word in input_sw_removed:
            stemWords.append(word)
            input_stemmed.append(stemmer.stem(word))
            input_lemmatized.append(lemmatizer.lemmatize(word))
        stems['token'] = stemWords
        stems['stem'] = input_stemmed
        stems['lemmas'] = input_lemmatized

        # WordNet hypernymns, hyponyms, meronyms, AND holonyms (As feature)
        input_synsets = pd.DataFrame()
        input_synsets_word = []
        input_hypernymns = []
        input_hyponyms = []
        input_meronyms = []
        input_holonyms = []
        input_hypernymns_str = []
        input_hyponyms_str = []
        input_meronyms_str = []
        input_holonyms_str = []
        input_bag_counter = Counter(input_sw_removed)
        for word in input_bag_counter.keys():
            input_synsets_word.append(word)
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
                    cos = get_cosine(Counter(input_bag), Counter(definition))
                    if cos > max_cos:
                        max_cos = cos
                        target_synset = synset
                if target_synset is None:
                    target_synset = synsets[0]
                if target_synset.hypernyms():
                    input_hypernymns += target_synset.hypernyms()
                    for hypernym in target_synset.hypernyms():
                        syn_hypernyms += str(hypernym) + '|'
                    input_hypernymns_str.append(syn_hypernyms)
                else:
                    input_hypernymns_str.append('null')
                if target_synset.hyponyms():
                    input_hyponyms += target_synset.hyponyms()
                    for hyponym in target_synset.hyponyms():
                        syn_hyponyms += str(hyponym) + '|'
                    input_hyponyms_str.append(syn_hyponyms)
                else:
                    input_hyponyms_str.append('null')
                if target_synset.part_meronyms():
                    for meronym in target_synset.part_meronyms():
                        syn_meronyms += str(meronym) + '|'
                    input_meronyms_str.append(syn_meronyms)
                else:
                    input_meronyms_str.append('null')
                if target_synset.part_holonyms():
                    input_holonyms += target_synset.part_holonyms()
                    for holonym in target_synset.part_holonyms():
                        syn_holonyms += str(holonym) + '|'
                    input_holonyms_str.append(syn_holonyms)
                else:
                    input_holonyms_str.append('null')
            else:
                input_hypernymns_str.append('null')
                input_hyponyms_str.append('null')
                input_meronyms_str.append('null')
                input_holonyms_str.append('null')
        input_synsets['token'] = input_synsets_word
        input_synsets['hypernyms'] = input_hypernymns_str
        input_synsets['hyponyms'] = input_hyponyms_str
        input_synsets['meronyms'] = input_meronyms_str
        input_synsets['holonyms'] = input_holonyms_str
        testData = testData.merge(stems, on ='token', how ='left')
        testData = testData.merge(input_synsets, on ='token', how ='left')
        
        testData['token'] = testData['token'].astype(str).astype('category')
        testData['wordTag'] = testData['wordTag'].astype(str).astype('category')
        testData['stem'] = testData['stem'].astype(str).astype('category')
        testData['lemmas'] = testData['lemmas'].astype(str).astype('category')
        testData['parseTree'] = testData['parseTree'].astype(str).astype('category')
        testData['hypernyms'] = testData['hypernyms'].astype(str).astype('category')
        testData['hyponyms'] = testData['hyponyms'].astype(str).astype('category')
        testData['meronyms'] = testData['meronyms'].astype(str).astype('category')
        testData['holonyms'] = testData['holonyms'].astype(str).astype('category')
        # Task 2 - Use statistical method (cosine-similarity) to calculate similarity of  user input and every faqs
        print("\033[7mTask 2 result\033[0m")
        cos_counter = {}
        for faq in corpus:
            cos = get_cosine(input_counter, Counter(faq.bag_q+faq.bag_a))
            cos_counter[corpus.index(faq)] = cos

        show_count = 0
        for index in sorted(cos_counter, key=cos_counter.get, reverse=True):
            if cos_counter[index] > 0 and show_count < MAX_SHOW:
                print("FAQ #", index, "\tSimilarity: ", cos_counter[index])
                corpus[index].print_all()
                print()
                show_count += 1
        # Task 3 - Print the 
        print "Result is: "
        print("\033[7mTask 3 result\033[0m")
        print "Input Token is: "
        print tokens
        print "Input word tags is: "
        print input_pos
        print "Input stem is: "
        print input_stemmed
        print "Input lemmas is: "
        print input_lemmatized
        print "Input parse tree is: "
        print parse_tree_str
        print "Input hypernymns is: "
        print input_hypernymns
        print "Input hyponyms is: "
        print input_hyponyms
        print "Input meronyms is: "
        print input_meronyms
        print "Input holonyms is: "
        print input_holonyms

        # Task 4 - Use statistical method (cosine-similarity) to calculate similarity of  user input and every faqs
        print("\033[7mTask 4 result\033[0m")
        # tokenizer
        tokens_counter_all = {}
        tag_counter_all = {}
        stem_counter_all = {}
        lemmatized_counter_all = {}
        parseTree_counter_all = {}
        hypernymns_counter_all = {}
        hyponyms_counter_all = {}
        meronyms_counter_all = {}
        holonyms_counter_all = {}
        cos_sum_all = {}
        tokens_counter = Counter(input_bag)
        tag_counter = Counter(input_pos)
        stem_counter = Counter(input_stemmed)
        lemmatized_counter = Counter(input_lemmatized)
        # parseTree_counter = Counter(input_parse_tree)
        hypernymns_counter = Counter(input_hypernymns)
        hyponyms_counter = Counter(input_hyponyms)
        meronyms_counter = Counter(input_meronyms)
        holonyms_counter = Counter(input_holonyms)

        for faq in corpus:
            # Tokens
            token_cos = get_cosine(tokens_counter, Counter(faq.bag_q+faq.bag_a))
            tokens_counter_all[corpus.index(faq)] = token_cos
            cos_sum_all[corpus.index(faq)] = token_cos
            # Tags
            tag_cos = get_cosine(tag_counter, Counter(faq.bag_q_pos+faq.bag_a_pos))
            tag_counter_all[corpus.index(faq)] = tag_cos
            cos_sum_all[corpus.index(faq)] += tag_cos
            # Stem
            stem_cos = get_cosine(stem_counter, Counter(faq.bag_q_stemmed+faq.bag_a_stemmed))
            stem_counter_all[corpus.index(faq)] = stem_cos
            cos_sum_all[corpus.index(faq)] += stem_cos
            # Lemmas
            lemmatized_cos = get_cosine(lemmatized_counter, Counter(faq.bag_q_lemmatized+faq.bag_a_lemmatized))
            lemmatized_counter_all[corpus.index(faq)] = lemmatized_cos
            cos_sum_all[corpus.index(faq)] += lemmatized_cos
            # Parse Tree
            # tree_cos = get_cosine(parseTree_counter, Counter(faq.parse_tree_q+faq.parse_tree_a))
            # parseTree_counter_all[corpus.index(faq)] = tree_cos
            # cos_sum_all[corpus.index(faq)] += tree_cos
            # Hypernymns
            hypernymns_cos = get_cosine(hypernymns_counter, Counter(faq.hypernymns))
            hypernymns_counter_all[corpus.index(faq)] = hypernymns_cos
            cos_sum_all[corpus.index(faq)] += hypernymns_cos
            # Hyponyms
            hyponyms_cos = get_cosine(hyponyms_counter, Counter(faq.hyponyms))
            hyponyms_counter_all[corpus.index(faq)] = hyponyms_cos
            cos_sum_all[corpus.index(faq)] += hyponyms_cos
            # Meronyms
            meronyms_cos = get_cosine(meronyms_counter, Counter(faq.meronyms))
            meronyms_counter_all[corpus.index(faq)] = meronyms_cos
            cos_sum_all[corpus.index(faq)] += meronyms_cos
            # Holonyms
            holonyms_cos = get_cosine(holonyms_counter, Counter(faq.holonyms))
            holonyms_counter_all[corpus.index(faq)] = holonyms_cos
            cos_sum_all[corpus.index(faq)] += holonyms_cos

        # print tokens_counter_all
        # print tag_counter_all
        # print stem_counter_all
        # print lemmatized_counter_all
        test2 = pd.DataFrame()
        tokensArray = []
        tagArray = []
        stemArray = []
        lemmatizedArray = []
        treeArray = []
        hypernymnsArray = []
        hyponymsArray = []
        meronymsArray = []
        holonymsArray = []
        totalArray = []
        for i in tokens_counter_all.keys():
            tokensArray.append(tokens_counter_all[i])
            tagArray.append(tag_counter_all[i])
            stemArray.append(stem_counter_all[i])
            lemmatizedArray.append(lemmatized_counter_all[i])
            # treeArray.append(parseTree_counter_all[i])
            hypernymnsArray.append(hypernymns_counter_all[i])
            hyponymsArray.append(hyponyms_counter_all[i])
            meronymsArray.append(meronyms_counter_all[i])
            holonymsArray.append(holonyms_counter_all[i])
            totalArray.append(cos_sum_all[i])
        test2['token'] = tokensArray
        test2['tag'] = tagArray
        test2['stem'] = stemArray
        test2['lemmatized'] = lemmatizedArray
        # test2['parseTree'] = treeArray
        test2['hypernymns'] = hypernymnsArray
        test2['hyponyms'] = hyponymsArray
        test2['meronyms'] = meronymsArray
        test2['holonyms'] = holonymsArray
        test2['sumAll'] = totalArray

        test2.to_csv('test2.csv')
        show_count = 0
        for index in sorted(cos_sum_all, key=cos_sum_all.get, reverse=True):
            if cos_sum_all[index] > 0 and show_count < MAX_SHOW:
                print("FAQ #", index, "\tSimilarity: ", cos_sum_all[index])
                corpus[index].print_all()
                print()
                show_count += 1
        # result = lgbm_model.predict(testData)
        # countR = Counter(result)
        # maxValue = 0 
        # maxIndex = 0
        # print countR
        # for index in countR.keys():
        #     if countR[index] > maxValue:
        #         maxValue = countR[index]
        #         maxIndex = index
        # maxIndex = round(maxIndex, 0)
        # result = corpus[int(maxIndex)]
        # print("FAQ #", maxIndex, "\tSimilarity: ", cos_counter[int(maxIndex)])
        # result.print_all()
    