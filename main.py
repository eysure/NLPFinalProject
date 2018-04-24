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

MAX_SHOW = 10                           # Amount of best result will be shown

print("CS6320 Natural Language Processing Final Project by AXZY")

# Read corpus (FAQs)
faq_tree, faqs = None, None
try:
    faq_tree = ET.parse('FAQs.xml')
    faqs = faq_tree.getroot()
except FileNotFoundError:
    print("😰 Sorry, cannot find FAQ file, "
          "please put 'FAQs.xml' file at the same dir as this python file and try again. "
          "Email to xyzhu@utdallas.edu if necessary.")
    exit(0)
except ET.ParseError:
    print("😰 Sorry, we find this 'FAQs.xml' cannot be parsed properly. "
          "Please check file integrity or re-download the project. "
          "Email to xyzhu@utdallas.edu if necessary.")
    exit(0)
print("✌️ Parse FAQ file successfully.")

# Establish corpus and extract features
print("🤔️ Processing FAQ data...")
corpus = []
for faq in faqs:
    corpus.append(FAQ(faq[0].text, faq[1].text))
    print(len(corpus), "\tProcessed.")
print("✌️ FAQ data processed.")

# Listen user input
while True:
    u_input = input("Please input your question (Input \"help\" to see manual):\n")

    # input & help & debug
    if u_input == "help":
        print("Manual: ")
    elif u_input == "show raw":
        ET.dump(faqs)
    elif u_input == "show":
        for faq in corpus:
            print(corpus.index(faq))
            faq.print()
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
        input_bag = word_tokenize(input_str)

        # parse bag to counter
        input_counter = Counter(input_bag)

        # POS Tagging (As feature)
        input_pos = nltk.pos_tag(input_bag)  # Tuple (word, POS)

        # Remove stop words
        input_sw_removed = [w for w in input_bag if w.lower() not in stop_words]

        # Stem (As feature) & Lemmatize (As feature)
        input_stemmed = []
        input_lemmatized = []
        for word in input_bag:
            input_stemmed.append(stemmer.stem(word))
            input_lemmatized.append(lemmatizer.lemmatize(word))

        # Tree Parse (As feature)
        input_tree = parser.raw_parse_sents(input_sents)

        # WordNet hypernymns, hyponyms, meronyms, AND holonyms (As feature)
        input_hypernymns = []
        input_hyponyms = []
        input_meronyms = []
        input_holonyms = []
        input_bag_counter = Counter(input_sw_removed)
        for word in input_bag_counter.keys():
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
                if target_synset.hyponyms():
                    input_hyponyms += target_synset.hyponyms()
                if target_synset.part_meronyms():
                    input_meronyms += target_synset.part_meronyms()
                if target_synset.part_holonyms():
                    input_holonyms += target_synset.part_holonyms()

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
                corpus[index].print()
                print()
                show_count += 1