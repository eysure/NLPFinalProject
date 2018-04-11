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

import re                               # Regular Expression
import xml.etree.ElementTree as ET      # XML Parser
from collections import Counter         # List Counter
from corpus import Corpus, Entry        # Our corpus model class
from cosine import get_cosine           # For Task 2

TOKENIZER = r"\d+\.\d+|'\w+|[\w\/\-]+"  # Tokenizer
MAX_SHOW = 10                           # Amount of best result will be shown

print("CS6320 Natural Language Processing Final Project by AXZY")

# read corpus (FAQs)
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

# Establish corpus with faq bag
corpus = []
for faq in faqs:
    corpus.append(Entry(faq[0].text, faq[1].text))
print("✌️ Tokenize successfully.")

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
    elif u_input == "show counter":
        for faq in corpus:
            print(corpus.index(faq))
            faq.print_counter()
    else:
        input_bag = re.findall(TOKENIZER, u_input.lower())
        counter_input = Counter(input_bag)

        # Task 2 - Use statistical method (cosine-similarity) to calculate similarity of  user input and every faqs
        cos_counter = {}
        for faq in corpus:
            cos = get_cosine(counter_input, faq.counter)
            cos_counter[corpus.index(faq)] = cos

        show_count = 0
        for index in sorted(cos_counter, key=cos_counter.get, reverse=True):
            if cos_counter[index] > 0 and show_count < MAX_SHOW:
                print("FAQ #", index, "\tSimilarity: ", cos_counter[index])
                corpus[index].print()
                print()
                show_count += 1
