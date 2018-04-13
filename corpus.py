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

import re
from collections import Counter

TOKENIZER = r"\d+\.\d+|'\w+|[\w\/\-]+"  # Tokenizer


def data_cleaning(string):
    string = string.lower()
    return string

# Class of entry of corpus
class Entry:
    def __init__(self, str_q, str_a):
        self.ori_str_q = str_q.strip()
        self.ori_str_a = str_a.strip()
        self.bag_q = re.findall(TOKENIZER, data_cleaning(str_q))
        self.bag_a = re.findall(TOKENIZER, data_cleaning(str_a))
        self.counter_q = Counter(self.bag_q)
        self.counter_a = Counter(self.bag_a)
        self.bag = self.bag_q + self.bag_a
        self.counter = Counter(self.bag)

    def print_ori(self):
        print ("Q: " + self.ori_str_q)
        print ("A: " + self.ori_str_a)

    def print_bag(self):
        print "Q: ", self.bag_q
        print "A: ", self.bag_a

    def print_counter(self):
        print "Q: ", self.counter_q
        print "A: ", self.counter_a

    def concat(self):
        return self.bag_q+self.bag_a


# Class of corpus
class Corpus:
    list = []

    # Add a FAQ entry
    def append(self, entry):
        self.list.append(entry)

    def get(self, index=None):
        if index is not None:
            return self.list[index]
        else:
            return self.list
