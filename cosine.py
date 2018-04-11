"""
Final Project - Natural Language Processing - S18
AXYZ group
Xinyang Zhu / 2021409044 / XXZ170001 / xyzhu@utdallas.edu / (469)-974-7431
Alvin Zhang / 2021352902 / YXZ165231 / yxz165231@utdallas.edu / (469)-500-2197

This document is the property of Xinyang Zhu & Alvin Zhang
This document may not be reproduced or transmitted in any form,
in whole or in part, without the express written permission of
Xinyang Zhu & Alvin Zhang

cosine.py
This file is mainly use for Task 2
"""

import math
import re
from collections import Counter

WORD = re.compile(r'\w+')


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def test():
    print("Input two sentence then I will tell you the cosine distance")

    text1 = input("Sentence 1: ")
    text2 = input("Sentence 2: ")

    vector1 = text_to_vector(text1)
    print(vector1)
    vector2 = text_to_vector(text2)
    print(vector2)

    cosine = get_cosine(vector1, vector2)

    print('Result:', cosine)
