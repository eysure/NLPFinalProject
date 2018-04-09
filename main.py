"""
Final Project - Natural Language Processing - S18
AXYZ group
Xinyang Zhu / 2021409044 / XXZ170001 / xyzhu@utdallas.edu / (469)-974-7431
Alvin Zhang / YXZ165231

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
print("✌️ Read FAQs successfully.")

# Establish faq bag
faq_bag = []
for faq in faqs:
    faq_bag.append([re.findall(TOKENIZER, faq[0].text), re.findall(TOKENIZER, faq[1].text)])

# Listen user input
while True:
    u_input = input("Please input your question (Input \"help\" to see manual):\n")

    # input & help & debug
    if u_input == "help":
        print("Manual: ")
    elif u_input == "show raw":
        ET.dump(faqs)
    elif u_input == "show":
        for faq in faqs:
            print(faq.attrib['id'])
            print("\tQ: ", faq[0].text.strip())
            print("\tA: ", faq[1].text.strip())
    elif u_input == "show bag":
        for faq_entry in faq_bag:
            print("faq_bag", "["+str(faq_bag.index(faq_entry))+"]")
            print("\t[0]:", faq_entry[0])
            print("\t[1]:", faq_entry[1])
            print()
    elif u_input == "show bag counter":
        for faq_entry in faq_bag:
            print("faq_bag", "["+str(faq_bag.index(faq_entry))+"]")
            print("\t[0]:", Counter(faq_entry[0]))
            print("\t[1]:", Counter(faq_entry[1]))
            print()
    else:
        input_bag = re.findall(TOKENIZER, u_input.lower())

        # Task 2 - Use statistical method (cosine-similarity) to calculate similarity of  user input and every faqs
        cos_counter = {}
        for faq_entry in faq_bag:
            cos = get_cosine(Counter(input_bag), Counter(faq_entry[0]+faq_entry[1]))
            cos_counter[faq_bag.index(faq_entry)] = cos

        show_count = 0
        for index in sorted(cos_counter, key=cos_counter.get, reverse=True):
            if cos_counter[index] > 0 and show_count < MAX_SHOW:
                print("FAQ #", index, "\tSimilarity: ", cos_counter[index])
                print("\tQ:", " ".join(faq_bag[index][0]))
                print("\tA:", " ".join(faq_bag[index][1]))
                print()
                show_count += 1
