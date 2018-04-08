# Final Project - Natural Language Processing - S18
# Xinyang Zhu / 2021409044 / XXZ170001 / xyzhu@utdallas.edu / (469)-974-7431
# Alvin Zhang / YXZ165231
#
# This document is the property of Xinyang Zhu & Alvin Zhang
# This document may not be reproduced or transmitted in any form,
# in whole or in part, without the express written permission of
# Xinyang Zhu & Alvin Zhang

import xml.etree.ElementTree as ET

print("CS6320 Natural Language Processing Final Project by AXZY")

# read corpus (FAQs)
faq_tree = None
faqs = None
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

# Listen user input
while True:
    u_input = input("Please input your question (Input \"help\" to see manual):\n")

    # Help & Debug
    if u_input == "help":
        print("Manual: ")
    elif u_input == "show raw":
        ET.dump(faqs)
    elif u_input == "show":
        for faq in faqs:
            print(faq.attrib['id'])
            print("\tQ: ", faq[0].text.split())
            print("\tA: ", faq[1].text.split())
    else:
        print("Processing your input...")
