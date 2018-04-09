# NLPFinalProject

## Abstract
This is the CS6320 Natural Language Processing Final Project created and developed by AXYZ group <br />
Xinyang Zhu / 2021409044 / XXZ170001 / xyzhu@utdallas.edu <br />
Alvin Zhang / YXZ165231 <br />

## Task 1: Create a corpus of 50 FAQs and Answers
- We use this file of FAQs.xml as our corpus. It contain more than 50 entries of question and answer of [Amazon Echo (2nd
Generation)](https://www.amazon.com/dp/B06XXM5BPP). <br />
- We extract and parse the top 50 [most helpful questions](https://www.amazon.com/ask/questions/asin/B06XXM5BPP), with
minimal cleaning of typo and indent errors.
- Then we add some "made up" questions just in case TA will ask about them :), index start from 50

## Task 2: Tokenize & Statistical matching algorithm
- We load the FAQs.xml file, tokenize each faq question and answer by word, calculate the count of word, construct
our corpus.
- Create the UI for user, listening user input, then tokenize the input by previous tokenizer
- for each FAQ, we calculate the cosine similarity between user input and FAQ.
[Learn more](https://en.wikipedia.org/wiki/Cosine_similarity)
- Select the top 10 result of FAQ, print on the screen with similarity.
Notice that if similarity is zero, no result will be printed.

## Task 3: Semantically matching algorithm
- TBD