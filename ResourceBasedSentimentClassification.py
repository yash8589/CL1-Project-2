# This module is written to do a Resource Based Semantic analyasis using hindi sentiwordnet.
from nltk.util import pr
import pandas as pd
import codecs
from nltk.tokenize import word_tokenize
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score
import re
from spacy.lang.hi import Hindi


data = pd.read_csv("HindiSentiWordnet.txt", delimiter=' ')
# print((data))

# print(data.index)
fields = ['POS_TAG', 'ID', 'POS', 'NEG', 'LIST_OF_WORDS']

#Creating a dictionary which contain a tuple for every word. Tuple contains a list of synonyms,
# positive score and negative score for that word.
words_dict = {}
for i in data.index:
    # print (data[fields[0]][i], data[fields[1]][i], data[fields[2]][i], data[fields[3]][i], data[fields[4]][i])

    words = data[fields[4]][i].split(',')
    for word in words:
        words_dict[word] = (data[fields[0]][i], data[fields[2]][i], data[fields[3]][i])
    # print(words_dict)

# This function determines sentiment of text.

# print(sentiment("कल गुलाम राम नहीं जीता"))
def sentiment(text):
   
    
    # words = text.split(" ")


    # not_stop_words = [word for word in words if word not in set(STOP_WORDS_HI) ]

    # # print(not_stop_words)
    # words = word_tokenize(not_stop_words)
    nlp = Hindi()
    doc = nlp(text)
    for i in doc:
        text += i.norm_
    # print(text)


    words = word_tokenize(text)

    votes = []
    pos_polarity = 0
    neg_polarity = 0
    neu_polarity = 0
    #adverbs, nouns, adjective, verb are only used
    allowed_words = ['a','v','r','n']
    for word in words:
        if word in words_dict:
            #if word in dictionary, it picks up the positive and negative score of the word
            pos_tag, pos, neg = words_dict[word]
            print(pos_tag, pos, neg)
            # print(word, pos_tag, pos, neg)
            if pos_tag in allowed_words:
                if pos > neg:
                    pos_polarity += pos
                    print(pos_polarity)
                    votes.append(1)
                elif neg > pos:
                    neg_polarity += neg
                    print(neg_polarity)
                    votes.append(-1)
                # elif neg == pos: 
                #     # neu_polarity = pos - neg
                #     print(neu_polarity)
                #     votes.append(0)
        
    #calculating the no. of positive and negative words in total in a review to give class labels
    pos_votes = votes.count(1)
    neg_votes = votes.count(-1)
    # print(votes.count())
    if pos_votes > neg_votes:
        return 1
    elif neg_votes > pos_votes:
        return -1
    else:
        if pos_polarity < neg_polarity:
            return -1
        elif pos_polarity > neg_polarity:
            return 1
        elif pos_polarity == neg_polarity:
            return 0


# pred_y = []
# actual_y = []
# # to calculate accuracy
# pos_reviews = codecs.open("pos_hindi.txt", "r", encoding='utf-8', errors='ignore').read()
# for line in pos_reviews.split('$'):
#     data = line.strip('\n')
#     if data:
#         pred_y.append(sentiment(data))
#         print(pred_y)
#         actual_y.append(1)
#         print(actual_y)
# #print(accuracy_score(actual_y, pred_y) * 100)
# print(len(actual_y))
# neg_reviews = codecs.open("neg_hindi.txt", "r", encoding='utf-8', errors='ignore').read()
# for line in neg_reviews.split('$'):
#     data=line.strip('\n')
#     if data:
#         pred_y.append(sentiment(data))
#         actual_y.append(0)
# print(len(actual_y))
# print(accuracy_score(actual_y, pred_y) * 100)
# print('F-measure:  ',f1_score(actual_y,pred_y))

if __name__ == '__main__':
    # print(sentiment("मैं इस उत्पाद से बहुत खुश हूँ  यह आराम दायक और सुन्दर है  यह खरीदने लायक है "))
    # print(sentiment("एक दिन चुन्नू हिरण उस जंगल में रहने के लिए आया।"))
    # print(sentiment("राम ने इनाम जीता"))
    print(sentiment("राम की मृत्यु हो गयी"))


if "भाग्यवान" in words_dict:
    pos_tag, pos, neg = words_dict["भाग्यवान"]
    print(pos_tag, pos, neg)
else:
    print("Not reading")