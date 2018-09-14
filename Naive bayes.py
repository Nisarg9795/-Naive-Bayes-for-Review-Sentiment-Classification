import fnmatch
import os
import re
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
Ratio = 0.3
path = '''D:/ASU/Fall'18/SML/Assignment/Assignment 1/movie review data/'''

def review(filename, path):
    fp = open(path+filename, 'r', encoding='utf-8')
    file_r = fp.read()
    file_s= re.sub('[^A-Za-z]', ' ', file_r)
    file_l = file_s.lower()
    file_w = word_tokenize(file_l)
    for w in file_w:
        if w in stopwords.words('english'):
            file_w.remove(w)

    for a in range(len(file_w)):
        file_w[a] = ps.stem(file_w[a])

    freq_d = {x: file_w.count(x) for x in file_w}
    return freq_d

neg_comp = []
pos_comp = []

for root, dirs, files in os.walk(path+'neg'):
    neg_comp += fnmatch.filter(files, '*.txt')
for root, dirs, files in os.walk(path+'pos'):
    pos_comp += fnmatch.filter(files, '*.txt')
print(len(pos_comp))
print(len(neg_comp))

#Splitting data into test and train
pos_train, pos_test = train_test_split(pos_comp, train_size=Ratio)
neg_train, neg_test = train_test_split(neg_comp, train_size=Ratio)

list_negative = {}
for i in range(len(neg_train)):
    temp = review(neg_train[i], path+'neg/')
    list_negative = {x: temp.get(x, 0) + list_negative.get(x, 0) for x in set(temp).union(list_negative)}

fop = open(path+'list_negative.txt', 'w')
for key in list_negative.keys():
    fop.write(str(key)+'--'+str(list_negative[key])+'\n\n')
fop.close()

list_positive = {}
for i in range(len(pos_train)):
    temp = review(pos_train[i], path+'pos/')
    list_positive = {x: temp.get(x, 0) + list_positive.get(x, 0) for x in set(temp).union(list_positive)}

fop = open(path+'list_positive.txt', 'w')
for key in list_positive.keys():
    fop.write(str(key)+'--'+str(list_positive[key])+'\n\n')
fop.close()

list_total = {x: list_positive.get(x, 0) + list_negative.get(x, 0) for x in set(list_positive).union(list_negative)}
print(len(list_total))

count_1 = len(dir_pos)
count_0 = len(dir_neg)
P_y_1 = count_1 / (count_0+count_1)
P_y_0 = count_0 / (count_0+count_1)

t = list(list_total.keys())
t.sort()
total_word = len(t)
print(total_word)
wy0 = {}
wy1 = {}
for i in t:
    if i in list_negative.keys():
        wy0[i] = (list_negative[i] / count_0)
    else:
        wy0[i] = 1/(count_0 + total_word)

    if i in list_positive.keys():
        wy1[i] = (list_positive[i] / count_1)
    else:
        wy1[i] = 1/(count_1 + total_word)

def predict(vocab_list):
    yw0 = P_y_0
    yw1 = P_y_1
    for word in vocab_list:
        if word in list_total.keys():
            yw0 *= wy0[word]
            yw1 *= wy1[word]
        else:
            pass
    if yw0 > yw1:
        return 0
    elif yw0 < yw1:
        return 1
    else:
        return 2

count = 0
total = len(pos_test) + len(neg_test)
for file in pos_test:
    test_dict = review(file, path+'pos/')
    ret = predict(list(test_dict.keys()))
    if ret == 1:
        count += 1

for file in neg_test:
    test_dict = review(file, path+'neg/')
    ret = predict(list(test_dict.keys()))
    if ret == 0:
        count += 1

print("Accuracy: ", 100*(count/total))