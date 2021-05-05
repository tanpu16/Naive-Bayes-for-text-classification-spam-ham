import os
import pandas as pd
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords 
import math
 
nltk.download('stopwords')

path_spam = "train/spam/"
path_ham = "train/ham/"

stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))

totalspamwordcount = 0
totalspamfilecount = 0
totalhamwordcount = 0
totalhamfilecount = 0
spam_words_prob = {} #spam word and it's conditional probability
ham_words_prob = {}   #ham word and it's conditional probability
spam_words_dict ={}    #word and it's count in spam train files
ham_words_dict={}     ##word and it's count in ham train files
prior_spam = 0
prior_ham =0
spam_no_word_probability = 0
ham_no_word_probability = 0

totalCorrectPredSpam = 0;
totalCorrectPredHam = 0;


def processfiles(file_path,is_stop_param):
    f= open(file_path,'r',errors='ignore')
    #text = f.read().lower()
    wordlist = list()
    for line in f:
        line = line.rstrip()
        words = re.sub('[^a-zA-Z0-9 ]','',line).split()
        for word in words:
            wordlist.append(word)
    wordlist = [stemmer.stem(word) for word in list(filter(None,wordlist))]
    if is_stop_param == True:
        wordlist = [word for word in wordlist if not word in stop_words]
    return wordlist


def multiNomialNBModel(is_stop_param):
    
    global totalspamwordcount,totalspamfilecount,totalhamwordcount,totalhamfilecount,spam_words_dict,ham_words_dict
    global prior_spam,prior_ham,spam_words_prob,ham_words_prob,spam_no_word_probability,ham_no_word_probability
            
    for file in os.listdir(path_spam):
        if file.endswith(".txt"):
            file_path = f"{path_spam}{file}"
            wordlistspam = processfiles(file_path,is_stop_param)
            for word in wordlistspam:
                if word in spam_words_dict:
                    spam_words_dict[word] +=1
                else:
                    spam_words_dict[word] =1
                    spam_words_prob[word] = 0
                totalspamwordcount +=1
            totalspamfilecount+=1


    for file in os.listdir(path_ham):
        if file.endswith(".txt"):
            file_path = f"{path_ham}{file}"
            wordlistham = processfiles(file_path,is_stop_param)
            for word in wordlistham:
                if word in ham_words_dict:
                    ham_words_dict[word] +=1
                else:
                    ham_words_dict[word] =1
                    ham_words_prob[word] = 0
                totalhamwordcount +=1
            totalhamfilecount+=1
    
    prior_spam = totalspamfilecount/(totalspamfilecount+totalhamfilecount)
    prior_ham = totalhamfilecount/(totalspamfilecount+totalhamfilecount)
    prior_spam = math.log(prior_spam)
    prior_ham = math.log(prior_ham)
    
    
    for word in spam_words_prob:
        k=1
        count_word = spam_words_dict[word]
        spam_words_prob[word] = math.log((count_word+k)/(totalspamwordcount+(k*len(spam_words_dict))))
        
    for word in ham_words_prob:
        k=1
        count_word = ham_words_dict[word]
        ham_words_prob[word] = math.log((count_word+k)/(totalhamwordcount+(k*len(ham_words_dict))))
    #for all the words not in spam or ham vocab
    k=1
    spam_no_word_probability = math.log(k/(totalspamwordcount+(k*len(spam_words_dict))))
    ham_no_word_probability = math.log((k/(totalhamwordcount+(k*len(ham_words_dict)))))

def applyMultinomialNB(is_stop_param, is_train):

    global spam_words_prob,ham_words_prob,spam_no_word_probability,ham_no_word_probability
    path_spam_test = "test/spam/"
    path_ham_test = "test/ham/"
    
    if is_train == True:
        path_spam_test = "train/spam/"
        path_ham_test = "train/ham/"
    
    accurate_prediction = 0
    total_word_count = 0
    
    for file in os.listdir(path_spam_test):
        if file.endswith(".txt"):
            file_path = f"{path_spam_test}{file}"
            wordlistspamtest = processfiles(file_path,is_stop_param)
            spam_words_prob_test = {}
            ham_words_prob_test = {}
            spam_words_dict_test = {}
            ham_words_dict_test = {}
            for word in wordlistspamtest:
                k=1  #laplace smoothing
                #for calculating probability of word given spam
                if word not in spam_words_dict_test :
                    spam_words_dict_test[word] =1
                    if word in spam_words_dict:
                        spam_words_prob_test[word] = spam_words_prob[word]
                    else:
                        spam_words_prob_test[word] = spam_no_word_probability
                else:
                        spam_words_dict_test[word] +=1
                        
                #for calculating probability of word given spam
                if word not in ham_words_dict_test :
                    ham_words_dict_test[word] =1
                    if word in ham_words_dict:
                        ham_words_prob_test[word] = ham_words_prob[word]
                    else:
                        ham_words_prob_test[word] = ham_no_word_probability
                else:
                        ham_words_dict_test[word] +=1

            #spam probaability
            spam_probability = prior_spam
            for w in spam_words_prob_test:
                spam_probability = spam_probability + (spam_words_prob_test[w]*spam_words_dict_test[w])
            spam_probability = math.exp(spam_probability)
            #ham probability
            ham_probability = prior_ham
            for w in ham_words_prob_test:
                ham_probability = ham_probability + (ham_words_prob_test[w]*ham_words_dict_test[w])
            ham_probability = math.exp(ham_probability)
            
            if spam_probability >= ham_probability:
                accurate_prediction +=1
            total_word_count +=1
            
        for file in os.listdir(path_ham_test):
            if file.endswith(".txt"):
                file_path = f"{path_ham_test}{file}"
                wordlisthamtest = processfiles(file_path,is_stop_param)
                spam_words_prob_test = {}
                ham_words_prob_test = {}
                spam_words_dict_test = {}
                ham_words_dict_test = {}
                for word in wordlisthamtest:
                    k=1  #laplace smoothing
                    #for calculating probability of word given spam
                    if word not in spam_words_dict_test :
                        spam_words_dict_test[word] =1
                        if word in spam_words_dict:
                            spam_words_prob_test[word] = spam_words_prob[word]
                        else:
                            spam_words_prob_test[word] = spam_no_word_probability
                    else:
                            spam_words_dict_test[word] +=1

                    #for calculating probability of word given spam
                    if word not in ham_words_dict_test :
                        ham_words_dict_test[word] =1
                        if word in ham_words_dict:
                            ham_words_prob_test[word] = ham_words_prob[word]
                        else:
                            ham_words_prob_test[word] = ham_no_word_probability
                    else:
                            ham_words_dict_test[word] +=1

                #spam probaability
                spam_probability = prior_spam
                for w in spam_words_prob_test:
                    spam_probability = spam_probability + (spam_words_prob_test[w]*spam_words_dict_test[w])
                spam_probability = math.exp(spam_probability)
                #ham probability
                ham_probability = prior_ham
                for w in ham_words_prob_test:
                    ham_probability = ham_probability + (ham_words_prob_test[w]*ham_words_dict_test[w])
                ham_probability = math.exp(ham_probability)

                if ham_probability >= spam_probability:
                    accurate_prediction +=1
                total_word_count +=1

        accuracy = accurate_prediction/total_word_count
        return accuracy
		
print("******************Train Set***************")
print("With stop word :")
multiNomialNBModel(False)
accuracy_ws = applyMultinomialNB(False,True)
print("Accuracy on train set with stop word : {}".format(accuracy_ws))

# print("\nWithout stop word :")
# multiNomialNBModel(True)
# accuracy_s = applyMultinomialNB(True,True)
# print("Accuracy on train set without stop word : {}".format(accuracy_s))

print("\n\n******************Test Set***************")
print("With stop word :")
multiNomialNBModel(False)
accuracy_ws = applyMultinomialNB(False,False)
print("Accuracy on test set with stop word : {}".format(accuracy_ws))

print("\nWithout stop word :")
multiNomialNBModel(True)
accuracy_s = applyMultinomialNB(True,False)
print("Accuracy on test set without stop word : {}".format(accuracy_s))
