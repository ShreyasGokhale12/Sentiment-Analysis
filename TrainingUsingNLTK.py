import nltk
import random
import pickle
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from nltk.classify import ClassifierI
from statistics import mode


short_pos = open("positive.txt","r", encoding='utf-8', errors='replace').read()
short_neg = open("negative.txt","r", encoding='utf-8', errors='replace').read()

documents = []
all_words = []
allowed_word_types = ['J']              # J stands for words which are adjectives. We wil consider adjectives only 

for r in short_pos.split('\n'):
    documents.append((r,'pos'))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)           # pos_tag = Part of Speech tagging to know which words are adjectives
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for r in short_neg.split('\n'):
    documents.append((r,'neg'))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
            
#---------------------------------------------------------------------------------------------------------------------#        

saved_documents = open('pickled_algos/documents.pickle','wb')
pickle.dump(documents , saved_documents)
saved_documents.close()

#---------------------------------------------------------------------------------------------------------------------#            

all_words = nltk.FreqDist(all_words)
word_features = list(all_words)[:5000]

saved_word_features = open('pickled_algos/word_features.pickle','wb')
pickle.dump(word_features , saved_word_features)
saved_word_features.close()

#---------------------------------------------------------------------------------------------------------------------#

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)      #returns boolean whether that word is present in top 5000 words
        
    return features

featuresets = [(find_features(sentence),category) for (sentence , category) in documents]

saved_featureset = open('pickled_algos/featuresets.pickle','wb')
pickle.dump(featuresets ,saved_featureset)
saved_featureset.close()

#---------------------------------------------------------------------------------------------------------------------#

random.shuffle(featuresets)

print(len(featuresets))

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

#---------------------------------------------------------------------------------------------------------------------#

classifier = nltk.NaiveBayesClassifier.train(training_set)
print('Original Accuracy on the training set in percent : ' , nltk.classify.accuracy(classifier , testing_set)*100)
classifier.show_most_informative_features(15)

save_classifier = open('pickled_algos/NaiveBayesClassifier','wb')
pickle.dump(classifier , save_classifier)
save_classifier.close()

#---------------------------------------------------------------------------------------------------------------------#

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open('pickled_algos/MNB_classifier','wb')
pickle.dump(MNB_classifier , save_classifier)
save_classifier.close()

#---------------------------------------------------------------------------------------------------------------------#

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open('pickled_algos/BernoulliNB_classifier','wb')
pickle.dump(BernoulliNB_classifier , save_classifier)
save_classifier.close()

#---------------------------------------------------------------------------------------------------------------------#

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open('pickled_algos/LogisticRegression_classifier','wb')
pickle.dump(LogisticRegression_classifier , save_classifier)
save_classifier.close()

#---------------------------------------------------------------------------------------------------------------------#

                               
