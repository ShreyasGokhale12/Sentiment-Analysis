There are 2 almost similar ways I used to train classifier for sentiment analysis
 
1st method:
The ipython notebook Processing.ipynb uses CountVectorizer from sklearn package to create a bag of words. The features of the classifier are vectors of whether a particular word from bag of words is present in the movie review and length of that particular movie review. The labels are 0(negative) and 1(positive). By adding more and different words to bag of words, the accuracy of classification increased to maximum of 70%. ModelTraining.ipynb contains training of model. This task was part of a kaggle competetion on sentiment analysis on the given dataset. I learnt about CountVectorizer from discussion forums of the competition 

2nd method:
While I was studying Natural Language Toolkit, I came across online articles explaining Sentiment Analysis. This code has some variations from the article I referred to. positive.txt contains positive movie reviews and negative.txt conttains negative movie reviews
