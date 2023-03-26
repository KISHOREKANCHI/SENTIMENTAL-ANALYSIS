#### Importing required Libraries
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from nltk.corpus import words
import matplotlib.pyplot as plt
import unidecode
import nltk
import re



#### Importing StopWords
nltk.download('stopwords')
stopwords =  set(stopwords.words("english"))



##### Loading training and testing Data
df_train = pd.read_csv("train.csv",encoding='latin-1')
df_test = pd.read_csv("test.csv",encoding='latin-1')
df_train.head()



### Dimesions of training dataset
df_train.shape

#### Checking for null Values
df_train.isnull().sum()

#### Attributes Information
df_train.info()

### Drop unneccesary features
df_train = df_train.drop(columns='Location')

### Review Dataset
df_train.head()

### Checking unique Sentiments
df_train['Sentiment'].unique()

### Label Encoding the Sentiment Feature
df_train['label'] = df_train.Sentiment.factorize()[0]

##### Review Dataset
df_train.head()

##### Storing unique Sentiments into target category
target_category = df_train['Sentiment'].unique()
target_category



#visuaization
df_train.groupby("Sentiment").Sentiment.count().plot.bar(ylim=0)

df_train.Sentiment.value_counts().plot(kind='pie', y='label',figsize=(10,8),autopct='%1.1f%%')
plt.show()

tweets = df_train.OriginalTweet
tweets.head(10)



#Data Preprocessing
def processing(text): 

    #Convert Accented Characters(û -> u)
    text = unidecode.unidecode(text)

    #Tokenisation
    tokenized_text = word_tokenize(text)

    #Stopwords Removal
    filtered_words = []
    for w in tokenized_text:
      if w not in stopwords:
        filtered_words.append(w)

    #Removing numbers and extra whitespaces using Regex
    pattern = '[a-zA-Z]'
    text = ''
    for i in filtered_words:
      if re.match(pattern,i):
        text += i+" "

    text = text.replace("'", "") 
        
    #lemmatization
    lemmatizer = WordNetLemmatizer()

    #Pos-Tagging
    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    lem_input = nltk.word_tokenize(text)
    lem_text= ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in lem_input])

    #stemming 
    stemmer= PorterStemmer()
    stem_input= nltk.word_tokenize(lem_text)
    stem_text=' '.join([stemmer.stem(word) for word in stem_input])
           
    #remove single letters
    preprocessed_text = ' '.join( [w for w in stem_text.split() if len(w)>1] )
          
    return preprocessed_text



###Apply Preprocessing on OriginalTweets
df_train['OriginalTweet']=df_train['OriginalTweet'].apply(processing)

###OriginalTweets after preprocessing
tweets = df_train['OriginalTweet']
tweets.head()

## Loading targetFeature
sentiment = df_train.Sentiment

##Splitting Data
X_train, X_test, Y_train, Y_test = train_test_split(tweets,sentiment, test_size = 0.3, random_state = 60,shuffle=True)

print(len(X_train))
print(len(X_test))



#Model1 - SGD Classifier
###Pipelining the Model
sgd = Pipeline([('tfidf', TfidfVectorizer()),('sgd', SGDClassifier())])

###Fit Data into the model
sgd.fit(X_train, Y_train)

##Prediction
test_predict = sgd.predict(X_test)

###Accuracy
train_accuracy = round(sgd.score(X_train,Y_train)*100)
test_accuracy =round(accuracy_score(test_predict, Y_test)*100)

print("SGD Train Accuracy Score : {}% ".format(train_accuracy ))
print("SGD Test Accuracy Score  : {}% ".format(test_accuracy ))
print()
print(classification_report(test_predict, Y_test, target_names=target_category))



#### Reviewing Test Dataset
df_test.head()

### Applying Model on Test Dataset
df_test['OriginalTweet'] = df_test['OriginalTweet'].apply(processing)
tweet = df_test['OriginalTweet']
y_predict = sgd.predict(tweet)

## Loading Target Feature
test_sentiments = df_test['Sentiment']

###Test Data Accuracy
test_accuracy =round(accuracy_score(test_sentiments, y_predict)*100)
print("SGD Classifier Test Accuracy Score  : {}% ".format(test_accuracy ))



#Model2 - RandomForest Classifier
###Pipelining the Model
rfc = Pipeline([('tfidf', TfidfVectorizer()),('rfc', RandomForestClassifier())])

###Fit Data into the model
rfc.fit(X_train, Y_train)

##Prediction
test_predict = rfc.predict(X_test)

###Accuracy
train_accuracy = round(rfc.score(X_train,Y_train)*100)
test_accuracy =round(accuracy_score(test_predict, Y_test)*100)

print("RFC Train Accuracy Score : {}% ".format(train_accuracy ))
print("RFC Test Accuracy Score  : {}% ".format(test_accuracy ))
print()
print(classification_report(test_predict, Y_test, target_names=target_category))



### Prediction on Test Dataset
tweet = df_test['OriginalTweet']
y_predict = rfc.predict(tweet)

###Accuracy
test_accuracy =round(accuracy_score(test_sentiments, y_predict)*100)
print("RFC Classifier Test Accuracy Score  : {}% ".format(test_accuracy ))



#Model3 - Logistic Regression
###Pipelining the Model
lr = Pipeline([('tfidf', TfidfVectorizer()),('lr', LogisticRegression())])

###Fit Data into the model
lr.fit(X_train, Y_train)

##Prediction
test_predict = lr.predict(X_test)

###Accuracy
train_accuracy = round(lr.score(X_train,Y_train)*100)
test_accuracy =round(accuracy_score(test_predict, Y_test)*100)

print("LR Train Accuracy Score : {}% ".format(train_accuracy ))
print("LR Test Accuracy Score  : {}% ".format(test_accuracy ))
print()
print(classification_report(test_predict, Y_test, target_names=target_category))



### Prediction on Test Dataset
tweet = df_test['OriginalTweet']
y_predict = lr.predict(tweet)

###Test Data Accuracy
test_accuracy =round(accuracy_score(test_sentiments, y_predict)*100)
print("LR Classifier Test Accuracy Score  : {}% ".format(test_accuracy ))







