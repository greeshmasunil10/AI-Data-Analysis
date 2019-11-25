import pandas as pd
import os
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer 
import numpy as np
from sklearn.metrics import confusion_matrix
import time


inputfile="Resources\\hn2018_2019.csv"
inputfile="Resources\\sample100.csv"
def trainModel():
    global dftrain
    dftrain= pd.read_csv(inputfile,encoding='ISO-8859-1',usecols=['Title', 'Post Type','Created At'])
    dftrain.columns = dftrain.columns.str.replace(' ', '')
    
    dftrain = dftrain[(dftrain.CreatedAt<'2019')]
    dftrain=tokenise_model(dftrain)
    dftest = dftrain[(dftrain.CreatedAt>'2018')]
    dftest=tokenise_model(dftest)
    dftrain=dftrain.drop(columns=['CreatedAt'])
#     print(dftrain.head())
# 
#     print(len(dftrain))
#     print(len(dftrain[dftrain.PostType==0]))
#     print(dftrain.head())
    

    
    cv = CountVectorizer()
    cv2= CountVectorizer()
    
    counts = cv.fit_transform(dftrain['Title'])
    counts2= cv2.fit_transform(dftest['Title'])
#     feat_dict=cv.vocabulary_.keys()
    transformer = TfidfTransformer().fit(counts)
    counts = transformer.transform(counts)
    
    transformer2 = TfidfTransformer().fit(counts2)
    counts2 = transformer2.transform(counts2)
#     print("printing...")
#     print(len(cv.vocabulary_.keys()))
#     print("featurenames")
#     print(cv.get_feature_names())
    print(cv.vocabulary_)
#     print(cv)
#     print(counts.shape)
#     print(counts.toarray())
#     print(counts)
#     dfk= pd.DataFrame(counts.toarray(), columns = cv.get_feature_names())
#     print(dfk)


#     print(cv.vocabulary_)
    # encode document
#     vector = cv.transform(dftrain.Title)
#     # summarize encoded vector
#     print("shape",vector.shape)
#     print("type",type(vector))
#     print("array",vector.toarray())
    
    
#     print((counts))
#     print(cv.get_feature_names())
#     for k,v in counts.items():
#         print(k,v)

    X_train, X_test, y_train, y_test = train_test_split(counts, dftrain['Title'], test_size=0.1)
#     X_train=dftrain
#     y_train= dftrain.PostType.tolist()
#     y_test= dftest.PostType.tolist()
#     X_train= dftrain.Title.tolist()
#     X_test=dftest.Title.tolist()
    print("ytrain",y_train)
    print("xtrain",X_train)
#     y_train=dftrain['Title']
# #     , X_test, y_train, y_test = train_test_split(counts, df['Title'], test_size=0.1, random_state=69)
    model = MultinomialNB().fit(X_train, y_train)
    print(model.predict_log_proba(X_train))
    print("predicted...")
    predicted = model.predict(X_test)
#     print(type(X_test))
#     print(X_train)
#     for it in model.predict_log_proba(X_train):
#         print(it)
#     print(len(model.predict_log_proba(X_train)))    
#     print(len(model.predict_log_proba(X_test)))    
#     print(predicted)
#     print(np.mean(predicted == y_test))
# #     print(confusion_matrix(y_test, predicted))
#     print("test....\n",y_test)
#     print()
#     print("predicted.....\n",predicted)
#     for it in predicted:
#         print(it)
#     print('_________________________________________________')    
#     for it in y_test:
#         print(it)
#     print(confusion_matrix(y_test, predicted).shape)
#     os.system("Notepad.exe op.txt")
#     print(dftrain)
#     print(dftrain.'Post_type')
#     print(dftrain.PostType)  
def tokenise_model(df):  
    df['PostType'] = df.PostType.map({'story': 0, 'ask_hn': 1, 'show_hn':2, 'post':3})
    df['Title'] = df.Title.map(lambda x: x.lower())
    df['Title'] = df.Title.str.replace('[^\w\s]', '')
    df['Title'] = df['Title'].apply(nltk.word_tokenize)
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    df['Title'] = df['Title'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
    df['Title'] = df['Title'].apply(lambda x: ' '.join(x))
    return df
    
    
def display_df():
    f1 = open("op.txt", "w",encoding='ISO-8859-1')
#     f1.write(str(df))
    f1.close()
def startprogram():
    trainModel()    
st= time.time()
startprogram() 
et= time.time()  
print("Total elapsed time:",round(et - st,1),"seconds")
