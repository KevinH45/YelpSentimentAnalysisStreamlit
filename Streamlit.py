

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split



st.title("Sentiment Analysis for Yelp Reviews")
st.write("*Please wait while the model loads, it should take around 20 seconds.*")

@st.cache
def train():
    df = pd.read_csv(r"train.csv") 
    df = df.dropna(how='any',axis=0)
    df = df.rename(columns={"1":"Label","Unfortunately, the frustration of being Dr. Goldberg's patient is a repeat of the experience I've had with so many other doctors in NYC -- good doctor, terrible staff.  It seems that his staff simply never answers the phone.  It usually takes 2 hours of repeated calling to get an answer.  Who has time for that or wants to deal with it?  I have run into this problem with many other doctors and I just don't get it.  You have office workers, you have patients with medical needs, why isn't anyone answering the phone?  It's incomprehensible and not work the aggravation.  It's with regret that I feel that I have to give Dr. Goldberg 2 stars.":"Text"})
    vectorizer = TfidfVectorizer(stop_words="english")
    x_train,x_test,y_train,y_test=train_test_split(df["Text"],df["Label"], test_size=0.2, shuffle=True, random_state=7)
    train=vectorizer.fit_transform(x_train) 


    model = MultinomialNB()
    model.fit(train,y_train)

    return(vectorizer,model)

def think(query,vectorizer,model):
    query = vectorizer.transform([query])
    return model.predict(query)

if 'vectorizer' or 'model' not in st.session_state:

    trainRes = train()
    st.session_state.vectorizer= trainRes[0]
    st.session_state.model = trainRes[1]

query = st.text_input("Input rating for sentiment anaylsis")

prediction = think(query,st.session_state.vectorizer,st.session_state.model)

if not query:
    #ignores all blank queries so that the model doesn't predict while empty
    pass
elif prediction==1:
    st.write("The model predicted: negative.")
elif prediction==2:
    st.write("The model predicted: positive.")
else:
    st.write("Something went wrong")
