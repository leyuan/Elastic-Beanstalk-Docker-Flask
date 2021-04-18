from flask import Flask
import pandas as pd
from gensim.utils import tokenize
import pickle
import numpy as np
#Test comment
print("Loading the data...")
#Load data
df_train = pd.read_csv("data/Train.csv")
df_test = pd.read_csv("data/Test.csv")

print("Loading the Machine Learning Models...")
#Load ML model
with open('ml_model/sentiment_model.pk','rb') as f:
    sentiment_predictor = pickle.load(f)
with open('ml_model/featurizer.pk','rb') as f:
    vectorizer = pickle.load(f)

#Predicting on data
cleaned_text_train = [' '.join(i) for i in df_train['text'][0:1000].apply(lambda x:list(tokenize(x,lowercase=True)))]
cleaned_text_test = [' '.join(i) for i in df_test['text'][0:1000].apply(lambda x:list(tokenize(x,lowercase=True)))]
X_train = vectorizer.transform(cleaned_text_train)
X_test = vectorizer.transform(cleaned_text_test)
all_predictions_train = sentiment_predictor.predict(X_train)
all_prediction_test = sentiment_predictor.predict(X_test)
#Initialize Flask App
app = Flask(__name__)


#Route for pages

@app.route("/")
def hello_world():
    return "Hello Careera!"

@app.route("/train_predict")
def predict_train():
    sample = np.random.randint(len(all_predictions_train))
    prediction = all_predictions_train[sample]
    text = cleaned_text_train[sample]
    output = {"Review:":text,"Prediction:":"Positive" if prediction==1 else "Negative"}
    return output




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
