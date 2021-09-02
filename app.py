from flask import Flask,render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot


app = Flask(__name__)

model = load_model('model/lstm_fakenews_model.h5')

@app.route('/',methods=['GET'])
def index():
    return  render_template('index.html')



@app.route('/result',methods=['POST'])
def fakeNewsDetector():
    if request.method == 'POST':
   
         message = str(request.form['title'])
         ps = PorterStemmer()

         corpus=[]
         review = re.sub('[^a-zA-Z]',' ',message)
         review = review.lower()
         review = review.split()
    
         review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
         review = ' '.join(review)
         corpus.append(review)
         
         onehot_corpus = [one_hot(words,5000)for words in corpus]
         x = pad_sequences(onehot_corpus,padding='pre',maxlen=20)
         
         result= model.predict_classes(x)
         result = result[0][0]
         return  render_template('index.html', result=result)
         
if __name__=="__main__":
    app.run(debug=True)
