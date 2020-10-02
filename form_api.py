from flask import Flask, request, render_template
import gensim
import gensim.downloader as api
from gensim import models
import pandas as pd
import numpy as np
import sklearn
import pickle
import json
from sklearn.metrics.pairwise import cosine_distances

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    #processed_text = text.upper()
    proccessed_text = text.split()
    #load model
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    #calc vector for resume input
    resume_vect = loaded_model.infer_vector(proccessed_text)
    #load job desc vectors
    jd = pd.read_csv('https://raw.githubusercontent.com/AVJdataminer/HireOne/master/data/vectors_data.csv')
    jn = jd.to_numpy()
    #calculate cosine distances
    cos_dist =[]
    for i in range(jd.shape[0]):
        cos_dist.append(float(cosine_distances(resume_vect[0:].reshape(1,-1),jn[i].reshape(1,-1))))
    #load job desc data to return
    df = pd.read_csv('https://raw.githubusercontent.com/AVJdataminer/HireOne/master/data/updated_job_description.csv', encoding = 'unicode_escape')
    role = df['role'].tolist()
    desc = df['description'].tolist()
    summary = pd.DataFrame({
        'Role Title': role,
        'Cosine Distances': cos_dist,
        'Job Description': desc
    })
    z = summary.sort_values(by ='Cosine Distances', ascending=True)
    z = z.head()
    text_vector = z.to_dict()
    return text_vector

if __name__=='__main__':
    app.run(debug=True)