from flask import Flask, render_template,request
from werkzeug.utils import secure_filename
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from gtts import gTTS
import os
import numpy as np
import networkx as nx

app=Flask(__name__)

def read_article(file_name):
    #file = open(file_name, "r")
    file=file_name
    filedata = file.readlines()
    print(filedata)
    article = filedata[0].decode("utf-8").split(". ")
    sentences = []

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text and split it
    sentences = read_article(file_name)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    tbar = " ".join(summarize_text) 
    print("Summary is\n")
    print(tbar)
    language = 'en'
    output = gTTS(text = tbar, lang=language,slow = False)
    output.save("sample1.mp3")
    return tbar
     

 

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/upload',methods=["POST"])
def upload():
    return render_template('choosefile.html')

@app.route('/summary',methods=["POST"])
def filename():
    file=request.files['file']
    text = request.form['input']
    summary = generate_summary( file, int(text))
    return render_template('landing.html',summary=summary)


    
    




if __name__ == "__main__":
    app.run(debug=True)