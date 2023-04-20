from flask import Flask,jsonify
from flask_cors import CORS
from utils.visualizations import get_releases_per_month,get_most_occured_words,get_word_cloud, get_releases_per_month_webex,get_word_cloud_webex,get_most_occured_words_webex,get_most_occured_words_webex_wtih_stopwords,get_most_occured_words_with_stop_words
import pandas as pd
from utils.topic_modelling import get_topic_modelling,get_topic_modelling_web
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route("/api/hello")
def hello():
    return "hello"

@app.route("/api/zoom_vis",methods = ['GET'])
def get_zoom_visualizations():
    releases_per_month=get_releases_per_month()
    top_15_words=get_most_occured_words()
    word_cloud=get_word_cloud()
    top_15_words_with_stopwords=get_most_occured_words_with_stop_words()
    response=jsonify({'releases_per_month':releases_per_month,'top15':top_15_words['most_common_words'],'bigrams':top_15_words['most_common_bigrams'],'word_cloud':word_cloud,'top15stopwords':top_15_words_with_stopwords['most_common_words'],'bigramsWithStopWords':top_15_words_with_stopwords['most_common_bigrams']})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/api/webex_vis",methods=['GET'])
def get_webex_visualizations():
    releases_per_month=get_releases_per_month_webex()
    top_15_words=get_most_occured_words_webex()
    word_cloud=get_word_cloud_webex()
    top_15_words_with_stopwords=get_most_occured_words_webex_wtih_stopwords()
    response=jsonify({'releases_per_month':releases_per_month,'top15':top_15_words['most_common_words'],'bigrams':top_15_words['most_common_bigrams'],'word_cloud':word_cloud,'top15stopwords':top_15_words_with_stopwords['most_common_words'],'bigramsWithStopWords':top_15_words_with_stopwords['most_common_bigrams']})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
    
@app.route("/api/zoom-topic-modelling",methods=['GET'])
def get_topic_modelling_zoom():
    topics=get_topic_modelling()
    response=jsonify({'old_topics':topics['old_feature_topics'],'new_topics':topics['new_feature_topics']})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
@app.route("/api/webex-topic-modelling",methods=['GET'])
def get_topic_modelling_webex():
    topics=get_topic_modelling_web()
    response=jsonify({'old_topics':topics['old_feature_topics'],'new_topics':topics['new_feature_topics']})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

