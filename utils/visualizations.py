import pandas as pd
import numpy as np
import json
from collections import Counter
from utils.preprocess import preprocess_df
import nltk

def get_releases_per_month():
    zoom_df = pd.read_csv('./datasets/zoom.csv')

    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    zoom_df['Release Month'] = zoom_df['Release Month'].apply(lambda x: month_names[int(x)-1])

    releases_per_month = zoom_df.groupby('Release Month')['Feature Title'].count().reset_index()

    releases_per_month['Release Month'] = pd.Categorical(releases_per_month['Release Month'], categories=month_names, ordered=True)
    releases_per_month = releases_per_month.sort_values('Release Month')

    releases_json = json.loads(releases_per_month.to_json(orient='records'))
    result={d['Release Month']:d['Feature Title'] for d in releases_json}
    return result

def get_most_occured_words():
    zoom_df = pd.read_csv('./datasets/zoom.csv')
    zoom_df = preprocess_df(zoom_df)
    all_features = " ".join(zoom_df['Feature Description'].astype(str))
    words = all_features.split()
    bigrams = list(nltk.bigrams(words))
    word_counts = Counter(words)
    bigram_counts = Counter(bigrams)
    most_common_words = dict(word_counts.most_common(15))
    most_common_bigrams = dict(bigram_counts.most_common(15))
    most_common_bigrams = {str(k[0]+' '+k[1]): v for k, v in most_common_bigrams.items()}
    return {'most_common_words': most_common_words, 'most_common_bigrams': most_common_bigrams}

def get_most_occured_words_with_stop_words():
    zoom_df = pd.read_csv('./datasets/zoom.csv')
    all_features = " ".join(zoom_df['Feature Description'].astype(str))
    words = all_features.split()
    bigrams = list(nltk.bigrams(words))
    word_counts = Counter(words)
    bigram_counts = Counter(bigrams)
    most_common_words = dict(word_counts.most_common(15))
    most_common_bigrams = dict(bigram_counts.most_common(15))
    most_common_bigrams = {str(k[0]+' '+k[1]): v for k, v in most_common_bigrams.items()}
    return {'most_common_words': most_common_words, 'most_common_bigrams': most_common_bigrams}

def get_word_cloud():
    zoom_df = pd.read_csv('./datasets/zoom.csv')
    zoom_df=preprocess_df(zoom_df)
    all_features=" ".join(zoom_df['Feature Description'].astype(str))
    words=all_features.split()
    word_counts=Counter(words)
    dictionary=dict(word_counts)
    return dictionary


def get_releases_per_month_webex():
    webex_df = pd.read_csv('./datasets/webex-dataset.csv')

    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    webex_df['Release Month'] = webex_df['Release Month'].apply(lambda x: month_names[int(x)-1])

    releases_per_month = webex_df.groupby('Release Month')['Feature Title'].count().reset_index()

    releases_per_month['Release Month'] = pd.Categorical(releases_per_month['Release Month'], categories=month_names, ordered=True)
    releases_per_month = releases_per_month.sort_values('Release Month')

    releases_json = json.loads(releases_per_month.to_json(orient='records'))
    result={d['Release Month']:d['Feature Title'] for d in releases_json}
    return result

def get_most_occured_words_webex():
    webex_df = pd.read_csv('./datasets/webex-dataset.csv')
    webex_df=preprocess_df(webex_df)
    all_features=" ".join(webex_df['Feature Description'].astype(str))
    words=all_features.split()
    bigrams = list(nltk.bigrams(words))
    word_counts = Counter(words)
    bigram_counts = Counter(bigrams)
    most_common_words = dict(word_counts.most_common(15))
    most_common_bigrams = dict(bigram_counts.most_common(15))
    most_common_bigrams = {str(k[0]+' '+k[1]): v for k, v in most_common_bigrams.items()}
    return {'most_common_words': most_common_words, 'most_common_bigrams': most_common_bigrams}

def get_most_occured_words_webex_wtih_stopwords():
    webex_df = pd.read_csv('./datasets/webex-dataset.csv')
    all_features=" ".join(webex_df['Feature Description'].astype(str))
    words=all_features.split()
    bigrams = list(nltk.bigrams(words))
    word_counts = Counter(words)
    bigram_counts = Counter(bigrams)
    most_common_words = dict(word_counts.most_common(15))
    most_common_bigrams = dict(bigram_counts.most_common(15))
    most_common_bigrams = {str(k[0]+' '+k[1]): v for k, v in most_common_bigrams.items()}
    return {'most_common_words': most_common_words, 'most_common_bigrams': most_common_bigrams}

def get_word_cloud_webex():
    webex_df = pd.read_csv('./datasets/webex-dataset.csv')
    webex_df=preprocess_df(webex_df)
    all_features=" ".join(webex_df['Feature Description'].astype(str))
    words=all_features.split()
    word_counts=Counter(words)
    dictionary=dict(word_counts)
    return dictionary
