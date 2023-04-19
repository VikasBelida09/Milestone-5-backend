from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
from utils.preprocess import preprocess_df

def model_topics(df, feature_name):
    tf_idf_vect = TfidfVectorizer()
    tfidf_vectors = tf_idf_vect.fit_transform(df[feature_name])
    lda = LatentDirichletAllocation(n_components=5)
    lda.fit(tfidf_vectors)
    feature_names = tf_idf_vect.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_word_indices = topic.argsort()[:-11:-1]
        top_word_importances = [topic[i] / topic.sum() for i in top_word_indices]
        top_words = [feature_names[i] for i in top_word_indices]
        topic_words = list(zip(top_words, top_word_importances))
        topics.append(topic_words)
    return topics

def get_topic_modelling():
    zoom_df = pd.read_csv('./datasets/zoom.csv')
    zoom_df = preprocess_df(zoom_df)

    old_df = zoom_df[zoom_df['Release Date'] < '2022-06-30']
    new_df = zoom_df[zoom_df['Release Date'] >= '2022-06-30']   
    old_topics = model_topics(old_df, 'Feature Description')
    new_topics = model_topics(new_df, 'Feature Description')
    return {'old_feature_topics':old_topics,'new_feature_topics':new_topics}

def get_topic_modelling_web():
    zoom_df = pd.read_csv('./datasets/webex-dataset.csv')
    zoom_df = preprocess_df(zoom_df)

    old_df = zoom_df[zoom_df['Release Month'] <= 6]
    new_df = zoom_df[zoom_df['Release Month'] > 6]   
    old_topics = model_topics(old_df, 'Feature Description')
    new_topics = model_topics(new_df, 'Feature Description')
    return {'old_feature_topics':old_topics,'new_feature_topics':new_topics}