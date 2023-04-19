import  spacy

def preprocess_df(df2):
    df2['Feature Title']=df2['Feature Title'].str.lower()
    df2['Feature Description']=df2['Feature Description'].str.lower()
    df2['Feature Description']=df2['Feature Description'].apply(remove_stop_words)
    df2['Feature Description']=df2['Feature Description'].apply(lemmatize_text)
    return df2

nlp = spacy.load("en_core_web_sm")
def remove_stop_words(text):
  doc=nlp(text)
  no_stop_words=[token.text for token in doc if not token.is_stop and not token.is_punct]
  return " ".join(no_stop_words)

def lemmatize_text(text):
   doc=nlp(text)
   lemmatized_words=[token.lemma_ for token in doc]
   return ' '.join(lemmatized_words)