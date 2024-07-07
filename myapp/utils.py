"""from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
from transformers import pipeline

# Charger le pipeline de résumé avec le modèle BART
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def preprocess_text(text):
    # Normaliser et nettoyer le texte si nécessaire
    return text

def generate_summary_supervised(text):
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

def generate_summary_unsupervised(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85)
    tfidf_matrix = vectorizer.fit_transform(text.split('. '))
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    sentences = text.split('. ')
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    top_indices = sentence_scores.argsort()[-top_n:][::-1]
    
    top_sentences = [sentences[i] for i in top_indices]
    summary = '. '.join(top_sentences)
    
    return summary"""
    
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

# Charger le pipeline de résumé avec le modèle BART
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Charger le modèle et le tokenizer T5
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

def preprocess_text(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    return tfidf_matrix, vectorizer

def generate_summary_supervised(text):
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

def generate_summary_unsupervised(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    tfidf_scores = tfidf_matrix.toarray()[0]

    top_indices = tfidf_scores.argsort()[-top_n:][::-1]
    top_features = feature_names[top_indices]

    sentences = text.split('.')
    summary_sentences = [sentence for sentence in sentences if any(feature in sentence for feature in top_features)]

    summary = '. '.join(summary_sentences)
    return summary


def generate_summary_t5(text):
    inputs = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


