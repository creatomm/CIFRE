from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
import networkx as nx
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# Initialisation du pipeline de résumé supervisé avec le modèle BART
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialisation du modèle T5 pour le résumé
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

def preprocess_text(document):
    """
    Prétraite le texte en calculant la matrice TF-IDF.
    
    Args:
        document: Le texte à prétraiter.
        
    Returns:
        La matrice TF-IDF et le vectorizer utilisé.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([document])
    return tfidf_matrix, vectorizer

def generate_summary_supervised(text):
    """
    Génère un résumé supervisé en utilisant le modèle BART.
    
    Args:
        text: Le texte à résumer.
        
    Returns:
        Le résumé généré.
    """
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

def generate_summary_unsupervised(text, top_n=5):
    """
    Génère un résumé non supervisé en utilisant l'algorithme TextRank.
    
    Args:
        text: Le texte à résumer.
        top_n: Le nombre de phrases à inclure dans le résumé.
        
    Returns:
        Le résumé généré.
    """
    # Étape 1 : Diviser le texte en phrases
    sentences = sent_tokenize(text)
    
    # Étape 2 : Calculer la représentation TF-IDF des phrases
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Étape 3 : Calculer la matrice de similarité
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Étape 4 : Construire le graphe et calculer les scores TextRank
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    # Étape 5 : Classer les phrases en fonction des scores et sélectionner les top N phrases
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_sentences = [ranked_sentences[i][1] for i in range(top_n)]
    
    # Étape 6 : Joindre les top phrases pour former le résumé
    summary = ' '.join(top_sentences)
    return summary

def generate_summary_t5(text):
    """
    Génère un résumé du texte fourni en utilisant le modèle T5.

    Cette fonction utilise le modèle T5 de la bibliothèque Transformers pour générer un résumé du texte d'entrée.
    Le modèle T5 est un modèle de traduction de texte en texte qui peut être utilisé pour diverses tâches de traitement du langage naturel, 
    y compris la génération de résumés.

    Paramètres:
    text (str): Le texte d'entrée à résumer.s

    Retourne:
    str: Un résumé du texte d'entrée sans les tokens spéciaux (<pad>, </s>, etc.).
    """
    inputs = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary



def generate_summary_sumy(text, sentence_count=3):
    """
    Génère un résumé en utilisant l'algorithme LSA via la bibliothèque Sumy.
    
    Args:
        text: Le texte à résumer.
        sentence_count: Le nombre de phrases à inclure dans le résumé.
        
    Returns:
        Le résumé généré.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

