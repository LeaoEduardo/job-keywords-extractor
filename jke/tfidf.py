import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

from jke import STOPWORDS

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TFIDF:
    def __init__(self, ngram):
        self.stop_words = set(stopwords.words('english')).union(STOPWORDS)
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(ngram_range=ngram)

    def remove_html_tags(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        plain_text = soup.get_text()
        return plain_text

    def convert_to_lowercase(self, text):
        return text.lower()

    def remove_special_characters(self, text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    def tokenize_text(self, text):
        return word_tokenize(text)

    def remove_stop_words(self, tokens):
        return [token for token in tokens if token not in self.stop_words]

    def lemmatize_tokens(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess_descriptions(self, html_descriptions):
        preprocessed_descriptions = []

        for html_description in html_descriptions:
            plain_text = self.remove_html_tags(html_description)
            lowercase_text = self.convert_to_lowercase(plain_text)
            clean_text = self.remove_special_characters(lowercase_text)
            tokens = self.tokenize_text(clean_text)
            filtered_tokens = self.remove_stop_words(tokens)
            lemmatized_tokens = self.lemmatize_tokens(filtered_tokens)

            preprocessed_descriptions.append(lemmatized_tokens)

        return preprocessed_descriptions

    def fit_transform_tfidf(self, preprocessed_descriptions):
        # Convert preprocessed descriptions back to strings
        preprocessed_descriptions = [' '.join(tokens) for tokens in preprocessed_descriptions]

        # Fit and transform the preprocessed descriptions
        tfidf_matrix = self.vectorizer.fit_transform(preprocessed_descriptions)

        return tfidf_matrix
    
    def rank_top_words(self, tfidf_matrix, n = 25):
        # Get the feature names (words) from the TF-IDF vectorizer
        feature_names = self.vectorizer.get_feature_names_out()

        # Get the IDF (Inverse Document Frequency) values for each feature (word)
        idf_values = self.vectorizer.idf_

        # Calculate the TF-IDF weights for each word
        tfidf_weights = tfidf_matrix.sum(axis=0).A1

        word_weight_dict = {}

        for word, idf, weight in zip(feature_names, idf_values, tfidf_weights):
            word_weight_dict[word] = weight * idf

        sorted_words_weights = sorted(word_weight_dict.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words_weights[:n]

        return top_words