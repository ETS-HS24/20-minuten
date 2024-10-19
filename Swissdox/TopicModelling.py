# Import required libraries
import nltk
import spacy
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from gensim.models import LdaModel
import numpy as np
import pandas as pd

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize spacy model and lemmatizer
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()

# Run this before in your cmd line:
# pip install gensim nltk spacy scikit-learn
# python -m spacy download en_core_web_sm

# Define a preprocessing function
def preprocess(texts):
    stop_words = set(stopwords.words('english'))
    processed_texts = []
    for doc in texts:
        # Tokenize the document
        doc = nlp(doc.lower())  # Lowercase and tokenize
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]

        # Lemmatize words and remove stopwords
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

        processed_texts.append(lemmatized_tokens)

    return processed_texts


# Function to perform topic modeling
def topic_modeling(texts, num_topics=5, num_words=10):
    # Preprocess the texts
    processed_texts = preprocess(texts)

    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(processed_texts)

    # Create a corpus: Term Document Frequency (Bag of Words model)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Build the LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Print the topics
    topics = lda_model.print_topics(num_topics=num_topics, num_words=num_words)
    for idx, topic in topics:
        print(f"Topic {idx + 1}: {topic}")

    return lda_model, corpus, dictionary


# Example usage with sample articles
if __name__ == '__main__':
    # Example newspaper articles (replace these with your actual articles)
    articles = [
        "The economy is facing significant challenges due to rising inflation and unemployment.",
        "Political tensions are rising as governments struggle to control the pandemic.",
        "Technology companies are seeing significant growth in the stock market.",
        "Sports teams are adjusting to new protocols due to the ongoing global health crisis.",
        "Environmental concerns are leading to an increase in renewable energy projects."
    ]

    # Run topic modeling
    lda_model, corpus, dictionary = topic_modeling(articles, num_topics=3, num_words=5)
