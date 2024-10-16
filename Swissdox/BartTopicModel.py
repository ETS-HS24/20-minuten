import torch
from torch.utils.data import DataLoader
from transformers import BartForSequenceClassification, BartTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

# Optional preprocessing steps (adjust as needed)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class BartTopicModel:
    def __init__(self, num_topics=10, num_epochs=10, batch_size=32, learning_rate=2e-5, model_name="facebook/bart-base"):
        self.num_topics = num_topics
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.vectorizer = None

        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForSequenceClassification.from_pretrained(model_name, num_labels=num_topics)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def preprocess_text(self, text):
        # Optional preprocessing steps (adjust as needed)
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer

        stop_words = set(stopwords.words("english"))
        stemmer = PorterStemmer()

        words = text.lower().split()
        words = [word for word in words if word not in stop_words]
        words = [stemmer.stem(word) for word in words]
        return " ".join(words)

    def train(self, articles):
        input_ids = self.tokenizer(articles.tolist(), padding=True, truncation=True, return_tensors="pt")["input_ids"]

        for epoch in range(self.num_epochs):
            self.model.train()
            for batch in DataLoader(input_ids, batch_size=self.batch_size):
                outputs = self.model(batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def extract_topics(self, articles):
        with torch.no_grad():
            input_ids = self.tokenizer(articles, padding=True, truncation=True, return_tensors="pt")["input_ids"]
            outputs = self.model(input_ids)
            topic_representations = outputs.logits.cpu().numpy()

        self.vectorizer = CountVectorizer()
        tfidf_transformer = TfidfTransformer()
        X = self.vectorizer.fit_transform(articles)
        X_tfidf = tfidf_transformer.fit_transform(X)

        lda = LatentDirichletAllocation(n_components=self.num_topics, random_state=42)
        lda.fit(X_tfidf)

        return lda

    def print_topics(self, lda, number_of_components=10):
        for i in range(self.num_topics):
            print(f"Topic {i}:")
            print(" ".join(self.vectorizer.get_feature_names_out()[lda.components_[i].argsort()[-number_of_components:]]))

if __name__ == "__main__":
    texts = ["Storm Sabine throws Zurich Airport's plan into disarray. A video from a reader-reporter shows the A380 having to abort the landing. The Singapore Airline A380 had to abort its approach to Kloten Airport. It worked the second time, as a video from a reader-reporter shows. It wasn't just the A380 giant that had to take off. As a reader reporter reports, a Swiss A330-300 was also hit. The plane came from Tel Aviv."]
    bart_topic_model = BartTopicModel()
    lda = bart_topic_model.extract_topics(texts)
    bart_topic_model.print_topics(lda, 10)