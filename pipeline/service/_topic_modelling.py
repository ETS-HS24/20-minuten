import pandas as pd
import logging
import nltk
import spacy
from spacy.cli.download import download as spacy_download
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from pathlib import Path
import os
import datetime
import glob

logger = logging.getLogger(__name__)

# Download NLTK data
for nltk_data in ["stopwords", "wordnet"]:
    try:
        nltk.data.find(nltk_data)
        logger.info(f"NLTK data {nltk_data} found locally, not downloading.")
    except LookupError:
        logger.info(f"NLTK data {nltk_data} not found... downloading it.")
        nltk.download(nltk_data)

# Initialize spacy model and lemmatizer
try:
    nlp_fr = spacy.load('fr_core_news_sm')
except OSError:
    spacy_download('fr_core_news_sm')
    nlp_fr = spacy.load('fr_core_news_sm')

try:
    nlp_de = spacy.load('de_core_news_sm')
except OSError:
    spacy_download('de_core_news_sm')
    nlp_de = spacy.load('de_core_news_sm')

lemmatizer = WordNetLemmatizer()


# Run this before in your cmd line:
# pip install gensim nltk spacy scikit-learn
# python -m spacy download fr_core_news_sm
# python -m spacy download de_core_news_sm

class TopicModellingService:
    default_model_path: str = './models'

    custom_stopwords: set = {" ", "\x96", "the", "to", "of", "20", "minuten"}

    @staticmethod
    def preprocess(corpus, language='german'):
        logger.info(f"Preprocessing {len(corpus)} texts for topic modelling.")

        stop_words = set(stopwords.words(language)) | TopicModellingService.custom_stopwords

        processed_texts = []
        processed_texts_bigrams = []
        processed_texts_trigrams = []
        nlp = nlp_de if language == 'german' else nlp_fr

        for doc in corpus:
            # Tokenize the document
            doc = nlp(str(doc).lower())  # Lowercase and tokenize
            tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and (token.pos_ == "NOUN" or token.pos_ == "PROPN" or token.pos_ == "PRON")]  # only nouns
            bigrams = list(ngrams(tokens, 2))
            trigrams = list(ngrams(tokens, 3))

            # Lemmatize words and remove stopwords
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
            lemmatized_bigrams_tokens = [lemmatizer.lemmatize(' '.join(token)) for token in bigrams]
            lemmatized_trigrams_tokens = [lemmatizer.lemmatize(' '.join(token)) for token in trigrams]

            processed_texts.append(lemmatized_tokens)
            processed_texts_bigrams.append(lemmatized_bigrams_tokens)
            processed_texts_trigrams.append(lemmatized_trigrams_tokens)

        return processed_texts, processed_texts_bigrams, processed_texts_trigrams

    @staticmethod
    def fit_lda(
            texts,
            language,
            num_topics=5,
            dataset_passes=5
    ):
        processed_texts, processed_texts_bigrams, processed_texts_trigrams = TopicModellingService.preprocess(texts, language)

        logger.info(f"Fitting LDA for {language}.")
        corpus = processed_texts + processed_texts_bigrams + processed_texts_trigrams
        dictionary = corpora.Dictionary(corpus)
        logger.info(f"Created dictionary with {len(dictionary)} entries.")

        corpus = [dictionary.doc2bow(text) for text in corpus]
        logger.info(f"Applying TFIDF to create a corpus, why?")

        logger.info(f"Fit online LDA with {dataset_passes}")
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=dataset_passes)

        return lda_model, corpus, dictionary

    @staticmethod
    def save_gensim_model(
            model: LdaModel,
            language: str | None = None,
            model_name: str | None = None,
            model_folder: str = default_model_path
    ) -> str:
        save_dir = Path(model_folder)
        save_dir.mkdir(parents=True, exist_ok=True)

        if not model_name:
            assert language, "Specify either model_name or language..."
            logger.info("No model_name defined, creating timestamped.")

            now = datetime.datetime.now()
            timestamp = now.strftime('%Y-%m-%d-%H-%M-%S')
            model_name = f"lda-model-{language}-{timestamp}-{model.num_topics}topics-{len(model.id2word)}dictsize"

        model_path = os.path.join(model_folder, model_name)

        Path(model_path).mkdir(parents=True, exist_ok=True)

        model.save(os.path.join(model_path, "model"))
        logger.info(f"Saved model to {model_path}")

        return model_path

    @staticmethod
    def load_gensim_model(
            language: str | None = None,
            full_model_path: str | None = None,
            default_model_folder: str | None = default_model_path
    ) -> LdaModel:

        if not full_model_path:
            assert language, f"If full_model_path is not set you must set the language."
            assert default_model_folder, f"If full_model_path is not set you must set the default_model_folder"

            all_models = glob.glob(os.path.join(default_model_folder, f"lda-model-{language}-*"), recursive=True)
            logger.info(f"Found {len(all_models)} models for {language}. Loading most recent one.")
            full_model_path = all_models[-1]

        assert not full_model_path.endswith("model"), "Provide the path to the folder, don't append 'model'."

        return LdaModel.load(os.path.join(full_model_path, "model"))

    @staticmethod
    def lda_top_words_per_topic(model, n_top_words):
        top_words_per_topic = []
        for topic_id in range(model.num_topics):
            top_words_per_topic.extend([(topic_id,) + x for x in model.show_topic(topic_id, topn=n_top_words)])
        return pd.DataFrame(top_words_per_topic, columns=["Topic", "Word", "Probability"])


if __name__ == '__main__':
    pass
