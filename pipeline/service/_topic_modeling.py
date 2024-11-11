import pandas as pd
import logging
import nltk
import spacy
from spacy.cli.download import download as spacy_download
from gensim import corpora
from gensim.models import LdaModel, LsiModel
from nltk.stem import WordNetLemmatizer
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
        nltk.download(nltk_data, quiet=True)

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

class TopicModelingService:
    default_model_path: str = './models'

    @staticmethod
    def preprocess(corpus, language='german'):
        logger.info(f"Preprocessing {len(corpus)} texts for topic modeling.")
        pos_to_remove = ["ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NUM", "PART", "PRON", "PUNCT", "SCONJ", "SYM"]
        nlp = nlp_de if language == 'german' else nlp_fr
        articles = nlp.pipe(corpus, disable=["tagger", "ner", "textcat"], n_process=4)
        tokenized_articles = []
        for article in articles:
            article_tokens = []
            for token in article:
                if (
                    token.pos_ not in pos_to_remove  # Remove defined parts of speech
                    and not token.is_stop  # Token is not a stopword
                    and not token.is_space
                    and not token.is_punct
                ):
                    article_tokens.append(token.lemma_.lower())
            tokenized_articles.append(article_tokens)
        return tokenized_articles

    @staticmethod
    def fit_model(
            texts,
            language,
            num_topics=5,
            dataset_passes=5,
            technique: str = 'lda',
    ):
        valid_models = ["lda", "lsa"]
        assert technique in valid_models, f"Please provide a valid model from: {valid_models}"
        processed_texts = TopicModelingService.preprocess(texts, language)
        logger.info(f"Fitting {technique.upper()} for {language}.")
        dictionary = corpora.Dictionary(processed_texts)
        logger.info(f"Created dictionary with {len(dictionary)} entries.")
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        logger.info(f"Applying TFIDF to create a corpus, why?")
        logger.info(f"Fit online {technique.upper()} with {dataset_passes}")
        if technique == 'lda':
            model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=dataset_passes)
        else:
            model = LsiModel(corpus, num_topics=num_topics, id2word=dictionary)
        return model, corpus, dictionary

    @staticmethod
    def save_gensim_model(
            model: LdaModel | LsiModel,
            technique: str = 'lda',
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
            model_name = f"{technique}-model-{language}-{timestamp}-{model.num_topics}topics-{len(model.id2word)}dictsize"

        model_path = os.path.join(model_folder, model_name)

        Path(model_path).mkdir(parents=True, exist_ok=True)

        model.save(os.path.join(model_path, "model"))
        logger.info(f"Saved model to {model_path}")

        return model_path

    @staticmethod
    def load_gensim_model(
            language: str | None = None,
            technique: str = 'lda',
            full_model_path: str | None = None,
            default_model_folder: str | None = default_model_path
    ) -> LdaModel | LsiModel:
        if not full_model_path:
            assert language, f"If full_model_path is not set you must set the language."
            assert default_model_folder, f"If full_model_path is not set you must set the default_model_folder"

            all_models = glob.glob(os.path.join(default_model_folder, f"{technique}-model-{language}-*"), recursive=True)
            logger.info(f"Found {len(all_models)} models for {language}. Loading most recent one.")
            full_model_path = all_models[-1]

        assert not full_model_path.endswith("model"), "Provide the path to the folder, don't append 'model'."
        model_name = os.path.join(full_model_path, "model")
        if technique == 'lda':
            return LdaModel.load(model_name)
        elif technique == 'lsa':
            return LsiModel.load(model_name)

    @staticmethod
    def get_top_words_per_topic(model, n_top_words, technique: str = 'lda'):
        top_words_per_topic = []
        for topic_id in range(model.num_topics):
            top_words_per_topic.extend([(topic_id,) + x for x in model.show_topic(topic_id, topn=n_top_words)])
        return pd.DataFrame(top_words_per_topic, columns=["topic", "word", "probability" if technique == 'lda' else "weight"])


if __name__ == '__main__':
    pass
