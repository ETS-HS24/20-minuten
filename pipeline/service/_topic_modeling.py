from types import SimpleNamespace
import pandas as pd
import logging
import nltk
import spacy
import torch
from spacy.cli.download import download as spacy_download
from gensim import corpora
from gensim.models import LdaModel, LsiModel
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import os
import datetime
import glob
from tqdm import tqdm
from typing import List, Type
from sentence_transformers import SentenceTransformer
from top2vec import Top2Vec
import numpy as np

tqdm.pandas()


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
    
    @staticmethod
    def fit_top2vec_model(
            data_column:pd.Series|list,
            transformer_name:str|os.PathLike = "sentence-transformers/LaBSE",
            umap_args:dict|None=None,
            hdbscan_args:dict|None=None,
        ) -> Top2Vec:
        logging.info(f"Fitting top2vec model on a corpus of {len(data_column)} documents.")
        assert len(data_column) > 0, "Empty data provided. Please provide samples in a list or Pandas data frame."

        original_env = os.environ.copy()
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        logging.debug(f"Loading model: {transformer_name}")
        pretrained_model = SentenceTransformer(str(transformer_name))

        logging.debug(f"Fitting Top2Vec pipeline to {len(data_column)} samples.")
        model = Top2Vec(
            documents=list(data_column), 
            embedding_model=pretrained_model.encode, 
            chunk_length=pretrained_model.max_seq_length, 
            chunk_overlap_ratio=.2,
            gpu_umap=False,
            umap_args=umap_args,
            gpu_hdbscan=False,
            hdbscan_args=hdbscan_args,
            speed="learn", 
            workers=4,
            verbose=True
        )
        logging.info(f"Model found {model.get_num_topics()} topics.")
        logging.debug(f"Finished fitting Top2Vec model, thanks for waiting.")

        # Reset original environment
        os.environ = original_env
        return model

    @staticmethod
    def save_top2vec_model(
        model:Top2Vec,
        model_save_path:str|os.PathLike|None,
        force_overwrite:bool=False
    ) -> Path:
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y-%m-%d-%H-%M-%S')

        if not model_save_path:
            logging.info("No model name given, generating default name.")
            model_save_path = Path(os.path.join(TopicModelingService.default_model_path, "top2vec", f"top2vec-{model.get_num_topics()}topics-{timestamp}"))
        
        save_path = Path(os.path.normpath(model_save_path))
        if not save_path.parent.exists():
            logging.info(f"Path for {save_path} does not exist, creating.")
            save_path.mkdir(parents=True, exist_ok=True)

        if not force_overwrite and save_path.exists():
            logging.warning(f"Model already exists, appending timestamp to modelname")
            new_path = save_path.with_stem(f"{save_path.stem}-{timestamp}")
            save_path.rename(new_path)

        model.save(str(save_path))
        return save_path

    @staticmethod
    def save_top2vec_embeddings(
        model:Top2Vec,
        embeddings_save_path:str|os.PathLike|None,
        force_overwrite:bool=False
    ) -> Path:
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y-%m-%d-%H-%M-%S')

        embeddings = model.document_vectors

        if not embeddings_save_path:
            embeddings_save_path = Path(os.path.join(TopicModelingService.default_model_path, "vectorspaces", f"top2vec-embeddings-{model.get_num_topics()}topics-{len(embeddings)}documents-{timestamp}.npy"))
        
        save_path = Path(os.path.normpath(embeddings_save_path))
        if not save_path.parent.exists():
            logging.info(f"Path for {save_path} does not exist, creating.")
            save_path.mkdir(parents=True, exist_ok=True)

        if not force_overwrite and save_path.exists():
            logging.warning(f"Embedding already exists, appending timestamp to modelname")
            new_path = save_path.with_suffix(timestamp)
            save_path.rename(new_path)

        np.save(save_path, embeddings)
        model.save(save_path)
        return save_path
    
    @staticmethod
    def load_top2vec_model(
        model_path:str|os.PathLike,
        transformer_name:str|os.PathLike = "sentence-transformers/LaBSE",
    ) -> Top2Vec:
        load_path = Path(model_path)
        assert load_path.exists(), "The model path does not exist."

        logging.info(f"Loading {transformer_name} and top2vec from {model_path}.")
        if torch.backends.mps.is_available(): # apple chip
            device = "mps"
        elif torch.cuda.is_available(): # gpu
            device = "cuda"
        else: # cpu
            device = "cpu"
        pretrained_model = SentenceTransformer(str(transformer_name), device=device)
        model = Top2Vec.load(str(load_path))
        model.set_embedding_model(pretrained_model.encode)
        
        # Mark model as loaded from disk
        model._loaded_from_disk = SimpleNamespace()
        model._loaded_from_disk = True

        return model


    @staticmethod
    def predict_or_get_top2vec_topics(
        model:Top2Vec,
        series:pd.Series,
        num_topics:int=1
    ) -> pd.Series:
        
        def get_topics(article:str) -> List[int]:
            topic_words, word_scores, topic_scores, topic_nums = model.query_topics(query=article, num_topics=num_topics)
            return topic_nums.tolist()
        
        if hasattr(model, "_loaded_from_disk"):
            if model._loaded_from_disk:
                model_loaded = True
            else:
                model_loaded = False
        else:
            model_loaded = False

        if model_loaded:
            # Very slow but guarantees that there is no data/prediction mismatch.
            logging.warning(f"Calculating topics because it is indicated that the model has been loaded from disk.")
            logging.warning(f"This will take some time to calculate... If you know what you are doing you can set `model._loaded_from_disk = False`")
            logging.warning(f"This might risk a document/topic row missmatch, I hope you know what you are doing.")
            topics = series.progress_apply(lambda row: get_topics(row))
        else:
            logging.warning(f"Loading topics directly from the model. Did you make sure that you are attaching the generated topics to the right dataset?")
            topic_nums, topic_score, topics_words, word_scores = model.get_documents_topics(model.document_ids, num_topics=2)
            topics = pd.Series(topic_nums.tolist(), index=series.index)
            print(f"Returning topics {np.count_nonzero(~np.isnan(topic_nums))} for {len(series)}.")

        return topics

if __name__ == '__main__':
    pass
