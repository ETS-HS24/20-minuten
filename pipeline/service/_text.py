import os

import pandas as pd
import spacy
from bs4 import BeautifulSoup
from typing import List
import re
import unicodedata
from tqdm import tqdm

from nltk.corpus.europarl_raw import german
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.data import find


class TextService:
    nlp_de = spacy.load("de_core_news_sm")
    nlp_fr = spacy.load("fr_core_news_sm")
    nltk_data_downloaded = False

    @staticmethod
    def download_nltk_data():
        if not TextService.nltk_data_downloaded:
            try:
                find('corpora/stopwords.zip')
            except LookupError:
                import nltk
                nltk.download('stopwords')

            try:
                find('corpora/wordnet.zip')
            except LookupError:
                import nltk
                nltk.download('wordnet')

            TextService.nltk_data_downloaded = True

    @staticmethod
    def drop_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
        return df.drop(columns_to_drop, axis=1)

    @staticmethod
    def remove_all_tags(df: pd.DataFrame, columns_to_clean=None) -> pd.DataFrame:
        if columns_to_clean is None:
            columns_to_clean = ["content"]
        for column in columns_to_clean:
            df[column] = df[column].str.replace(r'<[^>]*>', '', regex=True)
        return df

    @staticmethod
    def get_authors(article: str) -> List[str] | None:
        soup = BeautifulSoup(article, 'html.parser')
        if soup.p:
            last_p = soup.find_all("p")[-1].text
            if last_p[0] == "(" and last_p[-1] == ")":
                inner_last_p = last_p[1:-1]
                if "," in inner_last_p:
                    return_value = inner_last_p.split(",")
                elif "/" in inner_last_p:
                    return_value = inner_last_p.split("/")
                else:
                    return_value = [inner_last_p]

                # sanity check, there should hopefully not be a broken up string
                if len(return_value) > 3:
                    print(return_value)

                return return_value
            else:
                return None
        else:
            return None

    @staticmethod
    def remove_author_from_string_end(article: str, n_characters: int = 10) -> str:
        if len(article) <= n_characters:
            return article

        last_n_chars = article[n_characters:]
        cleaned = re.sub(r'\(.*?\)|', '', last_n_chars)
        return article[:n_characters] + cleaned

    @staticmethod
    def process_tags(df: pd.DataFrame) -> pd.DataFrame:
        # Extract and remove lead text
        df['lead_text'] = df['content'].str.extract(r'<ld>(.*?)</ld>', expand=False).str.replace(r'</?p[^>]*>', ' ',
                                                                                                 regex=True)
        df['content'] = df['content'].str.replace(r'<ld>.*?</ld>', ' ', regex=True)

        # Extract and remove subheadings
        df['subheadings'] = df['content'].str.extract(r'<zt>(.*?)</zt>', expand=False)
        df['content'] = df['content'].str.replace(r'<zt>.*?</zt>', ' ', regex=True)

        # Extract and remove author full name
        df['author'] = df['content'].str.extract(r'<au>(.*?)</au>', expand=False)
        df['content'] = df['content'].str.replace(r'<au>.*?</au>', ' ', regex=True)

        # Remove tags but keep annotated text within <a> tags
        df['content'] = df['content'].str.replace(r'<a[^>]*>(.*?)</a>', r'\1', regex=True)

        # Remove <tx>, <p>, <br>, <ka>, and <lg> tags
        df['content'] = df['content'].str.replace(r'</?(tx|p|br|ka|lg)[^>]*>', ' ', regex=True)

        # Extract authors from the last <p> element if it matches the criteria
        df['author_extracted'] = df['content'].apply(TextService.get_authors)

        # Remove the author from the text
        df['content'] = df['content'].apply(TextService.remove_author_from_string_end)

        # Remove all p tags
        df['content'] = df['content'].str.replace(r'<p>.*?</p>$', ' ', regex=True)

        # Remove all double spaces
        df['content'] = df['content'].str.replace(r'\s+', ' ', regex=True)

        # Remove non printable control characters
        df['content'] = df['content'].apply(
            lambda row: ''.join(char for char in row if not unicodedata.category(char).startswith("C")))

        # Remove characters from text
        df['content'] = df['content'].str.replace(r'[«»]', '', regex=True)

        # Replace characters with empty string 
        df['content'] = df['content'].str.replace(r'[-/|#]', ' ', regex=True)

        return df

    @staticmethod
    def lemmatize_content_nltk(df: pd.DataFrame) -> pd.DataFrame:
        TextService.download_nltk_data()
        lemmatizer = WordNetLemmatizer()

        custom_stop_words: set = {
            " ", "\x96", "the", "to", "of", "20", "minuten",
        }

        with open(os.path.normpath("./analysis/german_stopwords_full.txt"), "r") as file:
            german_stop_words_full = file.read().splitlines()

        with open(os.path.normpath("./analysis/french_stopwords_full.txt"), "r") as file:
            french_stop_words_full = file.read().splitlines()

        german_stop_words = set(stopwords.words("german")) | set(german_stop_words_full) | custom_stop_words
        french_stop_words = set(stopwords.words("french")) | set(french_stop_words_full) | custom_stop_words

        def lemmatize_text(doc: str, lang: str):
            if lang == "fr":
                doc = TextService.nlp_fr(str(doc).lower())
                tokenized_article = [token.text for token in doc if not token.is_stop and not token.is_punct]
                lemmatized_article = [lemmatizer.lemmatize(token) for token in tokenized_article if
                                      token not in french_stop_words]

            else:
                doc = TextService.nlp_de(str(doc).lower())
                tokenized_article = [token.text for token in doc if not token.is_stop and not token.is_punct]
                lemmatized_article = [lemmatizer.lemmatize(token) for token in tokenized_article if
                                      token not in german_stop_words]

            return ' '.join(lemmatized_article)

        tqdm.pandas()
        df["content_lemmatized"] = df.progress_apply(
            lambda x: pd.Series(lemmatize_text(x["content"], x["language"])), axis=1
        )
        return df

    @staticmethod
    def lemmatize_content_spacy(df: pd.DataFrame) -> pd.DataFrame:

        def lemmatize_text(doc: str, lang: str):
            if lang == "fr":
                docs = TextService.nlp_fr.pipe([doc], disable=["tagger", "ner", "textcat"], n_process=4)
            else:
                docs = TextService.nlp_de.pipe([doc], disable=["tagger", "ner", "textcat"], n_process=4)

            alphas = [token.lemma_.lower() for doc in docs for token in doc if
                      not token.is_alpha and not token.is_punct and not token.is_space]
            return ' '.join(alphas)

        tqdm.pandas()
        df["content_lemmatized"] = df.progress_apply(
            lambda x: lemmatize_text(x["content"], x["language"]), axis=1
        )
        return df
