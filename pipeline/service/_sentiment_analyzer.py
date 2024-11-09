from collections import Counter

import pandas as pd
from textblob import TextBlob
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, CamembertForSequenceClassification
from germansentiment import SentimentModel

from pipeline.models import Article


class SentimentService:
    _languages: set[str]
    _articles: list[Article]

    def __init__(self, articles: list[Article]):
        self._articles = articles
        self._languages = {article.language for article in articles}


    def basic_analysis(self):
        language_map = {lang: [] for lang in self._languages}

        for article in self._articles:
            language_map[article.language].append(article)

        sentiment_counts = {lang: Counter() for lang in self._languages}

        for article in self._articles:
            analysis = TextBlob(article.content)
            article.sentiment = analysis.sentiment
            sentiment_label = "positive" if analysis.sentiment.polarity > 0 else "negative" if analysis.sentiment.polarity < 0 else "neutral"
            sentiment_counts[article.language][sentiment_label] += 1

        for lang in self._languages:
            total_articles = len(language_map[lang])
            print(f"{lang.upper()} Articles Sentiment Counts:")
            for sentiment, count in sentiment_counts[lang].items():
                percentage = (count / total_articles) * 100 if total_articles > 0 else 0
                print(f"{sentiment.capitalize()} Articles: {count} ({percentage:.2f}%)")

    @staticmethod
    def sentimental_analysis(data_frame: pd.DataFrame, column_to_process: str = "content") -> pd.DataFrame:

        french_tokenizer = AutoTokenizer.from_pretrained("camembert-base")
        french_model = CamembertForSequenceClassification.from_pretrained("camembert-base")

        french_pipeline = pipeline('sentiment-analysis', model=french_model, tokenizer=french_tokenizer, device=0)

        german_model = SentimentModel()

        def process_content(content, lang):
            if lang == "fr":
                # Truncate the content to the maximum length allowed by the model
                inputs = french_tokenizer(content, truncation=True, max_length=512, return_tensors="pt")
                result = french_pipeline(content[:512])[0]  # Pass the truncated content directly
                label = result['label']
                score = round(result['score'], 3)

                # Map the labels to negative/neutral/positive
                if label == 'LABEL_0':
                    sentiment = 'negative'
                elif label == 'LABEL_1':
                    sentiment = 'positive'
                else:
                    sentiment = 'neutral'
            else:
                prediction = german_model.predict_sentiment([content], output_probabilities=True)
                sentiment = prediction[0][0]
                score = round(float([prediction_score for prediction_score in prediction[1][0] if prediction_score[0] == sentiment][0][1]), 3)

            return sentiment, score

        tqdm.pandas()
        # pass content and language to the function
        data_frame[['sentiment', 'score']] = data_frame.progress_apply(
            lambda x: pd.Series(process_content(x[column_to_process], x['language'])), axis=1)

        data_frame['sentiment'] = data_frame['sentiment'].astype(str)
        return data_frame
