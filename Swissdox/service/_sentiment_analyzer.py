from collections import Counter

from textblob import TextBlob

from Swissdox.models import Article


class SentimentAnalyzer:
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