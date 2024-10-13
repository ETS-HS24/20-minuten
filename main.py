import glob
import os
from Swissdox.service import FileReader, SentimentAnalyzer

if __name__ == "__main__":
    search_pattern = os.path.join('data', "**", "raw-data", "*.tsv")

    mathing_files = glob.glob(search_pattern, recursive=True)

    all_articles = FileReader.read_tsv_to_articles(mathing_files[0])

    sentiment_analyzer = SentimentAnalyzer(all_articles)
    sentiment_analyzer.basic_analysis()
