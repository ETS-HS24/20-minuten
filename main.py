import glob
import os
from pipeline.service import FileService, SentimentService, TextService

if __name__ == "__main__":
    search_pattern = os.path.join('data', "**", "raw-data", "*.tsv")

    mathing_files = glob.glob(search_pattern, recursive=True)
    file_path = mathing_files[0]

    # Read the TSV file into a DataFrame
    articles_df = FileService.read_tsv_to_df(file_path)
    # Save the DataFrame as a parquet file
    FileService.df_to_parquet(articles_df, 'articles_raw')

    # Remove tags from the content column
    cleaned_articles_df = TextService.remove_tags(articles_df)
    # Add sentiment to each article
    sentiment_df = SentimentService.sentimental_analysis(cleaned_articles_df)
    # Save the DataFrame as a parquet file
    FileService.df_to_parquet(sentiment_df, 'articles_sentiment')

    sentiment_df = FileService.read_parquet_to_df('articles_sentiment')
    a=10

