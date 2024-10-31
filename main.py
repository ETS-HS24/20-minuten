import glob
import os
import pandas as pd
from pathlib import Path
from pipeline.service import FileService, SentimentService, TextService, TopicModellingService

if __name__ == "__main__":

    file = Path("data/processed/df.csv")

    if not file.exists():

        ######## Read FIle #########
        search_pattern = os.path.join('data', "**", "raw-data", "*.tsv")

        mathing_files = glob.glob(search_pattern, recursive=True)
        file_path = mathing_files[0]

        articles_df = FileService.read_tsv_to_df(file_path)
        FileService.df_to_parquet(articles_df, 'articles_raw')


        ######### Pre-process Data #########
        cleaned_articles_df = TextService.dop_columns(df=articles_df, columns_to_drop=["rubric", "regional", "subhead"])
        cleaned_articles_df = TextService.process_tags(df=cleaned_articles_df)


        ############# Sentiment Analysis ############
        # Add sentiment to each article
        sentiment_df = SentimentService.sentimental_analysis(cleaned_articles_df)
        # Save the DataFrame as a parquet file
        FileService.df_to_parquet(sentiment_df, 'articles_sentiment')

        sentiment_df = FileService.read_parquet_to_df('articles_sentiment')

        # Save dataframe as csv
        sentiment_df.to_csv(file, encoding='utf-8-sig')

    df = pd.read_csv(file)

    ############# Topic Modelling #############
    number_of_articles = 10

    # French
    df_fr = df[df['language'] == 'fr']['content'].iloc[:number_of_articles].reset_index()
    df_list = []
    for index, row in df_fr.iterrows():
        topic_df_fr, lda_model_fr, corpus_fr, dictionary_fr = TopicModellingService.topic_modeling(row, 'french', num_topics=5, num_words=5, print_topics=False, export_to_csv=False)
        df_list.append(topic_df_fr)
    df_print = pd.concat(df_list, axis=0)
    TopicModellingService.to_csv(df_print, "data/topics/topics_fr")

    df_topics_fr = pd.read_csv("data/topics/topics_fr.csv")
    print(df_topics_fr)

    # German
    df_de = df[df['language'] == 'de']['content'].iloc[:number_of_articles].reset_index()
    df_list = []
    for index, row in df_de.iterrows():
        topic_df_de, lda_model_de, corpus_de, dictionary_de = TopicModellingService.topic_modeling(row, 'german', num_topics=5, num_words=5, print_topics=False, export_to_csv=False)
        df_list.append(topic_df_de)
    df_print = pd.concat(df_list, axis=0)
    TopicModellingService.to_csv(df_print, "data/topics/topics_de")

    df_topics_de = pd.read_csv("data/topics/topics_de.csv")
    print(df_topics_de)
