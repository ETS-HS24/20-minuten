import glob
import os
import pandas as pd
import logging
from pathlib import Path
from pipeline.service import FileService, SentimentService, TextService, TopicModellingService, TopicMatcherService

if __name__ == "__main__":
    
    force_recreate = False

    ######## Read File #########
    search_pattern = os.path.join('data', "**", "raw-data", "*.tsv")

    matching_files = glob.glob(search_pattern, recursive=True)
    file_path = matching_files[0]


    if not Path(FileService.get_parquet_path(file_name='articles_raw')).exists() or force_recreate: 
        logging.info(f"Recreating raw parquet.")
        articles_df = FileService.read_tsv_to_df(file_path)
        FileService.df_to_parquet(df=articles_df, file_name='articles_raw')
    else:
        logging.info(f"Not recreating raw transform")
        articles_df = FileService.read_parquet_to_df(file_name='articles_raw')

    if not Path(FileService.get_parquet_path(file_name='articles_sentiment')).exists() or force_recreate: 
        logging.info(f"Recreating sentiment analysis")
        ######### Pre-process Data #########
        cleaned_articles_df = TextService.dop_columns(df=articles_df, columns_to_drop=["rubric", "regional", "subhead"])
        cleaned_articles_df = TextService.process_tags(df=cleaned_articles_df)


        ############# Sentiment Analysis ############
        # Add sentiment to each article
        sentiment_df = SentimentService.sentimental_analysis(cleaned_articles_df)
        # Save the DataFrame as a parquet file
        FileService.df_to_parquet(sentiment_df, 'articles_sentiment')
    else:
        logging.info("Not recreating sentiment analysis")
        sentiment_df = FileService.read_parquet_to_df(file_name='articles_sentiment')


    ############# Topic Modelling #############
    number_of_articles = 10

    # French
    df_fr = sentiment_df[sentiment_df['language'] == 'fr']['content'].iloc[:number_of_articles].reset_index()
    df_list = []
    for index, row in df_fr.iterrows():
        topic_df_fr, lda_model_fr, corpus_fr, dictionary_fr = TopicModellingService.topic_modeling(row, 'french', num_topics=10, num_words=5, print_topics=False)
        df_list.append(topic_df_fr)
    df_print = pd.concat(df_list, axis=0)
    FileService.df_to_csv(df=df_print, file_name="topics_fr")

    df_topics_fr = topic_df_fr

    # German
    df_de = sentiment_df[sentiment_df['language'] == 'de']['content'].iloc[:number_of_articles].reset_index()
    df_list = []
    for index, row in df_de.iterrows():
        topic_df_de, lda_model_de, corpus_de, dictionary_de = TopicModellingService.topic_modeling(row, 'german', num_topics=10, num_words=5, print_topics=False)
        df_list.append(topic_df_de)
    df_print = pd.concat(df_list, axis=0)
    FileService.df_to_csv(df=df_print, file_name="topics_de")

    df_topics_de = topic_df_de

    # Matching topics German / French
    hit_list, corpus_embedding, top_k = TopicMatcherService.match(df_topics_de['Word'], df_topics_fr['Word'], print_matches=True)

    # print(hit_list)
