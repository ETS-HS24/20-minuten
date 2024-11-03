import glob
import os
import sys
import pandas as pd
import logging
from pathlib import Path
from pipeline.service import FileService, SentimentService, TextService, TopicModellingService, TopicMatcherService

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format='%(asctime)s | [%(filename)s:%(lineno)d] %(levelname)s | %(message)s')
    logger = logging.getLogger(__name__)

    force_recreate = False

    ######## Read File #########
    search_pattern = os.path.join('data', "**", "raw-data", "*.tsv")

    matching_files = glob.glob(search_pattern, recursive=True)
    file_path = matching_files[0]


    if not Path(FileService.get_parquet_path(file_name='articles_raw')).exists() or force_recreate: 
        logger.info(f"Recreating raw parquet.")
        articles_df = FileService.read_tsv_to_df(file_path)
        FileService.df_to_parquet(df=articles_df, file_name='articles_raw')
    else:
        logger.info(f"Not recreating raw transform")
        articles_df = FileService.read_parquet_to_df(file_name='articles_raw')

    if not Path(FileService.get_parquet_path(file_name='articles_sentiment')).exists() or force_recreate: 
        logger.info(f"Recreating sentiment analysis")
        ######### Pre-process Data #########
        cleaned_articles_df = TextService.dop_columns(df=articles_df, columns_to_drop=["rubric", "regional", "subhead"])
        cleaned_articles_df = TextService.process_tags(df=cleaned_articles_df)


        ############# Sentiment Analysis ############
        # Add sentiment to each article
        sentiment_df = SentimentService.sentimental_analysis(cleaned_articles_df)
        # Save the DataFrame as a parquet file
        FileService.df_to_parquet(sentiment_df, 'articles_sentiment')
    else:
        logger.info("Not recreating sentiment analysis")
        sentiment_df = FileService.read_parquet_to_df(file_name='articles_sentiment')


    ############# Topic Modelling #############
    number_of_articles = 500

    # French
    french_series = sentiment_df[sentiment_df['language'] == 'fr'].iloc[:number_of_articles]['content']
    french_model, _, _ = TopicModellingService.fit_lda(texts=french_series, language="french", dataset_passes=5)
    french_top_words_per_topic = TopicModellingService.lda_top_words_per_topic(model=french_model, n_top_words=10)
    model_path = TopicModellingService.save_gensim_model(model=french_model, language="french")

    FileService.df_to_csv(df=french_top_words_per_topic, file_name="topics_fr")

    df_topics_fr = french_top_words_per_topic

    # German
    german_series = sentiment_df[sentiment_df['language'] == 'de'].iloc[:number_of_articles]['content']
    german_model, _, _ = TopicModellingService.fit_lda(texts=german_series, language="german", dataset_passes=5)
    german_top_words_per_topic = TopicModellingService.lda_top_words_per_topic(model=german_model, n_top_words=10)
    model_path = TopicModellingService.save_gensim_model(model=german_model, language="german")

    FileService.df_to_csv(df=german_top_words_per_topic, file_name="topics_de")


    df_topics_de = german_top_words_per_topic

    # Matching topics German / French
    hit_list, corpus_embedding, top_k = TopicMatcherService.match(df_topics_de['Word'], df_topics_fr['Word'], print_matches=True)

    # print(hit_list)
    # print(corpus_embedding)
    # print(top_k)
