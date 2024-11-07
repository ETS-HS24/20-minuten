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
    _previous_step_recreate = False

    ######## Read File #########
    search_pattern = os.path.join('data', "**", "raw-data", "*.tsv")

    matching_files = glob.glob(search_pattern, recursive=True)

    if matching_files:
        file_path = matching_files[0]
    else:
        file_path = FileService.default_processed_path + '/articles_raw.tsv'

    if (
            not Path(FileService.get_parquet_path(file_name='articles_raw')).exists()
            or force_recreate
            or _previous_step_recreate
    ):
        logger.info(f"Recreating raw parquet.")
        articles_df = FileService.read_tsv_to_df(file_path)
        FileService.df_to_parquet(df=articles_df, file_name='articles_raw')
        _previous_step_recreate = True
    else:
        logger.info(f"Not recreating raw transform.")
        articles_df = FileService.read_parquet_to_df(file_name='articles_raw')

    if (
            not Path(FileService.get_parquet_path(file_name='articles_cleaned')).exists()
            or force_recreate
            or _previous_step_recreate
    ):
        logger.info("Recreating cleaned dataset.")
        ######### Pre-process Data #########
        cleaned_articles_df = TextService.drop_columns(df=articles_df,
                                                       columns_to_drop=["rubric", "regional", "subhead"])
        cleaned_articles_df = TextService.process_tags(df=cleaned_articles_df)
        FileService.df_to_parquet(cleaned_articles_df, 'articles_cleaned')
        _previous_step_recreate = True

    else:
        logger.info("Not recreating cleaned dataset.")
        cleaned_articles_df = FileService.read_parquet_to_df(file_name='articles_cleaned')

    if (
            not Path(FileService.get_parquet_path(file_name='articles_lemmatized')).exists()
            or force_recreate
            or _previous_step_recreate
    ):
        logger.info("Recreating lemmatized dataset.")
        ########## Lemmatize Data #########
        lemmatized_articles_df = TextService.lemmatize_content_nltk(df=cleaned_articles_df, column_to_process="content")
        FileService.df_to_parquet(lemmatized_articles_df, 'articles_lemmatized')
        _previous_step_recreate = True
    else:
        logger.info("Not recreating lemmatized dataset.")
        lemmatized_articles_df = FileService.read_parquet_to_df(file_name='articles_lemmatized')

    if (
            not Path(FileService.get_parquet_path(file_name='articles_sentiment')).exists()
            or force_recreate
            or _previous_step_recreate
    ):
        logger.info(f"Recreating sentiment analysis")
        ############# Sentiment Analysis ############
        # Add sentiment to each article
        sentiment_df = SentimentService.sentimental_analysis(data_frame=lemmatized_articles_df,
                                                             column_to_process="content_lemmatized")
        # Save the DataFrame as a parquet file
        FileService.df_to_parquet(sentiment_df, 'articles_sentiment')
        _previous_step_recreate = True
    else:
        logger.info("Not recreating sentiment analysis.")
        sentiment_df = FileService.read_parquet_to_df(file_name='articles_sentiment')

    #sys.exit()
    ############# Topic Modelling #############
    number_of_articles = 1000
    number_of_topics = 100
    number_of_top_words = 10
    ds_passes = 2
    match_score = 0.9

    # French
    french_series = sentiment_df[sentiment_df['language'] == 'fr'].iloc[:number_of_articles]['content']
    #french_series = sentiment_df[sentiment_df['language'] == 'fr']['content']
    french_model, _, _ = TopicModellingService.fit_lda(texts=french_series, language="french", num_topics=number_of_topics, dataset_passes=ds_passes)
    french_top_words_per_topic = TopicModellingService.lda_top_words_per_topic(model=french_model, n_top_words=number_of_top_words)
    #model_path = TopicModellingService.save_gensim_model(model=french_model, language="french")
    #model1 = TopicModellingService.load_gensim_model(language="french")
    #model2 = TopicModellingService.load_gensim_model(full_model_path=model_path)

    FileService.df_to_csv(df=french_top_words_per_topic, file_name="topics_fr")

    df_topics_fr = french_top_words_per_topic
    print(df_topics_fr)

    # German
    german_series = sentiment_df[sentiment_df['language'] == 'de'].iloc[:number_of_articles]['content']
    #german_series = sentiment_df[sentiment_df['language'] == 'de']['content']
    german_model, _, _ = TopicModellingService.fit_lda(texts=german_series, language="german", num_topics=number_of_topics, dataset_passes=ds_passes)
    german_top_words_per_topic = TopicModellingService.lda_top_words_per_topic(model=german_model, n_top_words=number_of_top_words)
    #model_path = TopicModellingService.save_gensim_model(model=german_model, language="german")

    FileService.df_to_csv(df=german_top_words_per_topic, file_name="topics_de")

    df_topics_de = german_top_words_per_topic
    print(df_topics_de)

    # Matching topics Query: French / Corpus: German
    best_matches_fr_de, _, _, _ = TopicMatcherService.match(df_topics_de['Word'], df_topics_fr['Word'], number_of_top=number_of_top_words, match_score=match_score, print_matches=True)

    # Matching topics Query: German / Corpus: French
    best_matches_de_fr, hit_list, corpus_embedding, top_k = TopicMatcherService.match(df_topics_fr['Word'], df_topics_de['Word'], number_of_top=number_of_top_words, match_score=match_score, print_matches=True, invert=True)

    print(pd.concat([best_matches_de_fr, best_matches_fr_de]).drop_duplicates())
    # print(hit_list)
    # print(corpus_embedding)
    # print(top_k)
