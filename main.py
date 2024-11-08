import glob
import os
import sys
import pandas as pd
import logging
from pathlib import Path
from pipeline.service import FileService, SentimentService, TextService, TopicModelingService, TopicMatcherService

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

    ############# Topic Modeling #############
    number_of_articles = 1000
    number_of_topics = 100
    number_of_top_words = 10
    ds_passes = 2
    match_score = 0.9

    # Data
    french_series = sentiment_df[sentiment_df['language'] == 'fr'].iloc[:number_of_articles]['content']
    german_series = sentiment_df[sentiment_df['language'] == 'de'].iloc[:number_of_articles]['content']

    ### LDA ###
    # French
    french_lda_model, _, _ = TopicModelingService.fit_model(texts=french_series, language="french", num_topics=number_of_topics, dataset_passes=ds_passes, technique='lda')
    df_lda_topics_fr = TopicModelingService.get_top_words_per_topic(model=french_lda_model, n_top_words=number_of_top_words, technique='lda')
    FileService.df_to_csv(df=df_lda_topics_fr, file_name="lda_topics_fr")

    # German
    german_lda_model, _, _ = TopicModelingService.fit_model(texts=german_series, language="german", num_topics=number_of_topics, dataset_passes=ds_passes, technique='lda')
    df_lda_topics_de = TopicModelingService.get_top_words_per_topic(model=german_lda_model, n_top_words=number_of_top_words, technique='lda')
    FileService.df_to_csv(df=df_lda_topics_de, file_name="lda_topics_de")

    # Matching topics Query: French / Corpus: German
    best_lda_matches_fr_de, _, _, _ = TopicMatcherService.match(df_lda_topics_de['Word'], df_lda_topics_fr['Word'], number_of_top=number_of_top_words, match_score=match_score, print_matches=True)

    # Matching topics Query: German / Corpus: French
    best_lda_matches_de_fr, _, _, _ = TopicMatcherService.match(df_lda_topics_fr['Word'], df_lda_topics_de['Word'], number_of_top=number_of_top_words, match_score=match_score, print_matches=True, invert=True)

    best_lda_matches = pd.concat([best_lda_matches_de_fr, best_lda_matches_fr_de])
    FileService.df_to_csv(df=best_lda_matches, file_name="best_lda_matches")

    # Matching by translation
    lda_matches_by_translation, lda_german_counts, lda_french_counts = TopicMatcherService.match_by_translation(df_lda_topics_de['Word'], df_lda_topics_fr['Word'])
    FileService.df_to_csv(df=lda_matches_by_translation, file_name="lda_matches_by_translation")

    print(lda_matches_by_translation)
    print(lda_german_counts)
    print(lda_french_counts)


    ### LSA ###
    # French
    french_lsa_model, _, _ = TopicModelingService.fit_model(texts=french_series, language="french", num_topics=number_of_topics, dataset_passes=ds_passes, technique='lsa')
    df_lsa_topics_fr = TopicModelingService.get_top_words_per_topic(model=french_lsa_model, n_top_words=number_of_top_words, technique='lsa')
    FileService.df_to_csv(df=df_lsa_topics_fr, file_name="lsa_topics_fr")

    # German
    german_lsa_model, _, _ = TopicModelingService.fit_model(texts=german_series, language="german", num_topics=number_of_topics, dataset_passes=ds_passes, technique='lsa')
    df_lsa_topics_de = TopicModelingService.get_top_words_per_topic(model=german_lsa_model, n_top_words=number_of_top_words, technique='lsa')
    FileService.df_to_csv(df=df_lsa_topics_de, file_name="lsa_topics_de")

    # Matching topics Query: French / Corpus: German
    best_lsa_matches_fr_de, _, _, _ = TopicMatcherService.match(df_lsa_topics_de['Word'], df_lsa_topics_fr['Word'], number_of_top=number_of_top_words, match_score=match_score, print_matches=True)

    # Matching topics Query: German / Corpus: French
    best_lsa_matches_de_fr, _, _, _ = TopicMatcherService.match(df_lsa_topics_fr['Word'], df_lsa_topics_de['Word'], number_of_top=number_of_top_words, match_score=match_score, print_matches=True, invert=True)

    best_lsa_matches = pd.concat([best_lsa_matches_de_fr, best_lsa_matches_fr_de])
    FileService.df_to_csv(df=best_lsa_matches, file_name="best_lsa_matches")

    # Matching by translation
    lsa_matches_by_translation, lsa_german_counts, lsa_french_counts = TopicMatcherService.match_by_translation(df_lsa_topics_de['Word'], df_lsa_topics_fr['Word'])
    FileService.df_to_csv(df=lsa_matches_by_translation, file_name="lsa_matches_by_translation")

    print(lsa_matches_by_translation)
    print(lsa_german_counts)
    print(lsa_french_counts)
