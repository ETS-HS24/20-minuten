import pandas as pd

from pipeline.models import Article


class FileService:

    @staticmethod
    def read_tsv_to_articles(file_path):
        # Read the TSV file into a DataFrame
        df = pd.read_csv(file_path, sep='\t')

        # Map DataFrame rows to Article instances
        articles = []
        for _, row in df.iterrows():
            article = Article(
                id=row['id'],
                pubtime=row['pubtime'],
                medium_code=row['medium_code'],
                medium_name=row['medium_name'],
                rubric=row['rubric'],
                regional=row['regional'],
                doctype=row['doctype'],
                doctype_description=row['doctype_description'],
                language=row['language'],
                char_count=row['char_count'],
                dateline=row['dateline'],
                head=row['head'],
                subhead=row['subhead'],
                article_link=row['article_link'],
                content_id=row['content_id'],
                content=row['content']
            )
            articles.append(article)

        return articles

    @staticmethod
    def read_tsv_to_df(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, sep='\t')

    @staticmethod
    def df_to_parquet(df: pd.DataFrame, file_name: str, output_dir: str = 'data/processed') -> None:
        file_path = f'{output_dir}/{file_name}.parquet'
        df.to_parquet(file_path)

    @staticmethod
    def read_parquet_to_df(file_name: str, file_dir: str = 'data/processed') -> pd.DataFrame:
        file_path = f'{file_dir}/{file_name}.parquet'
        return pd.read_parquet(file_path)