import pandas as pd
from pathlib import Path

from pipeline.models import Article


class FileService:
    default_processed_path: str = './data/processed'

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
    def df_to_parquet(df: pd.DataFrame, file_name: str, output_dir: str = default_processed_path) -> None:
        file_path = FileService.get_parquet_path(file_name=file_name, output_dir=output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df.to_parquet(file_path)

    @staticmethod
    def get_parquet_path(file_name: str, output_dir: str = default_processed_path) -> str:
        return f'{output_dir}/{file_name}.parquet'

    @staticmethod
    def df_to_csv(df: pd.DataFrame, file_name: str, output_dir: str = default_processed_path) -> None:
        file_path = FileService.get_csv_path(file_name=file_name, output_dir=output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path)

    @staticmethod
    def get_csv_path(file_name: str, output_dir: str = default_processed_path) -> str:
        return f'{output_dir}/{file_name}.csv'

    @staticmethod
    def read_parquet_to_df(file_name: str, file_dir: str = default_processed_path) -> pd.DataFrame:
        file_path = f'{file_dir}/{file_name}.parquet'
        return pd.read_parquet(file_path)

    @staticmethod
    def read_csv_to_df(file_name: str, file_dir: str = default_processed_path, sep: str = ",") -> pd.DataFrame:
        file_path = f'{file_dir}/{file_name}.csv'
        return pd.read_csv(file_path, sep=sep)
