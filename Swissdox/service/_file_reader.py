import pandas as pd

from Swissdox.models import Article


class FileReader:

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