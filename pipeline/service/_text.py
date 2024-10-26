import pandas as pd


class TextService:

    @staticmethod
    def dop_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
        return df.drop(columns_to_drop, axis=1)


    @staticmethod
    def remove_all_tags(df: pd.DataFrame, columns_to_clean=None) -> pd.DataFrame:
        if columns_to_clean is None:
            columns_to_clean = ["content"]
        for column in columns_to_clean:
            df[column] = df[column].str.replace(r'<[^>]*>', '', regex=True)
        return df

    @staticmethod
    def process_tags(df: pd.DataFrame) -> pd.DataFrame:
        # Extract and remove lead text
        df['lead_text'] = df['content'].str.extract(r'<ld>(.*?)</ld>', expand=False)
        df['content'] = df['content'].str.replace(r'<ld>.*?</ld>', '', regex=True)

        # Extract and remove subheadings
        df['subheadings'] = df['content'].str.extract(r'<zt>(.*?)</zt>', expand=False)
        df['content'] = df['content'].str.replace(r'<zt>.*?</zt>', '', regex=True)

        # Extract and remove author full name
        df['author'] = df['content'].str.extract(r'<au>(.*?)</au>', expand=False)
        df['content'] = df['content'].str.replace(r'<au>.*?</au>', '', regex=True)

        # Remove tags but keep annotated text within <a> tags
        df['content'] = df['content'].str.replace(r'<a[^>]*>(.*?)</a>', r'\1', regex=True)

        # Remove <tx>, <p>, <br>, <ka>, and <lg> tags
        df['content'] = df['content'].str.replace(r'</?(tx|p|br|ka|lg)[^>]*>', '', regex=True)

        # Remove last <p> element if it matches authors and extract to column
        df['author_extracted'] = df['content'].str.extract(r'<p>(.*?)(?=</p>$)', expand=False)
        df['content'] = df['content'].str.replace(r'<p>.*?</p>$', '', regex=True)

        return df