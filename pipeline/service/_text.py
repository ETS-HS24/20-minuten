import pandas as pd


class TextService:

    @staticmethod
    def remove_tags(df: pd.DataFrame, columns_to_clean=None) -> pd.DataFrame:
        if columns_to_clean is None:
            columns_to_clean = ["content"]
        for column in columns_to_clean:
            df[column] = df[column].str.replace(r'<[^>]*>', '', regex=True)
        return df