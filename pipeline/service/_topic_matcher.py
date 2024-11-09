from sentence_transformers import SentenceTransformer, util
import pandas as pd
import logging
import torch
from deep_translator import MyMemoryTranslator

logger = logging.getLogger(__name__)


class TopicMatcherService:

    @staticmethod
    def match_by_sentence_transformer(corpus, queries, number_of_top=5, match_score=0.9, invert=False):
        logger.info(f"Match by sentence transformer.")
        if torch.cuda.is_available():
            logger.info(f"GPU is available, setting mode to 'cuda'.")
            mode = 'cuda'
        else:
            logger.warning(f"GPU NOT available, setting mode to 'cpu'.")
            mode = 'cpu'
        query_label = 'german' if invert else 'french'
        corpus_label = 'french' if invert else 'german'
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=mode)
        hit_list = []
        result = []
        corpus_embedding = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)
        top_k = min(number_of_top, len(corpus))
        for query in queries:
            query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
            hits = util.semantic_search(query_embedding, corpus_embedding, score_function=util.dot_score)
            hits = hits[0]
            hit_list.append(hits)
            d: dict = {}
            for hit in hits[:top_k]:
                d[query_label] = query
                d[corpus_label] = corpus[hit['corpus_id']]
                d['score'] = round(hit['score'], 3)
            result.append(d)
        df = pd.DataFrame(result, columns=[query_label, corpus_label, "score"])
        best_matches = df[df["score"] >= match_score] if match_score else df
        return best_matches, df, hit_list, corpus_embedding, top_k

    @staticmethod
    def match_by_translation(german_tokens, french_tokens):
        logger.info(f"Match german and french tokens by translation.")
        matches = []
        german_token_list = german_tokens.to_list()
        french_token_list = french_tokens.to_list()
        try:
            translation = MyMemoryTranslator('de-DE', 'fr-FR').translate_batch(german_token_list)
            translation = [x.lower() for x in translation] # lowercase
            translation = [x.replace('la ', '').replace('le ', '') .replace('les ', '') for x in translation] # remove french articles
            logger.info(f"Translation of tokens done.")
        except Exception as e:
            logger.error(f"Translation of tokens could not be done, see error: {e}")
            return
        for french_translation, german_token in zip(translation, german_token_list):
            if french_translation in french_token_list:
                matches.append((german_token, french_translation))
        df = pd.DataFrame(matches, columns=["german", "french"])
        german_counts = 0
        french_counts = 0
        if len(df.index) != 0:
            german_counts = df['german'].value_counts()
            french_counts = df['french'].value_counts()
        return df, german_counts, french_counts


if __name__ == '__main__':
    pass
