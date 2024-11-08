from sentence_transformers import SentenceTransformer, util
import pandas as pd
import logging
import torch
from deep_translator import MyMemoryTranslator

logger = logging.getLogger(__name__)


class TopicMatcherService:

    @staticmethod
    def match(corpus, queries, number_of_top=5, match_score=0.9, print_matches=True, invert=False):
        if torch.cuda.is_available():
            logger.info(f"GPU is available, setting mode to 'cuda'.")
            mode = 'cuda'
        else:
            logger.warning(f"GPU NOT available, setting mode to 'cpu'.")
            mode = 'cpu'

        query_label = 'German' if invert else 'French'
        corpus_label = 'French' if invert else 'German'

        # As per documentation this model is optimized to cluster sentences or paragraphs... not words
        # It supports 50 languages among others german and french
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

            if print_matches:
                print()
                print("Query:", query)
                print("---------------------------")

            for hit in hits[:top_k]:
                d[query_label] = query
                d[corpus_label] = corpus[hit['corpus_id']]
                d['Score'] = round(hit['score'], 3)

                if print_matches:
                    print(f"{round(hit['score'], 3)} | {corpus[hit['corpus_id']]}")

            result.append(d)

        df = pd.DataFrame(result, columns=[query_label, corpus_label, "Score"])
        best_matches = df[df["Score"] >= match_score] if match_score else df

        return best_matches, hit_list, corpus_embedding, top_k

    @staticmethod
    def match_by_translation(german_tokens, french_tokens, print_matches=True):
        matches = []
        german_token_list = german_tokens.to_list()
        french_token_list = french_tokens.to_list()
        try:
            translation = MyMemoryTranslator('de-DE', 'fr-FR').translate_batch(german_token_list)
            if print_matches:
                print(translation)
        except Exception as e:
            print(e)
            return
        for index, german_token in enumerate(german_token_list):
            if translation[index] in french_token_list:
                matches.append((german_token, translation[index]))
                if print_matches:
                    print(german_token, translation[index])
        df = pd.DataFrame(matches)
        df.columns.array[0] = 'German'
        df.columns.array[1] = 'French'
        return df, df['German'].value_counts(), df['French'].value_counts()


if __name__ == '__main__':
    pass
