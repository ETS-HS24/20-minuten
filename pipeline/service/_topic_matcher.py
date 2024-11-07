from sentence_transformers import SentenceTransformer, util
import pandas as pd
import logging
import torch

logger = logging.getLogger(__name__)


class TopicMatcherService:

    @staticmethod
    def match(corpus, queries, number_of_top=5, match_score=0.9, print_matches=True):
        if torch.cuda.is_available():
            logger.info(f"GPU is available, setting mode to 'cuda'.")
            mode = 'cuda'
        else:
            logger.warning(f"GPU NOT available, setting mode to 'cpu'.")
            mode = 'cpu'

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
                d['French'] = query
                d['German'] = corpus[hit['corpus_id']]
                d['Score'] = round(hit['score'], 3)

                if print_matches:
                    print(f"{round(hit['score'], 3)} | {corpus[hit['corpus_id']]}")

            result.append(d)

        df = pd.DataFrame(result, columns=["French", "German", "Score"])
        best_matches = df[df["Score"] >= match_score] if match_score else df

        return best_matches, hit_list, corpus_embedding, top_k


if __name__ == '__main__':
    pass
