from sentence_transformers import SentenceTransformer, util
import pandas as pd
import logging
import torch

logger = logging.getLogger(__name__)

class TopicMatcherService:

    @staticmethod
    def match(corpus, queries, number_of_top=5, print_matches=True):

        if torch.cuda.is_available():
            logger.info(f"GPU is available, setting mode to 'cuda'.")
            mode = 'cuda'
        else:
            logger.warning(f"GPU NOT available, setting mode to 'cpu'.")
            mode = 'cpu'

        # As per documentation this model is optimized to cluster sentences or paragraphs... not words
        # It supports 50 languages among others german and french
        # bigger model, same authors: https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=mode)

        hit_list = []

        corpus_embedding = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)

        top_k = min(number_of_top, len(corpus))

        for query in queries:
            query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
            hits = util.semantic_search(query_embedding, corpus_embedding, score_function=util.dot_score)
            hits = hits[0]
            hit_list.append(hits)

            if print_matches:
                print()
                print("Query:", query)
                print("---------------------------")
                for hit in hits[:top_k]:
                    print(f"{round(hit['score'], 3)} | {corpus[hit['corpus_id']]}")

        return hit_list, corpus_embedding, top_k


if __name__ == '__main__':
    # Corpus with example sentences
    corpus = [
        'I am a boy',
        'What are you doing?',
        'Can you help me?',
        'A man is riding a horse.',
        'A woman is playing violin.',
        'A monkey is chasing after a goat',
        'The quick brown fox jumps over the lazy dog'
    ]

    # Query sentences:
    queries = ['I am in need of assistance', '我是男孩子', 'Qué estás haciendo']

    df_topics_fr = pd.read_csv("../../data/topics/topics_fr.csv")

    df_topics_de = pd.read_csv("../../data/topics/topics_de.csv")

    TopicMatcherService.match(df_topics_de['Word'], df_topics_fr['Word'])
