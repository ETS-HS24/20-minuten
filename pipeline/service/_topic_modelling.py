import nltk
import spacy
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize spacy model and lemmatizer
nlp_fr = spacy.load('fr_core_news_sm')
nlp_de = spacy.load('de_core_news_sm')
lemmatizer = WordNetLemmatizer()

# Run this before in your cmd line:
# pip install gensim nltk spacy scikit-learn
# python -m spacy download fr_core_news_sm
# python -m spacy download de_core_news_sm

class TopicModellingService:
    @staticmethod
    def preprocess(texts, language='german'):
        stop_words = set(stopwords.words(language))
        processed_texts = []
        for doc in texts:
            # Tokenize the document
            nlp = nlp_de if language == 'german' else nlp_fr
            doc = nlp(str(doc).lower())  # Lowercase and tokenize
            tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]

            # Lemmatize words and remove stopwords
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

            processed_texts.append(lemmatized_tokens)

        return processed_texts

    @staticmethod
    def topic_modeling(texts, language, num_topics=5, num_words=10, print_topics=False, export_to_csv=False, filepath="data/topics", filename="topics"):
        # print('-----')
        # print(f'Language: {language}')
        # print(f'Length of texts: {len(texts)} {texts}')

        # Preprocess the texts
        processed_texts = TopicModellingService.preprocess(texts, language)

        # Create a dictionary representation of the documents
        dictionary = corpora.Dictionary(processed_texts)

        # Create a corpus: Term Document Frequency (Bag of Words model)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]

        # Build the LDA model
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

        # Create dataframe
        df = TopicModellingService.get_dataframe(lda_model, num_topics, num_words)

        # Print topics
        if print_topics:
            # print('Print topics')
            TopicModellingService.print_topics(lda_model, num_topics, num_words)

        # Export to csv
        if export_to_csv:
            # print('Export topics to csv')
            TopicModellingService.to_csv(df, filepath + filename + "_" + language)

        return df, lda_model, corpus, dictionary

    @staticmethod
    def print_topics(lda_model, num_topics=5, num_words=10):
        # Print the topics
        topics = lda_model.print_topics(num_topics=num_topics, num_words=num_words)
        for idx, topic in topics:
            print(f"Topic {idx + 1}: {topic}")

    @staticmethod
    def get_dataframe(lda, num_topics=5, num_words=10, columns=['Topic', 'Word', 'Score']):
        top_words_per_topic = []
        for t in range(lda.num_topics):
            top_words_per_topic.extend([(t,) + x for x in lda.show_topic(t, topn=num_topics)])

        return pd.DataFrame(top_words_per_topic, columns=columns)

    @staticmethod
    def to_csv(df, filename):
        return df.to_csv(filename + ".csv", encoding='utf-8-sig')


if __name__ == '__main__':
    articles_fr = [
        "L'économie fait face à des défis importants en raison de l'inflation et du chômage.",
        "Les tensions politiques augmentent alors que les gouvernements luttent pour contrôler la pandémie.",
        "Les entreprises technologiques connaissent une croissance significative sur le marché boursier.",
        "Les équipes sportives s'adaptent à de nouvelles protocoles en raison de la crise sanitaire mondiale.",
        "Les préoccupations environnementales entraînent une augmentation des projets d'énergies renouvelables."
    ]

    # Run topic modeling for french
    df_fr, lda_model_fr, corpus_fr, dictionary_fr = TopicModellingService.topic_modeling(articles_fr, 'french', num_topics=5, num_words=5, print_topics=True, export_to_csv=True, filepath="../../data/topics/", filename="articles")
    print(df_fr)

    print()
    print('---')
    print()

    articles_de = [
        "Hunderte protestierten auf dem Bundesplatz gegen die zweite Entlassungswelle im Stahlwerk Gerlafingen. Die Zukunft der Arbeiter und ihrer Familien steht auf dem Spiel.",
        "Die Schweizer Tennishoffnung Dominic Stricker spricht vor dem Heimturnier in Basel über ihre Leidenszeit, ihren Höhenflug in Stockholm und ihren Fitnesszustand.",
        "In einer Asylunterkunft ist es zu einem Streit zwischen zwei Männern gekommen. Dabei wurde einer von ihnen durch mehrere Messerstiche lebensbedrohlich verletzt.",
        "Auch zwischen den Sessionen werden in Bundesbern wichtige Entscheide getroffen. Hier halten wir dich auf dem Laufenden.",
        "In Saint-Malo in der Bretagne kam es zu eindrücklichen Springtiden. Das Phänomen, das auch als Springflut bekannt ist, führt zu riesigen Wellen. Obwohl die Behörden zur Vorsicht mahnten, begaben sich Schaulustige in Gefahr."
    ]

    # Run topic modeling for german
    df_de, lda_model_de, corpus_de, dictionary_de = TopicModellingService.topic_modeling(articles_de, 'german', num_topics=5, num_words=5, print_topics=True, export_to_csv=True, filepath="../../data/topics/", filename="articles")
    print(df_de)
