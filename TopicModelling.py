# Import required libraries
import nltk
import spacy
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


def preprocess(texts, language='german'):
    stop_words = set(stopwords.words(language))
    processed_texts = []
    for doc in texts:
        # Tokenize the document
        nlp = nlp_de if language == 'german' else nlp_fr
        doc = nlp(doc.lower())  # Lowercase and tokenize
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]

        # Lemmatize words and remove stopwords
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

        processed_texts.append(lemmatized_tokens)

    return processed_texts


def topic_modeling(texts, language, num_topics=5, num_words=10):
    # Preprocess the texts
    processed_texts = preprocess(texts, language)

    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(processed_texts)

    # Create a corpus: Term Document Frequency (Bag of Words model)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Build the LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Print the topics
    topics = lda_model.print_topics(num_topics=num_topics, num_words=num_words)
    for idx, topic in topics:
        print(f"Topic {idx + 1}: {topic}")

    return lda_model, corpus, dictionary


if __name__ == '__main__':
    articles_fr = [
        "L'économie fait face à des défis importants en raison de l'inflation et du chômage.",
        "Les tensions politiques augmentent alors que les gouvernements luttent pour contrôler la pandémie.",
        "Les entreprises technologiques connaissent une croissance significative sur le marché boursier.",
        "Les équipes sportives s'adaptent à de nouvelles protocoles en raison de la crise sanitaire mondiale.",
        "Les préoccupations environnementales entraînent une augmentation des projets d'énergies renouvelables."
    ]

    # Run topic modeling for french
    lda_model_fr, corpus_fr, dictionary_fr = topic_modeling(articles_fr, 'french', num_topics=5, num_words=5)

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
    lda_model_de, corpus_de, dictionary_de = topic_modeling(articles_de, 'german', num_topics=5, num_words=5)
