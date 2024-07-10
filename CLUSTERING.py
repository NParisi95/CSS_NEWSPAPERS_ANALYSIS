# =================================
# CLUSTERING
# =================================

import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import string
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def main():
    # Load the dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fp_only_path = os.path.join(current_dir, 'ARTICLES DATA', 'MERGED DATASET', 'merged_output_FIRST_PAGE_ONLY.csv')
    df = pd.read_csv(fp_only_path)
    df = df[df["published_date"] != "2024-06-29"]
    df = df.dropna(subset=['Text_lemm'])

    stop_words = set(stopwords.words('italian'))

    def preprocess(text):
        custom_filters = [strip_punctuation, strip_numeric]
        tokens = preprocess_string(text, custom_filters)
        return [token for token in tokens if token not in stop_words and token not in string.punctuation]

    df['Text_lemm_processed'] = df['Text_lemm'].apply(preprocess)

    dictionary = corpora.Dictionary(df['Text_lemm_processed'])
    corpus = [dictionary.doc2bow(text) for text in df['Text_lemm_processed']]

    def topics_to_array(doc_topics, num_topics):
        array = np.zeros((len(doc_topics), num_topics))
        for i, doc in enumerate(doc_topics):
            for topic, prob in doc:
                array[i, topic] = prob
        return array

    coherence_scores = {}
    results = []

    for num_topics in range(2, 21):
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=df['Text_lemm_processed'], dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        coherence_scores[num_topics] = coherence_lda

        doc_topics = [lda_model.get_document_topics(bow) for bow in corpus]
        topic_matrix = topics_to_array(doc_topics, num_topics)

        df['Cluster'] = np.argmax(topic_matrix, axis=1)

        for i in range(num_topics):
            words = lda_model.show_topic(i, topn=10)
            words = [word for word, prob in words]
            results.append({
                'iteration': f'num_topics_{num_topics}',
                'cluster': i,
                'words': ', '.join(words)
            })

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(topic_matrix)

        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        tsne_result = tsne.fit_transform(topic_matrix)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['Cluster'])
        plt.title(f'PCA of LDA Topics (num_topics={num_topics})')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')

        plt.subplot(1, 2, 2)
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=df['Cluster'])
        plt.title(f't-SNE of LDA Topics (num_topics={num_topics})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

        plt.show()

    results_df = pd.DataFrame(results)
    results_df.to_csv('lda_cluster_words.csv', index=False)

    coherence_df = pd.DataFrame(list(coherence_scores.items()), columns=['num_topics', 'coherence_score'])
    coherence_df.to_csv('lda_coherence_scores.csv', index=False)

    for num_topics, coherence in coherence_scores.items():
        print(f'Number of Topics: {num_topics}, Coherence Score: {coherence}')

if __name__ == '__main__':
    main()
