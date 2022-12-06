import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from .format_helper import ProgressDecorator

warnings.simplefilter(action='ignore', category=FutureWarning)

tqdm.pandas()


class Checker:
    def __init__(self, model, corpus):
        self.cnt = 0
        self.model = model
        cnt = 0
        for word in corpus:
            if self.checkEmbedding(word):
                print(f"'{word : <32}'", end="       ")
                cnt += 1
                if not cnt % 4:
                    print()
        print("\n")
        print(f"Total number of words not in dictionary is {cnt}.")

    def checkEmbedding(self, sent):
        sent = sent.split()
        n = len(self.model["as"])
        sent_vec = np.zeros(n)
        word_cnt = 0
        for word in sent:
            if word in self.model:
                sent_vec += self.model[word]
                word_cnt += 1
        if word_cnt == 0:
            self.cnt += 1
            return True
        else:
            return False


def plot_dendrogram(agg_model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    model = agg_model
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


class Clustering:
    def __init__(self, model, dendrogram_flag=False, star_cnt=200):
        self.model = model
        self.lemmatizer = WordNetLemmatizer()
        self.dendrogram_flag = dendrogram_flag
        self.star_cnt = star_cnt

    def lemmatize(self, s):
        token_words = word_tokenize(s)
        word_list = [self.lemmatizer.lemmatize(word.lower(), 'v') for word in token_words]
        return " ".join((word_list))

    def getEmbedding(self, sent):
        sent = sent.split()
        n = len(self.model["as"])
        sent_vec = np.zeros(n)
        word_cnt = 0
        for word in sent:
            if word in self.model:
                sent_vec += self.model[word]
                word_cnt += 1
        if word_cnt != 0:
            sent_vec /= word_cnt
        return sent_vec

    @ProgressDecorator("Clustering Relationships")
    def get_cluster_word(self, df, threshold):
        def get_cluster_word(embed):
            return self.model.similar_by_vector(embed)[0][0]

        print(f"Total input length: {len(df)}")
        dfToReturn = df.copy()

        # Lemmatize
        dfToReturn["relationship_lem"] = dfToReturn.relationship.apply(self.lemmatize)

        print("." * self.star_cnt)
        print("Checking words not in model vocabulary: \n")
        Checker(self.model, dfToReturn.relationship_lem)
        print("." * self.star_cnt)

        # Get embedding for each word
        dfToReturn["embedding"] = dfToReturn.relationship_lem.apply(self.getEmbedding)

        # Extract valid embeddings for clustering, invalid embeddings are 0 everywhere
        validEmbeddings = dfToReturn[dfToReturn.embedding.apply(lambda x: np.any(x))].embedding
        print(f"Valid row count: {len(validEmbeddings)}")
        print("." * self.star_cnt)

        # Use AgglomerativeClustering on valid embeddings
        print("Start clustering")
        agg = AgglomerativeClustering(n_clusters=None, affinity="cosine", linkage="complete",
                                      distance_threshold=threshold).fit((np.array(list(validEmbeddings))))
        print("Finished clustering")
        print(f"Cluster centers count: {agg.n_clusters_}    with threshold of {threshold}")
        print("." * self.star_cnt)

        # Assign cluster centers accordingly
        dfToReturn["cluster"] = pd.DataFrame(agg.labels_, index=validEmbeddings.index)

        # Calculate average word2vec for each clusters
        cluster_embedding = pd.DataFrame(dfToReturn.groupby("cluster").embedding.apply(np.mean)).reset_index()

        # Infer the cluster word from the most similar vectors approach
        print("Start inferencing cluster center word")
        cluster_embedding["cluster_word"] = cluster_embedding.embedding.progress_apply(get_cluster_word)
        print()
        print("Finished inferencing cluster center word")

        if self.dendrogram_flag:
            # Plot the dendrogram plot (optional)
            figure(figsize=(10, 8), dpi=80)
            plot_dendrogram(agg, labels=list(df.relationship), leaf_font_size=10, orientation='right')
            plt.axvline(x=threshold, color='r', linestyle='dashed')
            plt.show()

        # Merge to return the final output dataframe
        df_final_result = dfToReturn.merge(cluster_embedding[["cluster", "cluster_word"]], on=["cluster"], how="left")
        return df_final_result.dropna()
