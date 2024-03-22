from collections import Counter
import csv
import json
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import time
import logging

UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "static/outputs"


INPUT_FILE_PATH = f"{UPLOADS_DIR}/input.csv"
CACHE_FOLDER = ".cache/huggingface/hub"
CLUSTERING_OUTPUT_FILE = f"{OUTPUTS_DIR}/clustering_output.csv"
PAIRWISE_SIMILARITIES_OUTPUT_FILE = f"{OUTPUTS_DIR}/pairwise_similarities.csv"
OUTPUT_FILE_PATH = f"{OUTPUTS_DIR}/output.csv"
STATS_FILE_PATH = f"{OUTPUTS_DIR}/stats.json"
LANGUAGE_MODEL = "BAAI/bge-large-en-v1.5"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_input_file(
    col_delimiter: str = ",",
    num_words_per_row: int = 5,
    word_column_template: str = "word%d",
    cluster_column_template: str = "cluster%d",
    excluded_words: list[str] = [],
):
    rows: list[list[str]] = []
    word_counts: Counter[str] = Counter()
    with open(INPUT_FILE_PATH, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=col_delimiter)
        headers = reader.__next__()
        col_idxs: list[int] = []

        # store the indices for all columns that will contain the cluster indices
        out_col_idxs: list[int] = []

        for i in range(1, num_words_per_row + 1):
            # we set up the column name for the i-th word by filling
            # in our %d placeholder
            word_column_name = word_column_template % i
            cluster_column_name = cluster_column_template % i

            col_idx = headers.index(word_column_name)
            col_idxs.append(col_idx)

            cluster_col_idx = headers.index(cluster_column_name)
            out_col_idxs.append(cluster_col_idx)

        for user_entry in reader:
            rows.append(user_entry)

            for word_col_idx in col_idxs:
                # get the next word provided by the current participant
                word = user_entry[word_col_idx]
                if word == "" or word is None:
                    continue
                # if the word is in the list of forbidden words, ignore it
                if word.strip() in excluded_words:
                    continue
                # otherwise, count the word
                word_counts[word] += 1

    return rows, word_counts, headers, col_idxs, out_col_idxs


def outlier_detection(
    words: list[str],
    norm_embeddings: np.ndarray,
    outlier_k: int = 5,
    outlier_detection_threshold: float = 1.0,
) -> tuple[list[str], np.ndarray]:
    # compute the overall cosine similarity matrix between all embeddings
    S = np.dot(norm_embeddings, norm_embeddings.T)
    # get the average cosine similarities to the OUTLIER_K nearest neighbors for
    # each word (excluding the word itself). The numpy.partition function helps us
    # with that because it can find the smallest values in an array efficiently.
    # So we use that to find the OUTLIER_K+1 smallest negative similarities,
    # take the second to OUTLIER_K+1 values of those (to exclude the similarity
    # to the word itself), swap the sign again, and take the average.
    avg_neighbor_sim = np.mean(
        -np.partition(-S, outlier_k + 1, axis=1)[:, 1 : outlier_k + 1], axis=1
    )
    outlier_threshold = np.mean(
        avg_neighbor_sim
    ) - outlier_detection_threshold * np.std(avg_neighbor_sim)

    outliers = avg_neighbor_sim < outlier_threshold

    # take only the remaining words
    remaining_indexes = np.where(np.logical_not(outliers))[0]
    words_remaining = []
    for i in remaining_indexes:
        words_remaining.append(words[i])

    return words_remaining, norm_embeddings[remaining_indexes, :]


def find_number_of_clusters(
    embeddings_normalized: np.ndarray,
    max_num_clusters: int,
    sample_weights: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> int:
    max_num_clusters = min(
        max_num_clusters, len(embeddings_normalized) - 1
    )  # Prevent more clusters than words

    # set up the list of Ks we want to try
    if max_num_clusters < 50:
        # for max_num_clusters < 50, we try every possible value
        K_values = list(range(2, max_num_clusters + 1))
    elif max_num_clusters < 100:
        # for max_num_clusters >= 50, we try every fifth value
        K_values = list(range(2, 51)) + list(range(55, max_num_clusters + 1, 5))
    else:
        # for max_num_clusters >= 100, we try every tenth value
        K_values = (
            list(range(2, 51))
            + list(range(55, 101, 5))
            + list(range(110, max_num_clusters + 1, 10))
        )

    sils = []
    bics = []
    for K in tqdm(K_values):
        print(f"Computing K = {K}")
        logging.info(f"Computing K = {K}")
        clustering = KMeans(n_clusters=K, n_init=10, random_state=seed)
        clustering.fit(embeddings_normalized, sample_weight=sample_weights)
        sil = silhouette_score(np.asarray(embeddings_normalized), clustering.labels_)
        sils.append(sil)
        # compute the BIC score, which is a combination of the distance of each
        # word to its cluster center - provided by the clustering itself -
        bic = -clustering.score(embeddings_normalized)
        # ... and the number of parameters in our model, estimated by K
        bic += K
        bics.append(bic)

    # post-process both scales between 0 and 1 to be easier to
    # read visually
    sils = np.array(sils)
    sils = (sils - np.min(sils)) / (np.max(sils) - np.min(sils))

    bics = -np.array(bics)
    bics = (bics - np.min(bics)) / (np.max(bics) - np.min(bics))

    # identify the number of clusters automatically by selecting
    # the K that achieves the best product of both silhouette score
    # and BIC. The product is chosen to achieve both high silhoutte
    # AND high BIC score.
    K = K_values[np.argmax(sils * bics)]

    return K


def cluster_and_merge(
    words: list[str],
    norm_embeddings: np.ndarray,
    K: int,
    sample_weights: Optional[np.ndarray] = None,
    merge_threshold=1.0,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    clustering = KMeans(n_clusters=K, n_init=10, random_state=seed)
    clustering.fit(norm_embeddings, sample_weight=sample_weights)
    cluster_idxs = np.copy(clustering.labels_)
    centers_normalized = clustering.cluster_centers_ / np.linalg.norm(
        clustering.cluster_centers_, axis=1, keepdims=True, ord=2
    )

    if sample_weights is None:
        sample_weights = np.ones(len(words))

    # Just to make the merging process more transparent, get the word
    # that is closest to each cluster center
    S = np.dot(centers_normalized, norm_embeddings.T)
    exemplars = []
    for k in range(K):
        exemplars.append(words[np.argmax(S[k, :])])

    # if so requested, merge clusters that are closeby
    if merge_threshold is not None and merge_threshold < 1.0:
        print(
            "Merging similar clusters together until the cosine similarity of all cluster centers is below %g"
            % merge_threshold
        )
        logging.info(
            "Merging similar clusters together until the cosine similarity of all cluster centers is below %g",
            merge_threshold,
        )
        # merge the closest clusters using Agglomorative Clustering
        # until everything is closer than the threshold
        meta_clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - merge_threshold,
            linkage="complete",
            metric="cosine",
        )
        meta_clustering.fit(np.asarray(centers_normalized))

        # print which clusters got merged together
        for label in np.unique(meta_clustering.labels_):
            merged = np.where(meta_clustering.labels_ == label)[0]
            if len(merged) > 1:
                merged_exemplars = [exemplars[k] for k in merged]
                print(
                    "the following clusters got merged together: %s"
                    % (", ".join(merged_exemplars))
                )
                logging.info(
                    "the following clusters got merged together: %s",
                    ", ".join(merged_exemplars),
                )

        # override the original k-means result with the merged clusters
        K_new = len(np.unique(meta_clustering.labels_))
        for i in range(len(cluster_idxs)):
            cluster_idxs[i] = meta_clustering.labels_[cluster_idxs[i]]

        # re-set the cluster centers to the weighted mean of all their
        # points
        centers_new = np.zeros((K_new, centers_normalized.shape[1]))
        for k in range(K_new):
            in_cluster_k = cluster_idxs == k
            centers_new[k, :] = np.dot(
                sample_weights[in_cluster_k], norm_embeddings[in_cluster_k, :]
            ) / np.sum(sample_weights[in_cluster_k])

        # normalize the cluster centers again to unit length
        centers_normalized = centers_new / np.linalg.norm(
            centers_new, axis=1, keepdims=True, ord=2
        )

    return cluster_idxs, centers_normalized


def output_clustering_results(
    clustering_output_file: str,
    K: int,
    cluster_idxs: np.ndarray,
    embeddings_normalized: np.ndarray,
    centers_normalized: np.ndarray,
    words: list[str],
    col_delimiter: str = ",",
):
    with open(clustering_output_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=col_delimiter, lineterminator="\n")
        writer.writerow(["word", "cluster_index", "similarity_to_center"])
        # similarity to center refers to the distance from embedding to the
        # cluster mean which is a measure of how representative
        # the word is for the cluster

        for k in range(K):
            # get the indices of all words in cluster k
            in_cluster_k = np.where(cluster_idxs == k)[0]

            if len(in_cluster_k) == 0:
                continue

            # compute the cosine similarity of the embeddings of all words
            # in cluster k to the mean of cluster k
            sim = np.dot(
                embeddings_normalized[in_cluster_k, :], centers_normalized[k, :]
            )
            # iterate over all words in cluster k - but sort descendingly
            # by the cosine similarity because we may want to label clusters by
            # the most similar words
            for i in np.argsort(-sim):
                cluster_col_idx = in_cluster_k[i]
                word = words[cluster_col_idx]
                k = cluster_idxs[cluster_col_idx]
                s = sim[i].item()
                writer.writerow([word, k, s])


def output_pairwise_similarities(
    pairwise_similarities_file: str,
    centers_normalized: np.ndarray,
    col_delimiter: str = ",",
):
    # compute the pairwise similarities between all cluster centers
    S = np.dot(centers_normalized, centers_normalized.T)

    # # get the indexes of the pair of clusters with the highest similarity
    # S_copy = S.copy()
    # # Set diagonal elements to a value less than 1.0 to exclude them from argmax
    # np.fill_diagonal(S_copy, -1)
    # # Get the index of the maximum value closest to 1.0
    # max_index = np.unravel_index(np.argmax(S_copy, axis=None), S_copy.shape)
    np.savetxt(pairwise_similarities_file, S, fmt="%.2f", delimiter=col_delimiter)


def output_cluster_indices(
    output_file_path: str,
    num_words_per_row: int,
    col_idxs: list[int],
    out_col_idxs: list[int],
    words: list[str],
    cluster_idxs: np.ndarray,
    rows: list[list[str]],
    headers: list[str],
    col_delimiter: str = ",",
):
    word_idx_map = {word: idx for idx, word in enumerate(words)}
    for user_entry in rows:
        for i in range(num_words_per_row):
            # get the next word provided by the current participant
            word = user_entry[col_idxs[i]]
            cluster_col_idx = word_idx_map.get(word)
            if cluster_col_idx is None:
                user_entry[out_col_idxs[i]] = ""
                continue
            k = cluster_idxs[cluster_col_idx]
            user_entry[out_col_idxs[i]] = k

    with open(output_file_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=col_delimiter, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def main(
    col_delimiter: str = ",",
    num_words_per_row: int = 5,
    word_column_template: str = "word%d",
    cluster_column_template: str = "cluster%d",
    excluded_words: list[str] = [],
    outlier_k: int = 5,
    outlier_detection_threshold: float = 1.0,
    automatic_k: bool = False,
    max_num_clusters: int = 100,
    seed: Optional[int] = None,
    K: int = 5,
    merge_threshold: float = 1.0,
):
    logger.info("Starting clustering")
    start_time = time.time()
    # read the input file
    rows, word_counts, headers, col_idxs, out_col_idxs = read_input_file(
        col_delimiter=col_delimiter,
        num_words_per_row=num_words_per_row,
        word_column_template=word_column_template,
        cluster_column_template=cluster_column_template,
        excluded_words=excluded_words,
    )
    word_count = sum(word_counts.values())
    unique_word_count = len(word_counts)
    print(f"Number of rows: {len(rows)}")
    logging.info(f"Number of rows: {len(rows)}")
    print(f"Number of unique words: {unique_word_count}, total words: {word_count}")
    logging.info(
        f"Number of unique words: {unique_word_count}, total words: {word_count}"
    )
    words = list(word_counts.keys())

    # embed the words using the language model
    logger.info("Embedding words")
    model = SentenceTransformer(LANGUAGE_MODEL, cache_folder=CACHE_FOLDER)
    norm_embeddings = model.encode(
        words, normalize_embeddings=True, convert_to_numpy=True
    )  # shape (no_of_unique_words, embedding_dim)
    norm_embeddings = np.array(norm_embeddings)  # Type casting (only for IDE)
    logger.info("Embedding complete")

    # remove outliers
    words_no_outliers, norm_embeddings_no_outliers = outlier_detection(
        words, norm_embeddings, outlier_k, outlier_detection_threshold
    )
    print(
        f"Number of remaining words after outlier detection: {len(words_no_outliers)}"
    )
    logging.info(
        f"Number of remaining words after outlier detection: {len(words_no_outliers)}"
    )
    logging.info(f"Shape of embeddings: {norm_embeddings_no_outliers.shape}")

    outliers = set(words) - set(words_no_outliers)

    # a list of how often each word was named
    sample_weights = []
    for word in words_no_outliers:
        sample_weights.append(word_counts[word])
    sample_weights = np.array(sample_weights)

    # find the number of clusters
    if automatic_k:
        K = find_number_of_clusters(
            norm_embeddings_no_outliers, max_num_clusters, sample_weights, seed
        )
        print(f"Automatically determined number of clusters: {K}")
        logging.info(f"Automatically determined number of clusters: {K}")

    # cluster the embeddings
    cluster_idxs, centers_normalized = cluster_and_merge(
        words_no_outliers,
        norm_embeddings_no_outliers,
        K,
        sample_weights,
        merge_threshold,
        seed,
    )
    K_new = centers_normalized.shape[0]
    if K_new < K:
        print(f"Reduced number of clusters by merging to {K_new}")
        logging.info(f"Reduced number of clusters by merging to {K_new}")

    # output the clustering results
    output_clustering_results(
        CLUSTERING_OUTPUT_FILE,
        K_new,
        cluster_idxs,
        norm_embeddings_no_outliers,
        centers_normalized,
        words_no_outliers,
        col_delimiter,
    )
    print(f"Clustering written to {CLUSTERING_OUTPUT_FILE}")
    logging.info(f"Clustering written to {CLUSTERING_OUTPUT_FILE}")

    # output the pairwise similarities between all cluster centers
    output_pairwise_similarities(PAIRWISE_SIMILARITIES_OUTPUT_FILE, centers_normalized)
    print(f"Pairwise similarities written to {PAIRWISE_SIMILARITIES_OUTPUT_FILE}")
    logging.info(
        f"Pairwise similarities written to {PAIRWISE_SIMILARITIES_OUTPUT_FILE}"
    )

    # update the input file with the cluster indices
    output_cluster_indices(
        OUTPUT_FILE_PATH,
        num_words_per_row,
        col_idxs,
        out_col_idxs,
        words_no_outliers,
        cluster_idxs,
        rows,
        headers,
        col_delimiter,
    )
    print(f"Cluster indices written to {OUTPUT_FILE_PATH}")
    logging.info(f"Cluster indices written to {OUTPUT_FILE_PATH}")

    execution_time = time.time() - start_time

    stats = {
        "num_rows": len(rows),
        "num_words_initial": word_count,
        "num_unique_words_initial": unique_word_count,
        "num_words_post_outlier_detection": len(words_no_outliers),
        "num_clusters_initial": K,
        "num_clusters_post_merging": K_new,
        "automatic_k": automatic_k,
        "max_num_clusters": max_num_clusters,
        "outlier_k": outlier_k,
        "outlier_detection_threshold": outlier_detection_threshold,
        "merge_threshold": merge_threshold,
        "seed": seed,
        "execution_time": execution_time,
        "language_model": LANGUAGE_MODEL,
        "num_outliers": len(outliers),
        "outliers": list(outliers),
    }
    with open(STATS_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        logger.info(f"Stats written to {STATS_FILE_PATH}")

    return stats


if __name__ == "__main__":
    main()
