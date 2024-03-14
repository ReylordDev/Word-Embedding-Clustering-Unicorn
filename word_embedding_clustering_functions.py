#!python
"""
This Python module collects utility functions that we use in
'word_embedding_clustering' and 'word_embedding_clustering_with_conditions'

The scheme is adapted from Nicolas et al. (2022): Spontaneous Stereotype
Content Model: Taxonomy, Properties, and Prediction. Journal of Personality
and Social Psychology, 123(5), 1243-1263. doi:10.1037/pspa0000312

"""

# Word Embedding Clustering for Psychological Research
#
# Copyright (C) 2023
# Benjamin Paaßen
# AG Knowledge Representation and Machine Learning
# Bielefeld University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__author__ = "Benjamin Paaßen"
__copyright__ = "Copyright (C) 2023, Benjamin Paaßen"
__license__ = "GPLv3"
__version__ = "0.9.0"
__maintainer__ = "Benjamin Paaßen"
__email__ = "bpaassen@techfak.uni-bielefeld.de"


# These lines are intended to prevent the error
# OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata.
import os
from typing import Optional

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUM_THREADS"] = "8"
# os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "8"

import numpy as np  # noqa: E402
from sklearn.cluster import AgglomerativeClustering  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from tqdm import tqdm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def outlier_detection(
    stereotypes: list[str],
    embeddings_normalized: np.ndarray,
    OUTLIER_K=5,
    OUTLIER_DETECTION_THRESHOLD=1,
    SHOW_PLOTS=False,
) -> tuple[list[str], np.ndarray]:
    """Excludes stereotypes from the given list which have lower average
    cosine similarity to their OUTLIER_K nearest neighbors than
    OUTLIER_DETECTION_THRESHOLD times the standard deviation below the mean.

    Parameters
    ----------
    stereotypes: list
      A list of stereotypes (as strings).
    embeddings_normalized: numpy.ndarray
      The embedding matrix for all the stereotypes (each row needs to
      represent one stereotype). This ought to be normalized, i.e. the
      Euclidean norm of each row should be 1.
    OUTLIER_K: int (default = 5)
      The number of neighbors we consider for outlier detection.
    OUTLIER_DETECTION_THRESHOLD: float (default = 1)
      The amount of standard deviations below the mean cosine similarity
      we accept before we say that a stereotype is an outlier.
    SHOW_PLOTS: bool (default = False)
      If this is set to True, we plot the outlier detection - otherwise,
      we do not.

    Returns
    -------
    stereotypes: list
      The remaining stereotypes, i.e. all stereotypes that are not outliers.
    embeddings_normalized: numpy.ndarray
      The remaining embeddings.

    """
    # compute the overall cosine similarity matrix between all embeddings
    S = np.dot(embeddings_normalized, embeddings_normalized.T)
    # get the average cosine similarities to the OUTLIER_K nearest neighbors for
    # each word (excluding the word itself). The numpy.partition function helps us
    # with that because it can find the smallest values in an array efficiently.
    # So we use that to find the OUTLIER_K+1 smallest negative similarities,
    # take the second to OUTLIER_K+1 values of those (to exclude the similarity
    # to the word itself), swap the sign again, and take the average.
    avg_neighbor_sim = np.mean(
        -np.partition(-S, OUTLIER_K + 1, axis=1)[:, 1 : OUTLIER_K + 1], axis=1
    )
    outlier_threshold = np.mean(
        avg_neighbor_sim
    ) - OUTLIER_DETECTION_THRESHOLD * np.std(avg_neighbor_sim)

    outliers = avg_neighbor_sim < outlier_threshold

    if SHOW_PLOTS:
        plt.plot(-np.sort(-avg_neighbor_sim))
        plt.plot(
            [0, len(avg_neighbor_sim)], [outlier_threshold, outlier_threshold], "r--"
        )
        plt.xlabel("word index")
        plt.ylabel("average cosine similarity to %d nearest neighbors" % (OUTLIER_K))
        plt.title("excluding %d outliers" % np.sum(outliers))
        plt.show()

    # take only the remaining stereotypes
    remaining_indexes = np.where(np.logical_not(outliers))[0]
    stereotypes_remaining = []
    for i in remaining_indexes:
        stereotypes_remaining.append(stereotypes[i])

    return stereotypes_remaining, embeddings_normalized[remaining_indexes, :]


def find_number_of_clusters(
    embeddings_normalized: np.ndarray,
    MAX_NUM_CLUSTERS: int,
    sample_weights: Optional[np.ndarray] = None,
    SHOW_PLOTS=False,
) -> int:
    """Identifies the optimal number of clusters for KMeans using
    both silhouette score and Bayesian information criterion.

    Parameters
    ----------
    embeddings_normalized: numpy.ndarray
      The embedding matrix for all the stereotypes (each row needs to
      represent one stereotype). This ought to be normalized, i.e. the
      Euclidean norm of each row should be 1.
    MAX_NUM_CLUSTERS: int
      The maximum number of clusters to be considered.
    sample_weights: numpy.ndarray (default = None)
      A weight for each stereotype, e.g. how often each stereotype was
      named in the data set. If this is not given, all stereotypes will
      be weighed equally.
    SHOW_PLOTS: bool (default = False)
      If this is set to True, we plot the silhouette score and Bayesian
      information cirterion for all sampled Ks. Otherwise, we don't.

    Returns
    -------
    K: int
      The automatically selected optimal number of clusters.

    """
    # The original paper by Nicolas et al. (2022) appears to select the
    # optimal number of clusters by using the nbclust package in R.
    # This package varies the number of clusters, computes a clustering
    # for each number, and evaluates the resulting clustering with 30
    # different quality indices. The actual number chosen is then the
    # compromise of all 30 indices.
    # Unfortunately, we do not have the nbclust package in Python. However,
    # we can reproduce at least two of the most prominent cluster quality
    # indices, namely the silhouette score and the Bayesian information
    # criterion (BIC). We plot both for varying numbers of clusters (K)
    # such that the user can estimate the best K.

    # set up the list of Ks we want to try
    if MAX_NUM_CLUSTERS < 50:
        # for MAX_NUM_CLUSTERS < 50, we try every possible value
        Ks = list(range(2, MAX_NUM_CLUSTERS + 1))
    elif MAX_NUM_CLUSTERS < 100:
        # for MAX_NUM_CLUSTERS >= 50, we try every fifth value
        Ks = list(range(2, 51)) + list(range(55, MAX_NUM_CLUSTERS, 5))
    else:
        # for MAX_NUM_CLUSTERS >= 100, we try every tenth value
        Ks = (
            list(range(2, 51))
            + list(range(55, 101, 5))
            + list(range(110, MAX_NUM_CLUSTERS, 10))
        )

    # initialize two lists, one for the silhouette score, one for BIC
    sils = []
    bics = []
    # iterate over all possible K (starting at 2, because K = 1 is trivial)
    for K in tqdm(Ks):
        # initialize a new KMeans clustering with the right K parameter
        clustering = KMeans(
            n_clusters=K,
            n_init=10,
        )
        # fit the clustering to the data, taking into account how often each
        # stereotype was named
        clustering.fit(embeddings_normalized, sample_weight=sample_weights)
        # compute the silhouette score and store it
        sil = silhouette_score(np.asarray(embeddings_normalized), clustering.labels_)
        sils.append(sil)
        # compute the BIC score, which is a combination of the distance of each
        # stereotype to its cluster center - provided by the clustering itself -
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
    K = Ks[np.argmax(sils * bics)]

    # plot the curves
    if SHOW_PLOTS:
        plt.plot(Ks, sils)
        plt.plot(Ks, bics)
        plt.plot([K, K], [0, 1], "r--")
        plt.xlabel("number of clusters")
        plt.ylabel("normalized scores")
        plt.legend(["silhouette score", "inverse BIC", "automatic suggestion"])
        plt.show()

    return K


def cluster_and_merge(
    stereotypes: list[str],
    embeddings_normalized: np.ndarray,
    K: int,
    sample_weights: Optional[np.ndarray] = None,
    MERGE_THRESHOLD=1.0,
    SHOW_PLOTS=False,
) -> tuple[np.ndarray, np.ndarray]:
    """Clusters the data using KMeans and merges closeby clusters using
    complete linkage clustering on the cosine similarity.

    Parameters
    ----------
    stereotypes: list
      A list of stereotypes (as strings).
    embeddings_normalized: numpy.ndarray
      The embedding matrix for all the stereotypes (each row needs to
      represent one stereotype). This ought to be normalized, i.e. the
      Euclidean norm of each row should be 1.
    K: int
      The number of clusters to be used.
    sample_weights: numpy.ndarray (default = None)
      A weight for each stereotype, e.g. how often each stereotype was
      named in the data set. If this is not given, all stereotypes will
      be weighed equally.
    MERGE_THRESHOLD: float (default = 1.)
      The cosine similarity threshold for merging nearby clusters.
    SHOW_PLOTS: bool (default = False)
      If this is set to True, we show all merged clusters.
      Otherwise, we don't.

    Returns
    -------
    cluster_idxs: numpy.ndarray
      The cluster indices for all stereotypes.
    centers_normalized: numpy.ndarray
      The embedding matrix for all cluster centers.

    """

    # initialize a new KMeans clustering with the right K parameter
    clustering = KMeans(n_clusters=K, n_init=10)
    # fit the clustering to the data, taking into account how often each
    # stereotype was named
    clustering.fit(embeddings_normalized, sample_weight=sample_weights)
    # extract the cluster indices for all data points
    cluster_idxs = np.copy(clustering.labels_)
    # get the normalized cluster means
    centers_normalized = clustering.cluster_centers_ / np.linalg.norm(
        clustering.cluster_centers_, axis=1, keepdims=True, ord=2
    )

    if sample_weights is None:
        sample_weights = np.ones(len(stereotypes))

    # Just to make the merging process more transparent, get the stereotype
    # that is closest to each cluster center
    S = np.dot(centers_normalized, embeddings_normalized.T)
    exemplars = []
    for k in range(K):
        exemplars.append(stereotypes[np.argmax(S[k, :])])

    # if so requested, merge clusters that are closeby
    if MERGE_THRESHOLD is not None and MERGE_THRESHOLD < 1.0:
        print(
            "Merging similar clusters together until the cosine similarity of all cluster centers is below %g"
            % MERGE_THRESHOLD
        )
        # Initialize previous_cluster_idxs
        previous_cluster_idxs = cluster_idxs.copy()

        # Loop until cluster_idxs didn't change in the previous loop
        while True:
            # merge the closest clusters using Agglomorative Clustering
            # until everything is closer than the threshold
            meta_clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1.0 - MERGE_THRESHOLD,
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
                    sample_weights[in_cluster_k], embeddings_normalized[in_cluster_k, :]
                ) / np.sum(sample_weights[in_cluster_k])

            # normalize the cluster centers again to unit length
            centers_normalized = centers_new / np.linalg.norm(
                centers_new, axis=1, keepdims=True, ord=2
            )

            if np.max(cluster_idxs) == np.max(previous_cluster_idxs):
                break

            # Update previous_cluster_idxs and previous_centers_normalized
            previous_cluster_idxs = cluster_idxs.copy()
            # compute the pairwise similarities between all cluster centers
            S = np.dot(centers_normalized, centers_normalized.T)

            # get the indexes of the pair of clusters with the highest similarity
            S_copy = S.copy()
            # Set diagonal elements to a value less than 1.0 to exclude them from argmax
            np.fill_diagonal(S_copy, -1)
            # Get the index of the maximum value closest to 1.0
            max_index = np.unravel_index(np.argmax(S_copy, axis=None), S_copy.shape)
            print(
                f"Pair of clusters with highest similarity: {max_index} with similarity {S[max_index]}"
            )

    return cluster_idxs, centers_normalized
