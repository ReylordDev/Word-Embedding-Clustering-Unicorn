#!python
"""

This Python script is intended as starting point for work by the
Unicorn Lab at Purdue University to cluster stereotypes using word
embeddings.

The scheme is adapted from Nicolas et al. (2022): Spontaneous Stereotype
Content Model: Taxonomy, Properties, and Prediction. Journal of Personality
ans Social Psychology, 123(5), 1243-1263. doi:10.1037/pspa0000312

Original author: Benjamin Paassen - Bielefeld University
Version 2023-09-30

The rough steps of processing are:
1.  We load data from the CSV file DATA_INPUT_FILE, where each participant
   in the study is represented by one row and their answers by columns.
   In particular, We extract the stereotypes from the columns
   STEREOTYPE_COLUMN. We also load the experimental condition from
   CONDITION_COLUMN because a separate clustering analysis should be
   performed for each condition.
2.  We embed all the stereotypes using a huggingface language model.
   The model can be changed by setting the LANGUAGE_MODEL parameter.
2b. (OPTIONAL) we perform outlier detection.
3.  (OPTIONAL) we cluster the embeddings for varying number of clusters
   and compute both silhouette scores and Bayesian Information Criterion
   to find the optimal number of clusters.
4.  We cluster the embeddings using K-Means with NUM_CLUSTER clusters.
4b. (OPTIONAL) we merge clusters which are very close together using
   agglomorative clustering (in particular: complete linkage)
5.  We write each unique word, its cluster index, and its cosine similarity
   to the cluster mean to a separate CLUSTERING_OUTPUT_FILE.
   We also write the similarities between cluster centers to
   CLUSTER_SIMILARITIES_FILE.
6.  We write the cluster indices for each stereotype into the
   CLUSTER_COLUMN columns and write the result into the CSV file
   DATA_OUTPUT_FILE.

Steps 2-5 are repeated for all experimental conditions.

We start by importing all the packages we will need
If you want to run this script, you will need

pip install sentence_transfomers
pip install scikit-learn

everything else should be automatically installed with these
two packages

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

import torch
import word_embedding_clustering_functions
from collections import Counter
import csv
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse

# Here, we define our parameters. We do that at the start of the file
# to make the parameters easier to change if we need to.

# The path to the input CSV file from which we read all the stereotypes
# that participants answered.
DATA_INPUT_FILE = "test_input_file_4.csv"

# The character that is used to delimit columns in the input (and output)
# file
COL_DELIMITER = ","

# The name of the columns from which we read the stereotypes. The %d
# is a placeholder.
STEREOTYPE_COLUMN = "Stereotype%d"

# The name of the column from which to read the experimental condition.
CONDITION_COLUMN = "Condition"

# The maximum number of stereotypes each study participant can provide/
# the number of stereotype columns we check for
NUM_STEREOTYPES = 10

# stereotypes which should be excluded from the analysis.
EXCLUDED_STEREOTYPES = ["", "white"]

# The language model to be used for word embeddings
LANGUAGE_MODEL = "BAAI/bge-large-en-v1.5"

# If this is True, the script will show intermediate plots to illustrate
# the outlier detection and the choice of K. If False, the plots won't be
# shown.
SHOW_PLOTS = False

# The number of neighbors to be considered for outlier detection
OUTLIER_K = 5

# The detection threshold for outlier detection, in the sense that any
# point is considered an outlier where the average cosine similarity
# to its OUTLIER_K nearest neighbors is OUTLIER_DETECTION_THRESHOLD
# standard deviations below the average
OUTLIER_DETECTION_THRESHOLD = 1

# the maximum number of clusters to be considered in step 3
MAX_NUM_CLUSTERS = 250

# Whether we want to run the auxiliary functions to help choose the
# NUM_CLUSTERS parameter
HELP_CHOOSE_NUM_CLUSTERS = True

# The number of clusters we use for clustering in step 4, defined
# separately for each condition.
# NOTE: This script also contains auxiliary functions to help with choosing
# this parameter.
# If HELP_CHOOSE_NUM_CLUSTERS is True, this parameter may be overriden
# by the user when executing the script.
NUM_CLUSTERS = {
    # The number of clusters for condition 0
    "0": 170,
    # The number of clusters for condition 1
    "1": 170,
}

# The cosine similarity threshold above which neighboring clusters
# get merged together. If this should not be done, set the threshold
# above 1
MERGE_THRESHOLD = 0.95

# The path to the output CSV file to which we copy the input data plus
# the cluster index for each stereotype
DATA_OUTPUT_FILE = "test_output_file_condition.csv"

# The name of the columns to which we write the cluster index after
# the analysis is complete. The %d is a placeholder.
CLUSTER_COLUMN = "Stereotype%d_cluster"

# The path to the output CSV file to which we write the clustering of
# all unique words. The %s is a placeholder for the experimental condition.
CLUSTERING_OUTPUT_FILE = "clustering_%s.csv"

# The path to the output CSV file to which we write the pairwise
# similarities between clusters. The %s is a placeholder for the
# experimental condition.
CLUSTER_SIMILARITIES_FILE = "cluster_similarities_%s.csv"


def main(
    args: argparse.Namespace,
):
    # STEP 1: OPEN THE INPUT DATA FILE AND READ ALL STEREOTYPES
    print(f'Starting step 1 of 6: Reading the input data from file "{args.input}"')

    rows = []
    # we also prepare a dictionary which will map the experimental condition
    # to a collection of unique stereotypes which were named in that condition,
    # which are in turn associated with how often they have been named
    condition_to_stereotype_counts_map = {}
    with open(args.input, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=args.col_delimiter)
        headers = reader.__next__()

        # get the index of the column from which we read the experimental condition
        cond_idx = headers.index(args.condition_column)

        # store the indices for all columns that may contain stereotypes
        col_idxs: list[int] = []

        for i in range(1, args.num_stereotypes + 1):
            # we set up the column name for the i-th stereotype by filling
            # in our %d placeholder
            stereotype_column_name = args.stereotype_column % i
            col_idx = headers.index(stereotype_column_name)
            col_idxs.append(col_idx)

        # store the indices for all columns that will contain the cluster indices
        out_col_idxs: list[int] = []

        for i in range(1, args.num_stereotypes + 1):
            # we set up the column name for the i-th stereotype by filling
            # in our %d placeholder
            cluster_column_name = args.cluster_column % i
            col_idx = headers.index(cluster_column_name)
            out_col_idxs.append(col_idx)

        for row in reader:
            rows.append(row)
            condition = row[cond_idx]
            # get the corresponding stereotype counter for the condition
            stereotype_counts: Counter[str] = (
                condition_to_stereotype_counts_map.setdefault(condition, Counter())
            )
            for col_idx in col_idxs:
                # get the next stereotype provided by the current participant
                stereotype = row[col_idx]
                # if the stereotype is in the list of forbidden words, ignore it
                if stereotype.strip() in args.excluded_stereotypes:
                    continue
                # otherwise, count the stereotype
                stereotype_counts[stereotype] += 1

    print(f"Completed step 1. Read {len(rows)} responses.")

    # We initialize the language model, which can be shared across conditions.
    print(
        f"Preparing step 2 of 6 by initializing the language model {args.language_model}. This may take a while when this script is run the first time."
    )
    model = SentenceTransformer(
        args.language_model, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # THE REMAINDER OF THIS ANALYSIS, WE WILL DO SEPARATELY FOR EACH CONDITION
    for condition in condition_to_stereotype_counts_map:
        stereotype_counts = condition_to_stereotype_counts_map[condition]
        no_of_unique_stereotypes = len(condition_to_stereotype_counts_map[condition])
        no_of_stereotypes = sum(condition_to_stereotype_counts_map[condition].values())
        print(
            f"Starting analysis for condition {condition} with {no_of_unique_stereotypes} unique stereotypes ({no_of_stereotypes} stereotypes overall)"
        )

        # get an ordered list of all stereotypes.
        stereotypes = list(stereotype_counts.keys())

        # STEP 2. EMBED ALL STEREOTYPES USING A LANGUAGE MODEL
        print("Starting step 2 of 6: Generating word embeddings.")

        print(
            f"Embedding {len(stereotypes)} unique stereotypes. This may take a few seconds."
        )
        embeddings_normalized = model.encode(
            stereotypes, normalize_embeddings=True, convert_to_numpy=True
        )  # shape (no_of_unique_stereotypes, embedding_dim)
        embeddings_normalized = np.array(embeddings_normalized)

        print(
            f"Filtering out words that are outliers, in the sense that their average cosine similarity to the {args.outlier_k} nearest neighbors is at least {args.outlier_detection_threshold} standard deviations below the average."
        )

        stereotypes, embeddings_normalized = (
            word_embedding_clustering_functions.outlier_detection(
                stereotypes,
                embeddings_normalized,
                args.outlier_k,
                args.outlier_detection_threshold,
                SHOW_PLOTS=args.show_plots,
            )
        )

        print(
            f"Completed outlier detection. {len(stereotypes)} unique stereotypes are remaining."
        )

        stereotype_idx_map = {}
        for i in range(len(stereotypes)):
            stereotype_idx_map[stereotypes[i]] = i

        # a list of how often each stereotype was named
        sample_weights = []
        for stereotype in stereotypes:
            sample_weights.append(stereotype_counts[stereotype])
        sample_weights = np.array(sample_weights)

        print(
            f"Completed step 2. Got an embedding matrix of size {embeddings_normalized.shape[0]} x {embeddings_normalized.shape[1]}"
        )

        if not args.help_choose_num_clusters:
            print("Skipping step 3 of 6.")
            K = args.num_clusters[condition]
        else:
            # STEP 3: HELP TO CHOOSE THE OPTIMAL NUMBER OF CLUSTERS
            print(
                "Starting step 3 of 6: trying to estimate the optimal number of clusters by repeating K-Means clustering for different K and computing quality measures for each K. This may take several minutes for large K."
            )

            K = word_embedding_clustering_functions.find_number_of_clusters(
                embeddings_normalized,
                args.max_num_clusters,
                sample_weights=sample_weights,
                SHOW_PLOTS=args.show_plots,
            )

            print(f"Completed step 3 of 6 with chosen number of clusters: {K}")

        # STEP 4: KMEANS CLUSTERING
        print(f"Starting step 4 of 6: Perform K-Means Clustering with {K} clusters.")

        cluster_idxs, centers_normalized = (
            word_embedding_clustering_functions.cluster_and_merge(
                stereotypes,
                embeddings_normalized,
                K,
                sample_weights=sample_weights,
                MERGE_THRESHOLD=args.merge_threshold,
                SHOW_PLOTS=args.show_plots,
            )
        )

        K_new = centers_normalized.shape[0]
        if K_new < K:
            print(f"Reduced number of clusters by merging to {K}")

        print("Completed step 4 of 6.")

        # STEP 5: WRITE CLUSTERING TO OUTPUT

        clustering_file_c: str = args.clustering_output_file % condition
        print(
            f'Starting step 5 of 6: Writing clustering to the file "{args.clustering_output_file}"'
        )

        with open(clustering_file_c, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=args.col_delimiter)
            writer.writerow(["stereotype", "cluster_index", "similarity_to_center"])
            # similarity to center refers to the distance from embedding to the
            # cluster mean which is a measure of how representative
            # the stereotype is for the cluster

            for k in range(K):
                # get the indices of all stereotypes in cluster k
                in_cluster_k = np.where(cluster_idxs == k)[0]

                if len(in_cluster_k) == 0:
                    continue

                # compute the cosine similarity of the embeddings of all stereotypes
                # in cluster k to the mean of cluster k
                sim = np.dot(
                    embeddings_normalized[in_cluster_k, :], centers_normalized[k, :]
                )
                # iterate over all stereotypes in cluster k - but sort descendingly
                # by the cosine similarity because we may want to label clusters by
                # the most similar stereotypes
                for i in np.argsort(-sim):
                    col_idx = in_cluster_k[i]
                    stereotype = stereotypes[col_idx]
                    k = cluster_idxs[col_idx]
                    s = sim[i]
                    writer.writerow([stereotype, k, s])

        # compute the pairwise similarities between all cluster centers
        S = np.dot(centers_normalized, centers_normalized.T)

        cluster_sim_file_c = args.cluster_similarities_file % condition
        print(
            f"Writing pairwise cluster center similarities to file {cluster_sim_file_c}."
        )
        np.savetxt(cluster_sim_file_c, S, fmt="%.2f", delimiter=args.col_delimiter)

        print("Completed step 5 of 6.")

        # STEP 6: WRITE CLUSTER INDICES BACK
        print("Preparing step 6 of 6 by writing cluster indices to data")

        for row in rows:
            if row[cond_idx] != condition:
                continue

            for i in range(args.num_stereotypes):
                # get the next stereotype provided by the current participant
                stereotype = row[col_idxs[i]]
                col_idx = stereotype_idx_map.get(stereotype)

                if col_idx is None:
                    row[out_col_idxs[i]] = ""
                    continue

                k = cluster_idxs[col_idx]
                row[out_col_idxs[i]] = k

    print(
        f'Starting step 6 of 6: Writing cluster indices for each stereotype to the file "{args.data_output_file}"'
    )
    with open(args.data_output_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=args.col_delimiter)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)

    print("Completed step 6 of 6. End of script.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster stereotypes using word embeddings."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DATA_INPUT_FILE,
        help="The path to the input CSV file from which we read all the stereotypes that participants answered.",
    )
    parser.add_argument(
        "--col_delimiter",
        type=str,
        default=COL_DELIMITER,
        help="The character that is used to delimit columns in the input (and output) file.",
    )
    parser.add_argument(
        "--stereotype_column",
        type=str,
        default=STEREOTYPE_COLUMN,
        help="The name of the columns from which we read the stereotypes. The %d is a placeholder.",
    )
    parser.add_argument(
        "--condition_column",
        type=str,
        default=CONDITION_COLUMN,
        help="The name of the column from which to read the experimental condition.",
    )
    parser.add_argument(
        "--num_stereotypes",
        type=int,
        default=NUM_STEREOTYPES,
        help="The maximum number of stereotypes each study participant can provide/ the number of stereotype columns we check for.",
    )
    parser.add_argument(
        "--excluded_stereotypes",
        type=list[str],
        default=EXCLUDED_STEREOTYPES,
        help="stereotypes which should be excluded from the analysis.",
    )
    parser.add_argument(
        "--language_model",
        type=str,
        default=LANGUAGE_MODEL,
        help="The language model to be used for word embeddings.",
    )
    parser.add_argument(
        "--show_plots",
        type=bool,
        default=SHOW_PLOTS,
        help="If this is True, the script will show intermediate plots to illustrate the outlier detection and the choice of K. If False, the plots won't be shown.",
    )
    parser.add_argument(
        "--outlier_k",
        type=int,
        default=OUTLIER_K,
        help="The number of neighbors to be considered for outlier detection.",
    )
    parser.add_argument(
        "--outlier_detection_threshold",
        type=int,
        default=OUTLIER_DETECTION_THRESHOLD,
        help="The detection threshold for outlier detection, in the sense that any point is considered an outlier where the average cosine similarity to its OUTLIER_K nearest neighbors is OUTLIER_DETECTION_THRESHOLD standard deviations below the average.",
    )
    parser.add_argument(
        "--max_num_clusters",
        type=int,
        default=MAX_NUM_CLUSTERS,
        help="the maximum number of clusters to be considered in step 3",
    )
    parser.add_argument(
        "--help_choose_num_clusters",
        type=bool,
        default=HELP_CHOOSE_NUM_CLUSTERS,
        help="Whether we want to run the auxiliary functions to help choose the NUM_CLUSTERS parameter",
    )
    parser.add_argument(
        "--num_clusters",
        type=list[str],
        default=[f"{v}" for v in NUM_CLUSTERS.values()],
        help="The number of clusters we use for clustering in step 4, defined separately for each condition.",
    )
    parser.add_argument(
        "--merge_threshold",
        type=float,
        default=MERGE_THRESHOLD,
        help="The cosine similarity threshold above which neighboring clusters get merged together. If this should not be done, set the threshold above 1",
    )
    parser.add_argument(
        "--data_output_file",
        type=str,
        default=DATA_OUTPUT_FILE,
        help="The path to the output CSV file to which we copy the input data plus the cluster index for each stereotype",
    )
    parser.add_argument(
        "--cluster_column",
        type=str,
        default=CLUSTER_COLUMN,
        help="The name of the columns to which we write the cluster index after the analysis is complete. The %d is a placeholder.",
    )
    parser.add_argument(
        "--clustering_output_file",
        type=str,
        default=CLUSTERING_OUTPUT_FILE,
        help="The path to the output CSV file to which we write the clustering of all unique words. The %s is a placeholder for the experimental condition.",
    )
    parser.add_argument(
        "--cluster_similarities_file",
        type=str,
        default=CLUSTER_SIMILARITIES_FILE,
        help="The path to the output CSV file to which we write the pairwise similarities between clusters. The %s is a placeholder for the experimental condition.",
    )
    # parser.print_help()
    args = parser.parse_args()
    main(args)
