#!python
"""
This Python script is intended as starting point for work by the
Unicorn Lab at Purdue University to cluster stereotypes using word
embeddings.

The scheme is adapted from Nicolas et al. (2022): Spontaneous Stereotype
Content Model: Taxonomy, Properties, and Prediction. Journal of Personality
and Social Psychology, 123(5), 1243-1263. doi:10.1037/pspa0000312

Original author: Benjamin Paassen - Bielefeld University
Version 2023-09-30

The rough steps of processing are:
1.  We load data from the CSV file DATA_INPUT_FILE, where each participant
   in the study is represented by one row and their answers by columns.
   In particular, We extract the stereotypes from the columns
   STEREOTYPE_COLUMN.
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


__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright (C) 2023, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '0.9.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'



import word_embedding_clustering_functions
from collections import Counter
import csv
import numpy as np
from sentence_transformers import SentenceTransformer

# Here, we define our parameters. We do that at the start of the file
# to make the parameters easier to change if we need to.

# The path to the input CSV file from which we read all the stereotypes
# that participants answered.
DATA_INPUT_FILE   = 'test_input_file.csv'
# The character that is used to delimit columns in the input (and output)
# file
COL_DELIMITER     = ','
# The name of the columns from which we read the stereotypes. The %d
# is a placeholder.
STEREOTYPE_COLUMN = 'Stereotype%d'
# The maximum number of stereotypes each study participant can provide/
# the number of stereotype columns we check for
NUM_STEREOTYPES   = 1
# stereotypes which should be excluded from the analysis.
EXCLUDED_STEREOTYPES = ['', 'white']
# The language model to be used for word embeddings
LANGUAGE_MODEL    = 'BAAI/bge-large-en-v1.5'
# If this is True, the script will show intermediate plots to illustrate
# the outlier detection and the choice of K. If False, the plots won't be
# shown.
SHOW_PLOTS = True
# The number of neighbors to be considered for outlier detection
OUTLIER_K = 3
# The detection threshold for outlier detection, in the sense that any
# point is considered an outlier where the average cosine similarity
# to its OUTLIER_K nearest neighbors is OUTLIER_DETECTION_THRESHOLD
# standard deviations below the average
OUTLIER_DETECTION_THRESHOLD = 3
# the maximum number of clusters to be considered in step 3
MAX_NUM_CLUSTERS  = 15
# Whether we want to run the auxiliary functions to help choose the
# NUM_CLUSTERS parameter
HELP_CHOOSE_NUM_CLUSTERS = True
# The number of clusters we use for clustering in step 4.
# NOTE: This script also contains auxiliary functions to help with choosing
# this parameter.
# If HELP_CHOOSE_NUM_CLUSTERS is True, this parameter may be overriden
# by the user when executing the script.
NUM_CLUSTERS = 3
# The cosine similarity threshold above which neighboring clusters
# get merged together. If this should not be done, set the threshold
# above 1
MERGE_THRESHOLD = 0.95
# The path to the output CSV file to which we copy the input data plus
# the cluster index for each stereotype
DATA_OUTPUT_FILE  = 'test_output_file.csv'
# The name of the columns to which we write the cluster index after
# the analysis is complete. The %d is a placeholder.
CLUSTER_COLUMN    = 'Stereotype%d_code_match'
# The path to the output CSV file to which we write the clustering of
# all unique words.
CLUSTERING_OUTPUT_FILE = 'clustering.csv'
# The path to the output CSV file to which we write the pairwise
# similarities between clusters.
CLUSTER_SIMILARITIES_FILE = 'cluster_similarities.csv'


# STEP 1: OPEN THE INPUT DATA FILE AND READ ALL STEREOTYPES
print('Starting step 1 of 6: Reading the input data from file \"%s\"' % DATA_INPUT_FILE)

# we keep track of the data itself because need to copy it
# to the output later
rows = []
# This counter stores which stereotypes users have named and how often
stereotype_counts = Counter()
# now we open the input data file in UTF-8 encoding
with open(DATA_INPUT_FILE, encoding = 'utf-8') as f:
  # we put the input file into a special reader object for CSV files
  # that makes it easier to read CSV tables, row by row.
  reader  = csv.reader(f, delimiter = COL_DELIMITER)
  # we read the first row of the input file, which should contain
  # all the column headers
  headers = reader.__next__()
  # we start a list to store the indices for all columns that may
  # contain stereotypes
  col_idxs = []
  # we iterate over the numbers from 1 to the number of stereotypes
  for i in range(1, NUM_STEREOTYPES + 1):
    # we set up the column name for the i-th stereotype by filling
    # in our %d placeholder
    stereotype_column_name = STEREOTYPE_COLUMN % i
    # we identify the column index
    j = headers.index(stereotype_column_name)
    # .. and store it
    col_idxs.append(j)
  # retrieve the column indices for the output columns from the header of the
  # input table
  out_col_idxs = []
  # we iterate over the numbers from 1 to the number of stereotypes
  for i in range(1, NUM_STEREOTYPES + 1):
    # we set up the column name for the i-th stereotype by filling
    # in our %d placeholder
    cluster_column_name = CLUSTER_COLUMN % i
    # we identify the column index
    j = headers.index(cluster_column_name)
    # .. and store it
    out_col_idxs.append(j)
  # we iterate over all remaining rows of the input file. Recall:
  # each row represents the answers from one participant
  for row in reader:
    # store the current row
    rows.append(row)
    # check if we have already initialized
    # iterate over all stereotype columns
    for j in col_idxs:
      # get the next stereotype provided by the current participant
      stereotype = row[j]
      # if the stereotype is in the list of forbidden words, ignore it
      if stereotype.strip() in EXCLUDED_STEREOTYPES:
        continue
      # otherwise, count the stereotype
      stereotype_counts[stereotype] += 1

# At this point, the input data file is closed and 
# 'condition_to_stereotype_counts' contains, for each condition,
# all the stereotypes that have been named and how often they
# have been named. We report some statistics to the user
print('Completed step 1. Read %d responses.' % len(rows))




# We initialize the language model
print('Prepating step 2 of 6 by initializing the language model %s. This may take a while when this script is run the first time.' % LANGUAGE_MODEL)
model = SentenceTransformer(LANGUAGE_MODEL)

no_of_unique_stereotypes = len(stereotype_counts)
no_of_stereotypes        = sum(stereotype_counts.values())
print('Starting analysis with %d unique stereotypes (%d stereotypes overall)' % (no_of_unique_stereotypes, no_of_stereotypes))

# get an ordered list of all stereotypes. 
stereotypes = list(stereotype_counts.keys())

# STEP 2. EMBED ALL STEREOTYPES USING A LANGUAGE MODEL
print('Starting step 2 of 6: Generating word embeddings.')

# Then, we call the encoding function. This will generate a matrix of
# word embeddings with one row per stereotype and as many columns as there
# are embedding dimensions (e.g. 2014). The content of the matrix will be
# floating point numbers. These numbers have no meaning by themselves, but
# words which occur in similar contexts in the English language will have
# similar numbers - similar in the sense that the cosine similarity between
# two words that occur in similar contexts will be higher compared to
# two words that occur in dissimilar contexts.
# Note: Cosine similarity is also the similarity measure that
# Nicolas et al. (2022) suggest and is, indeed, a standard tool when
# working with word embeddings.
print('Embedding %d unique stereotypes. This may take a few seconds.' % len(stereotypes))
embeddings = model.encode(stereotypes)

# To prepare further processing, we do one more step, namely normalizing
# the length of each embedding vector. This step is taken to ensure that
# we only consider the angle between vectors (i.e. their cosine similarity)
# and not their length, which may be distorted.

# compute the (Euclidean) length of each embedding vector
Z = np.sqrt(np.sum(embeddings ** 2, 1))
# divide each embedding vector by its length
embeddings_normalized = embeddings / np.expand_dims(Z, 1)

print('Filtering out words that are outliers, in the sense that their average cosine similarity to the %d nearest neighbors is at least %g standard deviations below the average.' % (OUTLIER_K, OUTLIER_DETECTION_THRESHOLD))

stereotypes, embeddings_normalized = word_embedding_clustering_functions.outlier_detection(stereotypes, embeddings_normalized, OUTLIER_K, OUTLIER_DETECTION_THRESHOLD, SHOW_PLOTS = SHOW_PLOTS)

print('completed outlier detection. %d unique stereotypes are remaining.' % len(stereotypes))

# for the convenience in subsequent processing, we set up
# two further data structures.
# first, a dictionary that maps from a stereotype to its index in the
# stereotype list. We will need that later to map back from stereotypes
# to their cluster index.
stereotype_idx = {}
for i in range(len(stereotypes)):
  stereotype_idx[stereotypes[i]] = i
# second, a list of how often each stereotype was named
sample_weights = []
for stereotype in stereotypes:
  sample_weights.append(stereotype_counts[stereotype])
sample_weights = np.array(sample_weights)

print('Completed step 2. Got an embedding matrix of size %d x %d' % embeddings_normalized.shape)

if not HELP_CHOOSE_NUM_CLUSTERS:
  print('Skipping step 3 of 6.')
  K = NUM_CLUSTERS
else:
  # STEP 3: HELP TO CHOOSE THE OPTIMAL NUMBER OF CLUSTERS
  print('Starting step 3 of 6: trying to estimate the optimal number of clusters by repeating K-Means clustering for different K and computing quality measures for each K. This may take several minutes for large K.')

  K = word_embedding_clustering_functions.find_number_of_clusters(stereotypes, embeddings_normalized, MAX_NUM_CLUSTERS, sample_weights = sample_weights, SHOW_PLOTS = SHOW_PLOTS)

  print('completed step 3 of 6 with chosen number of clusters: %d' % K)

# STEP 4: KMEANS CLUSTERING
print('Starting step 4 of 6: Perform K-Means Clustering with %d clusters.' % K)

cluster_idxs, centers_normalized = word_embedding_clustering_functions.cluster_and_merge(stereotypes, embeddings_normalized, K, sample_weights = sample_weights, MERGE_THRESHOLD = MERGE_THRESHOLD, SHOW_PLOTS = SHOW_PLOTS)

K_new = centers_normalized.shape[0]
if K_new < K:
  print('reduced number of clusters by merging to %d' % K_new)

print('Completed step 4 of 6.')

# STEP 5: WRITE CLUSTERING TO OUTPUT
print('Starting step 5 of 6: Writing clustering to the file \"%s\"' % CLUSTERING_OUTPUT_FILE)

# We write a table with three columns: the first for every
# unique stereotype, the second for the cluster index of that stereotype,
# and the third for the distance of the stereotype's embedding to the cluster
# mean, which indicates how representative this stereotype is for the cluster.
with open(CLUSTERING_OUTPUT_FILE, 'w', encoding = 'utf-8') as f:
  # set up a CSV writer to make the output writing easier
  writer = csv.writer(f, delimiter = COL_DELIMITER)
  # write a header
  writer.writerow(['stereotype', 'cluster_index', 'similarity_to_center'])
  # iterate over all clusters
  for k in range(K):
    # get the indices of all stereotypes in cluster k
    in_cluster_k = np.where(cluster_idxs == k)[0]
    # if the cluster is empty, ignore it
    if len(in_cluster_k) == 0:
      continue
    # compute the cosine similarity of the embeddings of all stereotypes
    # in cluster k to the mean of cluster k
    sim = np.dot(embeddings_normalized[in_cluster_k, :], centers_normalized[k, :])
    # iterate over all stereotypes in cluster k - but sort descendingly
    # by the cosine similarity because we may want to label clusters by
    # the most similar stereotypes
    for i in np.argsort(-sim):
      j = in_cluster_k[i]
      # get the ith unique stereotype
      stereotype = stereotypes[j]
      # get its cluster index
      k = cluster_idxs[j]
      # get its similarity to the cluster center
      s = sim[i]
      # write a row to the output table
      writer.writerow([stereotype, k, s])

# compute the pairwise similarities between all cluster centers
S = np.dot(centers_normalized, centers_normalized.T)
# write them to the desired output file
print('Writing pairwise cluster center similarities to file %s.' % CLUSTER_SIMILARITIES_FILE)
np.savetxt(CLUSTER_SIMILARITIES_FILE, S, fmt = '%.2f', delimiter = COL_DELIMITER)

print('Completed step 5 of 6.')

# STEP 6: WRITE CLUSTER INDICES BACK
print('Preparing step 6 of 6 by writing cluster indices to data')

# iterate over all data
for row in rows:

  # iterate over the number of stereotype columns
  for i in range(NUM_STEREOTYPES):
    # get the next stereotype provided by the current participant
    stereotype = row[col_idxs[i]]
    # get the index of the stereotype (if it exists)
    j = stereotype_idx.get(stereotype)
    # if this index does not exist (e.g. because the stereotype is
    # an outlier) just write an empty value
    if j is None:
      row[out_col_idxs[i]] = ''
      continue
    # otherwise, get the cluster index for the stereotype index
    k = cluster_idxs[j]
    # write it to the corresponding output column
    row[out_col_idxs[i]] = k

print('Starting step 6 of 6: Writing cluster indices for each stereotype to the file \"%s\"' % DATA_OUTPUT_FILE)
# At this point, we have written the correct cluster index at each
# desired location of the input table. Time to write the data to the
# output file
with open(DATA_OUTPUT_FILE, 'w', encoding = 'utf-8') as f:
  # set up a CSV writer to make the output writing easier
  writer = csv.writer(f, delimiter = COL_DELIMITER)
  # copy the header from the input file
  writer.writerow(headers)
  # write all data rows to the output
  for row in rows:
    writer.writerow(row)

print('Completed step 6 of 6. End of script.')
