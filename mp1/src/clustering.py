import argparse
from nltk.cluster.util import cosine_distance
from nltk.cluster.kmeans import KMeansClusterer
import numpy as np
from multiprocessing import Pool

# python clustering.py phrase_YELP.emb --n_cluster 5 10 25 --repeats 5
# python clustering.py phrase_DBLP.emb --n_cluster 5 10 25 --repeats 5
vectors = None
repeats = None
names = None

def nltk_clustering(n, filename):
    global vectors
    global names
    global repeats
    # Clustering
    print("Begin clustering, n = {:d}...".format(n))

    clusterer = KMeansClusterer(n, cosine_distance, repeats=repeats)
    clustered = clusterer.cluster(vectors, assign_clusters=True, trace=False)
    clustered = np.array(clustered)

    index = sorted(clustered)
    # print(clustered.argsort())
    names = list(names[clustered.argsort()])

    # write result to file
    print("Saving result to file...")
    output = filename[:-4] + "_" + str(n) + "_clustered.txt"
    with open(output, "w") as f:
        current_idx = None
        for itr, idx in zip(names, index):
            if current_idx != idx:
                current_idx = idx
                f.write("\nCluster {:d} (description: )\n".format(current_idx))
            else:
                pass
            f.write(itr + "\n")
    #
    print("Clustered result saved in {0}".format(output))

def kmeans_clustering_cos(args):
    global vectors
    global names
    global repeats
    repeats = args.repeats
    # load word2vec vectors
    print("Loading word2vec vectors from {:s}".format(args.input))
    with open(args.input, "r") as f:
        data = f.read().splitlines()[1:]
        names = np.array([itr.split()[0] for itr in data])
        vectors = np.array([itr.split()[1:] for itr in data], dtype=np.float)
    print("Number of vectors: {:d}".format(vectors.shape[0]))
    print("Dimension of vectors: {:d}".format(vectors.shape[1]))
    #
    n_threads = len(args.n_cluster)

    #
    param = [(n, filename) for n, filename in \
             zip(args.n_cluster, [args.input] * n_threads)]
    with Pool(processes=n_threads) as p:
        p.starmap(nltk_clustering, param)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input file to be clustered.")
    parser.add_argument("-n", "--n_cluster", nargs='+', type=int, help="Number of clusters.")
    parser.add_argument("-r", "--repeats", type=int, help="Number of clusters.")
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="verbose ouptut")
    args = parser.parse_args()

    kmeans_clustering_cos(args)