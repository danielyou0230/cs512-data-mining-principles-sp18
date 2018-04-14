import argparse
import os
import numpy as np
import pandas as pd
from utils import readlines, write_to_file, save_sparse, load_sparse
from scipy.sparse import csr_matrix
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain
from tqdm import tqdm

# python src/vectorization.py data/cleaned_data.txt --output=training
# python src/vectorization.py data/cleaned_validation.txt --fit


def vectorization(file, output, vector_model=None, encoder_model=None,
                  venues_list=None, amount=100):
    """
    """
    if vector_model is None:
        vector_model = "model/vectorizer.pkl"
    if encoder_model is None:
        encoder_model = "model/encoder.pkl"
    if venues_list is None:
        venues_list = "data/labels.txt"

    h_vector_model = vector_model[:-4] + "_h" + vector_model[-4:]

    sampled_trainset = "text_features.txt"
    trainset = "data/trainset.tsv"

    # Load cleaned data
    data = readlines(file, delimiter="\t", lower=True)
    # Load venues list
    all_venues = readlines(venues_list, lower=True)
    all_venues = list(chain.from_iterable(all_venues))
    unique_venues = list(np.unique(all_venues))

    # Convert to pandas.DataFrame
    cols = ["ID", "TITLE", "VENUE", "CITE_PPRS", "CITE_VEN"]
    df = pd.DataFrame(data, columns=cols)

    ppr_id = df["ID"].values[:, np.newaxis]
    titles = df["TITLE"].values
    venues = df["VENUE"].values
    cite_v = df["CITE_VEN"].values

    # Fit Vectorization model (Normal)
    vectorizer = CountVectorizer(max_features=3000)
    vectorizer.fit_transform(titles)
    print("Save Vectorizer to file {0}".format(vector_model))
    joblib.dump(vectorizer, vector_model)

    # Fit Vectorization model (Heterogeneous)
    extra_space = np.chararray(cite_v.shape)
    extra_space[:] = " "

    h_raw = np.core.defchararray.add(titles, extra_space)
    h_raw = np.core.defchararray.add(h_raw, cite_v)
    h_vectorizer = CountVectorizer(max_features=3000)
    h_vectorizer.fit_transform()
    print("Save Heterogeneous Vectorizer to file {0}".format(h_vector_model))
    joblib.dump(h_vectorizer, h_vector_model)

    # Vectorization
    # titles_feature = [vectorizer.transform([itr]).toarray() for itr in titles]
    titles_feature = vectorizer.transform(titles).toarray()
    print(" - Feature dimension: {:4d}".format(titles_feature.shape[1]))
    print(titles_feature)

    # Label Encoder
    encoder = LabelEncoder()
    encoder.fit(unique_venues)
    print("Save LabelEncoder to file {0}".format(encoder_model))
    joblib.dump(encoder, encoder_model)

    # Convert to labels
    print("Transforming venues to labels...")
    venues_feature = encoder.transform(venues)[:, np.newaxis]
    # Sample a subset of data for assignments
    # convert title feature vector to strings
    # output_ttl = [", ".join([str(e) for e in titles_feature[itr,:]])
    #               for itr in range(amount)]
    print("Creating output array...")
    output_ttl = list()
    for itr in tqdm(range(len(venues))):
        index = np.where(titles_feature[itr,:] == 1)[0]
        arr = ", ".join([str(itr) for itr in index])
        output_ttl.append(arr)
    output_ttl = np.array(output_ttl)[:, np.newaxis]

    # merge the feature vector and labels
    # output_arr = np.hstack([output_ttl, venues_feature[:amount]])
    n_bow = len(vectorizer.vocabulary_)
    info = np.array([ppr_id.shape[0], n_bow, encoder.classes_.shape[0]])
    info = info[np.newaxis,:]
    #
    output_arr = np.hstack([ppr_id, output_ttl, venues_feature])
    output_arr = np.vstack([info, output_arr])

    output_col = ["id", "feature_vector", "encoded_labels"]
    # Convert to pandas.DataFrame
    output_df = pd.DataFrame(output_arr, columns=output_col)

    # print("Saving sampled training set to {0}\n".format(sampled_trainset))
    print("Saving sampled training set to {0}\n".format(trainset))
    # output_df.to_csv(sampled_trainset, sep='\t', index=False, header=False)
    output_df.to_csv(trainset, sep='\t', index=False, header=False)

    # Save to file
    # print("Saving data and labels to npy")
    # np.save("data/{:s}_text.npy".format(output), titles_feature)
    # np.save("data/{:s}_label.npy".format(output), venues_feature)
    # write_to_file(output, feature)

def fit_encoder(file, output, vector_model, encoder_model):
    """
    """
    if vector_model is None:
        vector_model = "model/vectorizer.pkl"
    if encoder_model is None:
        encoder_model = "model/encoder.pkl"
    testset = "data/testset.tsv"
    h_vector_model = vector_model[:-4] + "_h" + vector_model[-4:]

    # Load cleaned data
    data = readlines(file, delimiter="\t", lower=True)
    # Convert to pandas.DataFrame
    cols = ["ID", "TITLE", "VENUE", "CITE_PPRS", "CITE_VEN"]
    df = pd.DataFrame(data, columns=cols)

    ppr_id = df["ID"].values[:, np.newaxis]
    titles = df["TITLE"].values
    venues = df["VENUE"].values
    cite_v = df["CITE_VEN"].values

    # Load Vectorizer
    print("Loading Vectorizer from {0}".format(vector_model))
    vectorizer = joblib.load(vector_model)
    # Vectorization
    print("Vectorizing text features...")
    # titles_feature = [vectorizer.transform([itr]).toarray() for itr in titles]
    titles_feature = vectorizer.transform(titles).toarray()
    
    # Fit Vectorization model (Heterogeneous)
    extra_space = np.chararray(cite_v.shape)
    extra_space[:] = " "

    h_feature = np.core.defchararray.add(titles, extra_space)
    h_feature = np.core.defchararray.add(h_feature, cite_v)
    h_vectorizer = CountVectorizer(max_features=3000)
    h_vectorizer.fit_transform()
    print("Save Heterogeneous Vectorizer to file {0}".format(h_vector_model))
    joblib.dump(h_vectorizer, h_vector_model)

    # Load LabelEncoder
    print("Loading LabelEncoder from {0}".format(encoder_model))
    encoder = joblib.load(encoder_model)
    # Convert to labels
    print("Encoding venues...\n")
    venues_feature = encoder.transform(venues)[:, np.newaxis]

    print("Creating output array...")
    output_ttl = list()
    for itr in tqdm(range(len(venues))):
        index = np.where(titles_feature[itr,:] == 1)[0]
        arr = ", ".join([str(itr) for itr in index])
        output_ttl.append(arr)
    output_ttl = np.array(output_ttl)[:, np.newaxis]

    # merge the feature vector and labels
    # output_arr = np.hstack([output_ttl, venues_feature[:amount]])
    n_bow = len(vectorizer.vocabulary_)
    info = np.array([ppr_id.shape[0], n_bow, encoder.classes_.shape[0]])
    info = info[np.newaxis,:]
    #
    output_arr = np.hstack([ppr_id, output_ttl, venues_feature])
    output_arr = np.vstack([info, output_arr])

    output_col = ["id", "feature_vector", "encoded_labels"]
    # Convert to pandas.DataFrame
    output_df = pd.DataFrame(output_arr, columns=output_col)

    # print("Saving sampled training set to {0}\n".format(sampled_trainset))
    print("Saving dataset to {0}\n".format(testset))
    # output_df.to_csv(sampled_trainset, sep='\t', index=False, header=False)
    output_df.to_csv(testset, sep='\t', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File to be parsed.")
    parser.add_argument("--output", help="Output filename.")
    parser.add_argument("--venues", help="Contains all venues.")
    parser.add_argument("--vector", help="LabelEncoder filename.")
    parser.add_argument("--encoder", help="LabelEncoder filename.")
    parser.add_argument("--fit", action="store_true",
                        help="Fit input with the given LabelEncoder.")
    parser.add_argument("--amount", nargs="?", type=int, default=100,
                        help="Amount of data to be sample.")

    args = parser.parse_args()

    # Create model save path
    if not os.path.exists("model/"):
        os.makedirs("model/")

    if args.fit:
        fit_encoder(args.file, args.output, args.vector, args.encoder)
    else:
        vectorization(args.file, args.output, args.vector, args.encoder,
                      args.venues, args.amount)