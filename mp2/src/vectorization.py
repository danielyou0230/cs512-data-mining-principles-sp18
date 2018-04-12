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


# python src/vectorization.py data/cleaned_data.txt \
# --output=data/text_features.txt --amount=100
# python src/vectorization.py data/cleaned_validation.txt \
# --output=data/text_features_val.txt --fit


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

    sampled_trainset = "text_features.txt"

    # Load cleaned data
    data = readlines(file, delimiter="\t", lower=True)
    # Load venues list
    all_venues = readlines(venues_list, lower=True)
    all_venues = list(chain.from_iterable(all_venues))
    unique_venues = list(np.unique(all_venues))

    # Convert to pandas.DataFrame
    cols = ["ID", "TITLE", "VENUE", "CITE_PPRS", "CITE_VEN"]
    df = pd.DataFrame(data, columns=cols)

    titles = df["TITLE"].as_matrix()
    venues = df["VENUE"].as_matrix()

    # Fit Vectorization model
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(titles)
    print("Save Vectorizer to file {0}".format(vector_model))
    joblib.dump(vectorizer, vector_model)

    # Vectorization
    # titles_feature = [vectorizer.transform([itr]).toarray() for itr in titles]
    titles_feature = vectorizer.transform(titles).toarray()
    print(titles_feature[0, :])

    # Label Encoder
    encoder = LabelEncoder()
    encoder.fit(unique_venues)
    print("Save LabelEncoder to file {0}".format(encoder_model))
    joblib.dump(encoder, encoder_model)

    # Convert to labels
    venues_feature = encoder.transform(venues)[:, np.newaxis]

    # Sample a subset of data for assignments
    # convert title feature vector to strings
    output_ttl = [", ".join([str(e) for e in titles_feature[itr,:]])
                  for itr in range(amount)]
    output_ttl = np.array(output_ttl)[:, np.newaxis]
    # merge the feature vector and labels
    output_arr = np.hstack([output_ttl, venues_feature[:amount]])
    output_col = ["feature_vector", "encoded_labels"]
    # Convert to pandas.DataFrame
    output_df = pd.DataFrame(output_arr, columns=output_col)

    print("Saving sampled training set to {0}".format(sampled_trainset))
    output_df.to_csv(sampled_trainset, sep='\t', index=False, header=False)
    # Save to file
    # write_to_file(output, feature)

def fit_encoder(file, output, vector_model, encoder_model):
    """
    """
    if vector_model is None:
        vector_model = "model/vectorizer.pkl"
    if encoder_model is None:
        encoder_model = "model/encoder.pkl"

    # Load cleaned data
    data = readlines(file, delimiter="\t", lower=True)
    # Convert to pandas.DataFrame
    cols = ["ID", "TITLE", "VENUE", "CITE_PPRS", "CITE_VEN"]
    df = pd.DataFrame(data, columns=cols)

    titles = df["TITLE"].as_matrix()
    venues = df["VENUE"].as_matrix()

    # Load Vectorizer
    print("Loading Vectorizer from {0}".format(vector_model))
    vectorizer = joblib.load(vector_model)
    # Vectorization
    print("Vectorizing text features...")
    titles_feature = [vectorizer.transform([itr]).toarray() for itr in titles]

    # Load LabelEncoder
    print("Loading LabelEncoder from {0}".format(encoder_model))
    encoder = joblib.load(encoder_model)
    # Convert to labels
    print("Encoding venues...")
    venues_feature = encoder.transform(venues)

    # Save to file
    # write_to_file(output, feature)


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