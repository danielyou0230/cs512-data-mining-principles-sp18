import argparse
import os
import numpy as np
import pandas as pd
from utils import readlines, write_to_file
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain


# python src/vectorization.py data/cleaned_data.txt --output=data/text_features.txt
# python src/vectorization.py data/cleaned_validation.txt --output=data/text_features_val.txt --fit


def vectorization(file, output, vector_model=None, encoder_model=None):
    """
    """
    if vector_model is None:
        vector_model = "model/vectorizer.pkl"
    if encoder_model is None:
        encoder_model = "model/encoder.pkl"

    # Load cleaned data
    data = readlines(file, delimiter="\t", lower=True)
    titles = [itr[1] for itr in data]
    # venues = [itr[2] for itr in data]
    venues = list()
    unique_venues = list(np.unique(venues))

    # Load venues list
    # venue_list = readlines(file, delimiter="\t", lower=True)
    # venue_list = list(chain.from_iterable(venue_list))

    # Fit Vectorization model
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(titles)
    print("Save Vectorizer to file {0}".format(vector_model))
    joblib.dump(vectorizer, vector_model) 

    # Vectorization
    titles_feature = [vectorizer.transform([itr]).toarray() for itr in titles]

    # Label Encoder
    encoder = LabelEncoder()
    encoder.fit(unique_venues)
    print("Save LabelEncoder to file {0}".format(encoder_model))
    joblib.dump(encoder, encoder_model) 

    # Convert to labels
    venues_feature = encoder.transform(venues)
    # print(venues_feature)

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
    titles = [itr[1] for itr in data]
    venues = [itr[2] for itr in data]
    unique_venues = list(np.unique(venues))

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
    parser.add_argument("--vector", help="LabelEncoder filename.")
    parser.add_argument("--encoder", help="LabelEncoder filename.")
    parser.add_argument("--fit", action="store_true",
                        help="Fit input with the given LabelEncoder.")

    args = parser.parse_args()

    # Create model save path
    if not os.path.exists("model/"):
        os.makedirs("model/")

    if args.fit:
        fit_encoder(args.file, args.output, args.vector, args.encoder)
    else:
        vectorization(args.file, args.output, args.vector, args.encoder)