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
# python src/vectorization.py data/cleaned_data.txt --output=training --hin
# python src/vectorization.py data/cleaned_validation.txt --fit
# python src/vectorization.py data/cleaned_validation.txt --fit --hin


def generate_features(data, hin=False):
    """
    Generate features according to given specs.

    Args:
        data(ndarray): Data contains ID, TITLE, VENUE, CITE_PPRS, CITE_VEN
        hin(bool): Whether to use heterogeneous features (optional).

    Returns:
        ppr_id(ndarray): Paper IDs.
        context(ndarray): Titles.
        venues(ndarray): Venues.
    """

    # Convert to pandas.DataFrame
    cols = ["ID", "TITLE", "VENUE", "CITE_PPRS", "CITE_VEN"]
    df = pd.DataFrame(data, columns=cols)

    ppr_id = df["ID"].values[:, np.newaxis]
    titles = df["TITLE"].values
    venues = df["VENUE"].values
    cite_v = df["CITE_VEN"].values

    # Fit Vectorization model (Normal)
    if hin:
        context = list()
        # ***CAUSED MEMORY ERROR***
        # Fit Vectorization model (Heterogeneous)
        # Fill in spaces in each column (concatenate extra features)
        # extra_space = np.chararray(cite_v.shape)
        # extra_space[:] = " "
        # context = np.core.defchararray.add(titles, extra_space)
        # context = np.core.defchararray.add(context, cite_v)

        # Use for loop to 
        for idx in range(titles.shape[0]):
            # titles[idx] += " " + cite_v[idx]
            context.append(titles[idx] + " " + cite_v[idx])

    else:
        context = titles

    return ppr_id, np.array(context), venues


def save_features(ppr_id, features, venues_feature, n_bow, n_venues, save_name):
    """
    Save features to .csv file.

    Args:
        ppr_id(ndarray): Paper IDs.
        features(ndarray): Paper titles features.
        venues_feature(ndarray): Paper venues features.
        n_bow(int): Number of bag-of-words (word space).
        n_venues(int): Number of venus (venue space).
        save_name(str): Save filename.
    """

    # Sample a subset of data for assignments
    # Convert title feature vector to strings
    print("Creating output array...")
    output_context = list()
    for itr in tqdm(range(features.shape[0])):
        index = np.where(features[itr,:] == 1)[0]
        arr = ", ".join([str(itr) for itr in index])
        output_context.append(arr)
    output_context = np.array(output_context)[:, np.newaxis]

    # Merge the feature vector and labels
    # output_arr = np.hstack([output_context, venues_feature[:amount]])
    output_arr = np.hstack([ppr_id, output_context, venues_feature])
    # Extra information in header
    info = np.array([ppr_id.shape[0], n_bow, n_venues])[np.newaxis,:]
    # Add extra information to output array
    output_arr = np.vstack([info, output_arr])

    # Convert to pandas.DataFrame
    output_col = ["id", "feature_vector", "encoded_labels"]
    output_df = pd.DataFrame(output_arr, columns=output_col)

    # print("Saving sampled training set to {0}\n".format(sampled_trainset))
    print("Saving transformed featrues to {0}\n".format(save_name))
    # output_df.to_csv(sampled_trainset, sep='\t', index=False, header=False)
    output_df.to_csv(save_name, sep='\t', index=False, header=False)


def feature_transform(file, output, hin=False, vector_model=None,
                      encoder_model=None, venues_file=None, amount=100):
    """
    Train models and transform given corpus to feature vectors.

    Args:
        file(str): Input corpus filename.
        output(str): Output filename.
        hin(bool): Use heterogeneous features if asserted.
        vector_model(str): Filename for vectorizer model.
        encoder_model(str): Filename for encoder model.
        venues_file(str): File containing all possible venues.
        amount(int): Number of instance to be sampled for demo.
    """

    if vector_model is None:
        vector_model = "model/{0}vectorizer.pkl".format("h_" if hin else "")
    if encoder_model is None:
        encoder_model = "model/{0}encoder.pkl".format("h_" if hin else "")
    testset = "data/{0}testset.tsv".format("h_" if hin else "")
    if venues_file is None:
        venues_file = "data/labels.txt"
    print("Using {0}geneous features".format("Hetero" if hin else "Homo"))

    sampled_trainset = "{0}text_features.txt".format("h_" if hin else "")
    trainset = "data/{0}trainset.tsv".format("h_" if hin else "")

    # Load cleaned data
    data = readlines(file, delimiter="\t", lower=True)
    # Load venues list
    all_venues = readlines(venues_file, lower=True)
    all_venues = list(chain.from_iterable(all_venues))
    unique_venues = list(np.unique(all_venues))

    # Acquire features in dataset
    ppr_id, context, venues = generate_features(data, hin=hin)

    # Fit Vectorization model (Normal/Heterogeneous)
    vectorizer = CountVectorizer(max_features=3000)
    vectorizer.fit_transform(context)
    print("Save Vectorizer to file {0}".format(vector_model))
    joblib.dump(vectorizer, vector_model)

    # Vectorization
    # titles_feature = [vectorizer.transform([itr]).toarray() for itr in context]
    context_feature = vectorizer.transform(context).toarray()
    print(" - Feature dimension: {:4d}".format(context_feature.shape[1]))

    # Label Encoder
    encoder = LabelEncoder()
    encoder.fit(unique_venues)
    print("Save LabelEncoder to file {0}\n".format(encoder_model))
    joblib.dump(encoder, encoder_model)

    # Convert to labels
    print("Transforming venues to labels...")
    venues_feature = encoder.transform(venues)[:, np.newaxis]

    # Some useful figures
    n_bow = len(vectorizer.vocabulary_)
    n_venues = encoder.classes_.shape[0]
    # Save features to file
    save_features(ppr_id, context_feature, venues_feature, n_bow, n_venues, trainset)


def fit_encoder(file, output, hin, vector_model, encoder_model):
    """
    Transform given corpus to feature vectors with given models.

    Args:
        file(str): Input corpus filename.
        output(str): Output filename.
        hin(bool): Use heterogeneous features if asserted.
        vector_model(str): Filename for vectorizer model.
        encoder_model(str): Filename for encoder model.
    """

    if vector_model is None:
        vector_model = "model/{0}vectorizer.pkl".format("h_" if hin else "")
    if encoder_model is None:
        encoder_model = "model/{0}encoder.pkl".format("h_" if hin else "")
    testset = "data/{0}testset.tsv".format("h_" if hin else "")

    print("Using {0}geneous features".format("Hetero" if hin else "Homo"))

    # Load cleaned data
    data = readlines(file, delimiter="\t", lower=True)

    # Acquire features in dataset
    ppr_id, context, venues = generate_features(data, hin=hin)

    # Load Vectorizer
    print("Loading Vectorizer from {0}".format(vector_model))
    vectorizer = joblib.load(vector_model)
    # Vectorization
    print("Vectorizing text features...")
    # context_feature = [vectorizer.transform([itr]).toarray() for itr in context]
    context_feature = vectorizer.transform(context).toarray()

    # Load LabelEncoder
    print("Loading LabelEncoder from {0}".format(encoder_model))
    encoder = joblib.load(encoder_model)
    # Convert to labels
    print("Encoding venues...\n")
    venues_feature = encoder.transform(venues)[:, np.newaxis]

    # Some useful figures
    n_bow = len(vectorizer.vocabulary_)
    n_venues = encoder.classes_.shape[0]
    # Save features to file
    save_features(ppr_id, context_feature, venues_feature, n_bow, n_venues, testset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File to be parsed.")
    parser.add_argument("--output", help="Output filename.")
    parser.add_argument("--hin", action="store_true", help="Use heterogeneous features.")
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
        fit_encoder(args.file, args.output, args.hin, args.vector, args.encoder)
    else:
        feature_transform(args.file, args.output, args.hin, args.vector,
                          args.encoder, args.venues, args.amount)