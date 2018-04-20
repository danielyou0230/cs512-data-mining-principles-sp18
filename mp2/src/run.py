import argparse
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from tqdm import tqdm

# python src/run.py --model=model/clf.pkl --train=data/trainset.tsv
# python src/run.py --model=model/clf.pkl --train=data/h_trainset.tsv

# python src/run.py --model=model/clf.pkl --train=data/trainset.tsv --test=data/testset.tsv
# python src/run.py --model=model/clf.pkl --train=data/h_trainset.tsv --test=data/h_testset.tsv

# python src/run.py --model=model/clf.pkl --test=data/testset.tsv --predict
# python src/run.py --model=model/clf.pkl --test=data/h_testset.tsv --predict


def read_dataset(file):
    """
    Read dataset from .csv file.

    Args:
        file(str): Path to the csv file.

    Return:
        X_train(ndarray): Training instances.
        y_train(ndarray): Training labels.
        id_data(ndarray): Training content id.
    """

    print("Loading training data from {0}".format(file))
    cols = ["id", "feature_vector", "encoded_labels"]
    df = pd.read_csv(file, delimiter="\t", header=None, names=cols)

    # Some useful information
    id_data = df["id"].values
    n_data = int(id_data[0])
    print(" - Total number of data: {:8d}\n".format(n_data))

    n_bow = int(df["feature_vector"].values[0])
    n_class = int(df["encoded_labels"].values[0])

    # Expand Training feature vectors and labels 
    X_train = expand_dense_matrix(df["feature_vector"].values[1:], n_bow)
    # y_train = expand_dense_matrix(df["encoded_labels"].as_matrix()[1:], n_class)
    y_train = df["encoded_labels"].astype('int64')[1:].values
    del df

    return X_train, y_train, id_data


def expand_dense_matrix(dense, size):
    """
    Expand a given dense matrix to a sparse matrix.

    Args:
        dense(ndarray of str): Dense matrix with each entry defines the corrsponding
                               row to be a one.
        size(int): Size of the sparse matrix.
    
    Return:
        sparse(ndarray of int): Sparse matrix with one at elements defined in dense.
    """

    # Initialize sparse matrix
    sparse = np.zeros((len(dense), size))

    # Expand dense matrix to sparse matrix
    print("Expanding dense matrix...")
    for idx, itr in tqdm(enumerate(dense)):
        # Skip outliers
        if type(itr) is not str:
            continue
        # Extract the indices in the entry
        indices = [int(i) for i in itr.split(", ")]
        # Fill ones according to the extracted indices
        sparse[idx, :][indices] = 1

    return sparse


def eval_f1(clf, X, y_true):
    """
    Fit the input data with given classifier (clf) and evaluate F1 score
    on micro, macro, and weighted scores.

    Args:
        clf(sklearn clf object): Trained model.
        X(ndarray): Evaluating instances.
        y_true(ndarray): Evaluating labels.

    Return:
        y_pred(ndarray): Prediction given the clf and training instances.
    """

    # Predict on given data
    y_pred = clf.predict(X)

    # Calculating some metrics
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    print("Evaluation on training set:")
    print("F1-micro: {:2.5f} | F1-macro: {:2.5f} | F1-weighted: {:2.5f}"
          .format(f1_micro, f1_macro, f1_weighted))

    return y_pred


def run(train_file, test_file=None, model_file=None):
    """
    Train the classifier with given training data and self-evaluate
    on training set, if test_file is given, then it will also evaluate
    on the testing set after self-evaluation on training set.

    Args:
        train_file(str): Path to the training dataset.
        test_file(str): Path to the testing dataset (optional).
        model_file(str): Filename of the model to be saved.
    """

    # Load dataset
    X_train, y_train, _ = read_dataset(train_file)

    print("Initializing SGDClassifier and fitting it with training data")
    clf = SGDClassifier(n_jobs=-1, verbose=1)
    # Fit classifier with the given training instances and labels
    clf.fit(X_train, y_train)

    print("Saving model to file: {:s}".format(model_file))
    joblib.dump(clf, model_file)

    # Predict and self-evaluate classifier
    print("Self evaluating...")
    eval_f1(clf, X_train, y_train)

    if test_file is not None:
        predict(test_file, model_file=None, clf=clf)


def predict(test_file, model_file, clf=None):
    """
    Making predictions with given data and model.

    Args:
        test_file(str): Path to the testing dataset.
        model_file(str): Path to the trained model file.
        clf(sklearn clf object):
    """

    # Load classifier if clf is None
    if model_file is not None and clf is None:
        clf = joblib.load(model_file)

    # Load dataset
    X_test, y_test, id_test = read_dataset(test_file)

    # Predict and evaluate classifier
    y_pred = eval_f1(clf, X_test, y_test)

    # Save result
    output_arr = np.hstack([id_test, y_pred[:, np.newaxis]])
    df = pd.DataFrame(output_arr)
    df.to_csv("prediction.tsv", sep='\t', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Output filename or pre-trained \
                        model if --fit is given.")
    parser.add_argument("--train", help="Training set filename")
    parser.add_argument("--test", help="Testing set filename.")
    parser.add_argument("--predict", action="store_true",
                        help="Fit model with given training set.")

    args = parser.parse_args()

    # Create model save path
    if not os.path.exists("model/"):
        os.makedirs("model/")

    if args.predict:
        predict(args.test, args.model)
    else:
        run(args.train, args.test, args.model)