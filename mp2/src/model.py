import argparse
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from autosklearn.classification import AutoSklearnClassifier
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from tqdm import tqdm

# python src/model.py --model=model/automl.pkl --train=data/trainset.tsv
# python src/model.py --model=model/automl.pkl --train=data/trainset.tsv --test=data/testset.tsv
# python src/model.py --model=model/automl.pkl --test=data/testset.tsv --predict


def expand_dense_matrix(dense, size):
    # Initialize sparse matrix
    sparse = np.zeros((len(dense), size))

    # Expand dense matrix
    print("Expanding dense matrix...")
    for idx, itr in tqdm(enumerate(dense)):
        if type(itr) is not str:
            continue
        indices = [int(i) for i in itr.split(", ")]
        sparse[idx, :][indices] = 1

    return sparse

def f1_eval(y_true, y_pred):
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    print("Evaluation on training set:")
    print("F1-micro: {:2.5f} | F1-macro: {:2.5f} | F1-weighted: {:2.5f}"
          .format(f1_micro, f1_macro, f1_weighted))


def run(train, test=None, model=None):
    """
    """
    # clf = SGDClassifier()
    # clf.fit(X, Y)
    print("Loading training data from {0}".format(train))
    cols = ["id", "feature_vector", "encoded_labels"]
    df = pd.read_csv(train, delimiter="\t", header=None, names=cols)

    # Some useful information
    n_data = int(df["id"].as_matrix()[0])
    print(" - Total number of data: {:8d}\n".format(n_data))
    n_bow = int(df["feature_vector"].as_matrix()[0])
    n_class = int(df["encoded_labels"].as_matrix()[0])

    # Expand Training feature vectors and labels 
    X_train = expand_dense_matrix(df["feature_vector"].as_matrix()[1:], n_bow)
    y_train = expand_dense_matrix(df["encoded_labels"].as_matrix()[1:], n_class)
    del df

    print("Initializing AutoSklearnClassifier and fitting with training data")
    clf = AutoSklearnClassifier()
    clf.fit(X_train, y_train)

    print("Saving model to file: {:s}".format(model))
    joblib.dump(clf, model)

    # Self evaluation
    print("Self evaluating...")
    y_pred = clf.predict(X_test)
    f1_eval(y_train, y_pred)

    if test is not None:
        predict(test, model, clf=clf)

def predict(test, model, clf=None):

    if clf is None:
        clf = joblib.load(model)

    # Load testing data
    print()
    print("Loading testing data from {0}".format(test))
    cols = ["id", "feature_vector", "encoded_labels"]
    df = pd.read_csv(test, delimiter="\t", header=None, names=cols)

    # Some useful information
    n_data = int(df["id"].as_matrix()[0])
    print(" - Total number of data: {:8d}\n".format(n_data))
    n_bow = int(df["feature_vector"].as_matrix()[0])
    n_class = int(df["encoded_labels"].as_matrix()[0])

    id_test = df["id"].as_matrix()[1:][:, np.newaxis]
    # Expand Training feature vectors and labels 
    X_test = expand_dense_matrix(df["feature_vector"].as_matrix()[1:], n_bow)
    y_test = expand_dense_matrix(df["encoded_labels"].as_matrix()[1:], n_class)
    del df

    # Predict
    y_pred = clf.predict(X_test)
    f1_eval(y_test, y_pred)

    # Save result
    output_arr = np.hstack(id_test, y_pred[:, np.newaxis])
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