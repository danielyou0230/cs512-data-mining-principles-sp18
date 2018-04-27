import argparse
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import tqdm

# python src/run.py --model=model/clf.pkl --train=data/trainset.tsv
# python src/run.py --model=model/h_clf.pkl --train=data/h_trainset.tsv --hin

# python src/run.py --model=model/clf.pkl --train=data/trainset.tsv --test=data/testset.tsv
# python src/run.py --model=model/h_clf.pkl --train=data/h_trainset.tsv --test=data/h_testset.tsv --hin

# python src/run.py --model=model/clf.pkl --test=data/testset.tsv --predict
# python src/run.py --model=model/h_clf.pkl --test=data/h_testset.tsv --predict --hin


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

    return X_train, y_train, id_data[1:]


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
        per_class_eval(dict): Dictionary contains labels as keys and
                              [precision, recall] as values.
    """

    # Predict on given data
    y_pred = clf.predict(X)

    # Calculating some metrics
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    print("F1 Evaluations:")
    print("F1-micro: {:2.5f} | F1-macro: {:2.5f} | F1-weighted: {:2.5f}"
          .format(f1_micro, f1_macro, f1_weighted))

    per_class_pre = list(precision_score(y_true, y_pred, average=None))
    per_class_rec = list(recall_score(y_true, y_pred, average=None))
    class_space = list(np.unique(y_true))
    per_class_scores = list(zip(per_class_pre, per_class_rec))
    per_class_eval = dict(zip(class_space, per_class_scores))

    return y_pred, per_class_eval


def run(train_file, test_file=None, model_file=None, hin=False, encoder_model=None):
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
        predict(test_file, model_file=None, clf=clf, hin=hin, encoder_model=encoder_model)


def predict(test_file, model_file, clf=None, hin=False, encoder_model=None):
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
    y_pred, per_class_eval = eval_f1(clf, X_test, y_test)

    # Inverse transform the labels to venues (human-interpretable)
    if encoder_model is None:
        encoder_model = "model/{0}encoder.pkl".format("h_" if hin else "")
    # Load LabelEncoder
    print("Loading LabelEncoder from {0}".format(encoder_model))
    encoder = joblib.load(encoder_model)
    # Convert to labels
    print("Inverse transforming labels to venues...\n")
    venues = encoder.inverse_transform(y_pred)

    # Save result
    output_arr = np.hstack([id_test[:, np.newaxis], venues[:, np.newaxis]])
    df = pd.DataFrame(output_arr)
    pred_file = "{0}prediction.tsv".format("h_" if hin else "")
    df.to_csv(pred_file, sep='\t', index=False, header=False)
    print("Prediction saved to {0}".format(pred_file))
    del df, output_arr

    # Save per class precision and recall score
    all_classes = encoder.classes_
    # All distinct class labels contained in the prediction
    eval_classes = list(per_class_eval.keys())
    eval_classes = list(encoder.inverse_transform(eval_classes))
    # Change the label in per_class_eval to text
    per_class_eval = dict(zip(eval_classes, list(per_class_eval.values())))

    # Create an array to record the corresponding precision and recall for each venue
    stat = np.empty([all_classes.shape[0], 2])
    stat.fill(np.nan)
    for idx, itr in enumerate(all_classes):
        # Fill in precision and recall if the predication contains the venue
        try:
            stat[idx, 0] = per_class_eval[itr][0]
            stat[idx, 1] = per_class_eval[itr][1]
        # np.nan otherwise (default)
        except:
            pass

    output_arr = np.hstack([all_classes[:, np.newaxis], stat])
    df = pd.DataFrame(output_arr)
    pr_file = "{0}precision_recall.tsv".format("h_" if hin else "")
    df.to_csv(pr_file, sep='\t', index=False, header=["venue", "precision", "recall"])
    print("Precision and Recall for all classes saved to {0}".format(pr_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Output filename or pre-trained \
                        model if --fit is given.")
    parser.add_argument("--train", help="Training set filename")
    parser.add_argument("--test", help="Testing set filename.")
    parser.add_argument("--hin", action="store_true", help="Use heterogeneous features.")
    parser.add_argument("--encoder", help="LabelEncoder filename.")
    parser.add_argument("--predict", action="store_true",
                        help="Fit model with given training set.")
    args = parser.parse_args()

    # Create model save path
    if not os.path.exists("model/"):
        os.makedirs("model/")

    if args.predict:
        predict(args.test, args.model, hin=args.hin)
    else:
        run(args.train, args.test, args.model, args.hin)