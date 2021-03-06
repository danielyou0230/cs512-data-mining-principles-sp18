import argparse
from utils import readlines, remove_redundant_char, tokenize_context, find_frequent_words, filter_title, write_to_file, sample_data
from itertools import chain


# python src/cleanup_dataset.py data/subset.txt --output=data/cleaned.txt --thread=3
# python src/cleanup_dataset.py data/training.txt --output=data/cleaned_data.txt --thread=10
# python src/cleanup_dataset.py data/validation.txt --output=data/cleaned_validation.txt --thread=10 --cleanup_only


def cleanup(file, output, cleanup_only=False, threshold=5, thread=None):
    """
    Cleanup the dataset according to the specs of the task.

    Args:
        file(str): Input corpus filename.
        output(str): Output filename.
        cleanup_only(bool): Just cleanup the words using predefined frequent words.
        threshold(int): The threshold to filter out infrequent words.
        thread(int): Number of thread to run simultaneously
    """

    # 1. Load and convert each title to lowercase.
    data = readlines(file, delimiter="\t", lower=True)

    # 2. Remove all characters that are not 
    #    (1) lowercase characters (a-z), 
    #    (2) whitespace, or 
    #    (3) hyphen '-'
    data = remove_redundant_char(data, index=1)

    # 3. Tokenize each title into words by splitting on whitespace.
    words, data = tokenize_context(data, index=1, thread=thread)

    # 4. Remove all tokens that appear fewer than 5 times in the dataset.
    # 4-1. Find frequent words
    if not cleanup_only:
        frequent_words = find_frequent_words(words, threshold=threshold)
        write_to_file("models/frequent_words.txt", frequent_words)
    else:
        print("Loading frequent_words from training set.")
        frequent_words = readlines("models/frequent_words.txt", lower=True)
        frequent_words = list(chain.from_iterable(frequent_words))

    # 4-2. Remove infrequent words in titles
    data = filter_title(data, index=1, frequent_words=frequent_words, thread=thread)
    # Save to file
    write_to_file(output, data, delimiter="\t", row_as_line=True)


def sample_dataset(file, amount):
    """
    Sample the given amount of data from the file.

    Args:
        file(str): File to be sampled.
        amount(int): Amount of data to be drawn from the file.
    """

    # Load and convert each title to lowercase.
    data = readlines(file, delimiter="\t", lower=True)
    # Sample
    sample_data(file, data, amount=amount)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File to be parsed.")
    parser.add_argument("--output", help="Output filename.")
    parser.add_argument("--cleanup_only", action="store_true", 
                        help="Just cleanup the .")
    parser.add_argument("--threshold", nargs='?', type=int, default=5, 
                        help="Occurance below threshold would be filtered out.")
    parser.add_argument("--thread", type=int, help="Number of threads to run.")
    parser.add_argument("--sample", action="store_true", help="Sample the given file.")
    parser.add_argument("--amount", type=int, default=50,
                        help="Number of samples to be drawn from the file.")

    args = parser.parse_args()

    if args.sample:
        sample_dataset(args.file, args.amount)
    else:
        cleanup(args.file, args.output, args.cleanup_only, args.threshold, args.thread)