import re
import nltk
import numpy as np
from collections import Counter
from itertools import chain
from tqdm import tqdm
from pprint import pprint
from multiprocessing import Pool, cpu_count


def split_data(data, n_slice):
    """
    Split data to minibatches with last batch may be larger or smaller.
    Arguments:
        data(ndarray): Array of data.
        n_slice(int): Number of slices to separate the data.
    Return:
        partitioned_data(list): List of list containing any type of data.
    """
    n_data = len(data) if type(data) is list else data.shape[0]
    # Slice data for each thread
    print(" - Slicing data for threading...")
    print(" - Total number of data: {0}".format(n_data))
    per_slice = int(n_data / n_slice)
    partitioned_data = list()
    for itr in range(n_slice):
        # Generate indices for each slice
        idx_begin = itr * per_slice
        # last slice may be larger or smaller
        idx_end = (itr + 1) * per_slice if itr != n_slice - 1 else None
        #
        partitioned_data.append((itr, data[idx_begin:idx_end]))
    #
    return partitioned_data


def generic_threading(n_jobs, data, method, param=None, shared=False):
    """
    Generic threading method.
    Arguments:
        n_jobs(int): number of thead to run the target method
        data(ndarray): Data that will be split and distributed to threads.
        method(method object): Threading target method
        param(tuple): Tuple of additional parameters needed for the method.
        shared: (undefined)
    Return:
        result(list of any type): List of return values from the method.
    """
    # Threading settings
    n_cores = cpu_count()
    n_threads = n_cores * 2 if n_jobs == None else n_jobs
    print("Number of CPU cores: {:d}".format(n_cores))
    print("Number of Threading: {:d}".format(n_threads))
    #
    thread_data = split_data(data, n_threads)
    if param is not None:
        thread_data = [itr + param for itr in thread_data]
    else:
        pass
    #
    print(" - Begin threading...")
    # Threading
    with Pool(processes=n_threads) as p:
        result = p.starmap(method, thread_data)
    #
    print("\n" * n_threads)
    print("All threads completed.")
    return result


def readlines(file, begin=None, limit=None, delimiter=None, lower=False):
    """
    Read and split all content in the files line by line with delimiter
    if specified.

    Args:
        file(str): File to be read.
        begin(int): Index of the first line to be read.
        limit(int): Index of the last line to be read, or the amount of
            samples drawn from the dataset if rand is asserted.
    Return:
        data(list of strings): Lines from the files
    """
    print()
    print("Loading lines in file: {0}".format(file))
    #
    with open(file, "r") as f:
        lines = f.read().replace('\u2028',' ').splitlines()[begin:limit]

    if lower:
        print(" - Convert all context to lower case.")
    else:
        print(" - Keep all context original cases.")

    data = list()
    for itr in lines:
        data.append(itr.lower().split(delimiter) if lower else itr.split(delimiter))
    print("Total {0} lines loaded.".format(len(lines)))

    return data

def write_to_file(file, data, delimiter=None, row_as_line=False):
    """
    Write strings to files.
    Arguments:
        file(str): Output filename.
        data(list): List of list of strings / List of strings.
    """
    #
    lines = list()
    if row_as_line:
        for itr in data:
            # Only one word
            if type(itr) is not list:
                lines.append(itr)
            # Multi-words
            else:
                separator = "{0}".format(delimiter)
                lines.append(separator.join(itr))
    else:
        lines = data
    #
    print("Writing result to file...")
    with open(file, "w") as f:
        if type(lines[0]) == list:
            for itr in tqdm(list(chain.from_iterable(lines))):
                f.write(itr + "\n")
        else:
            for itr in tqdm(lines):
                f.write(itr + "\n")
    print("File saved in {:s}".format(file))

def remove_redundant_char(data, index):
    """
    Args:
        data(list):
        index(int): The index to be processed.
    Returns:
        data(list): Same as original data but with the object in 
                   the designated index in each row modified.
    """
    print()
    print("Removing redundant characters")
    # Define punctuations to remove as regex object
    pattern = re.compile('[^A-Za-z0-9-]+')

    # Iterative remove punctuation based on the predefined pattern
    for itr in tqdm(range(len(data))):
        data[itr][index] = re.sub(pattern, ' ', data[itr][index])

    return data

def tokenize_context(data, index, thread=None):
    """
    Args:
    Returns:
        words()
    """
    print()
    print("Tokenizing titles")
    #
    if thread is None:
        result = _tokenize_context(0, data, index)
    else:
        # Threading
        param = (index,)
        result = generic_threading(thread, data, _tokenize_context, param)
        words = list(chain.from_iterable([itr[0] for itr in result]))
        data = list(chain.from_iterable([itr[1] for itr in result]))
    # print(len(result))
    return words, data


def _tokenize_context(thread_idx, data, index):
    """
    """
    desc = "Thread {:2d}".format(thread_idx + 1)
    #
    words = list()
    for itr in tqdm(range(len(data)), position=thread_idx, desc=desc):
        tokenized = nltk.word_tokenize(data[itr][index])
        data[itr][index] = tokenized
        words.append(tokenized)

    # Unpack the 2-d list to 1-d list
    return list(chain.from_iterable(words)), data

def find_frequent_words(words, threshold):
    """
    Args:
        words(list): Messy list of duplicated words to be counted.
        threshold(int): The threshold to filer out infrequent words.
    Returns:
        words(list): 
    """
    print()
    print("Finding frequent words in titles")
    # Counting the occurence of distinct words in the list "words"
    occurence = dict(Counter(words))
    # Create a list indicating frequent words
    frequent_words = list()
    # pprint(occurence)

    for itr_word, itr_occ in tqdm(occurence.items()):
        # Remove words with occurence >= threshold
        if itr_occ >= threshold:
            frequent_words.append(itr_word)

    frequent_words.sort()
    #
    freq_w = len(frequent_words)
    org_w  = len(list(np.unique(words)))
    percentage = 100. * (org_w - freq_w) / org_w
    print(" - Occurence Threshold: {:3d}".format(threshold))
    print(" - Frequent words: {:6d} (Raw: {:6d}, reduced by {:2.2f}%)"
          .format(freq_w, org_w, percentage))
    return frequent_words

def filter_title(data, index, frequent_words, thread=None):
    """

    Args:
        data(ndarray):
        index(int):
        frequent_words(list): 

    Returns:
        result(ndarray):
    """
    print()
    print("Filtering the title with frequent words.")
    #
    if thread is None:
        result = _filter_title(0, data, index, frequent_words)
    else:
        # Threading
        param = (index, frequent_words)
        result = generic_threading(thread, data, _filter_title, param)
        result = list(chain.from_iterable(result))

    return result

def _filter_title(thread_idx, data, index, frequent_words):
    """
    """
    desc = "Thread {:2d}".format(thread_idx + 1)
    #
    for itr in tqdm(range(len(data)), position=thread_idx, desc=desc):
        filtered = list()
        for itr_word in frequent_words:
            # Append frequent words in title to list
            if itr_word in data[itr][index]:
                filtered.append(itr_word)
            #
            else:
                pass
        # Replace original title with frequent words
        filtered = list(np.unique(filtered))
        data[itr][index] = " ".join(filtered)

    return data