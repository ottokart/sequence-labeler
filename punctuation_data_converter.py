# coding: utf-8
from __future__ import division

import random
import os
import sys
import operator
import cPickle
import codecs
import fnmatch

DATA_PATH = "./data"

END = "</S>"
UNK = "<UNK>"

SPACE = "_SPACE"

MAX_WORD_VOCABULARY_SIZE = 100000
MIN_WORD_COUNT_IN_VOCAB = 2
MAX_SEQUENCE_LEN = 50

TRAIN_FILE = os.path.join(DATA_PATH, "train")
DEV_FILE = os.path.join(DATA_PATH, "dev")
TEST_FILE = os.path.join(DATA_PATH, "test")

PUNCTUATION_VOCABULARY = {SPACE, ",COMMA", ".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK", ":COLON", ";SEMICOLON", "-DASH"}
PUNCTUATION_MAPPING = {}

# Comma, period & question mark only:
# PUNCTUATION_VOCABULARY = {SPACE, ",COMMA", ".PERIOD", "?QUESTIONMARK"}
# PUNCTUATION_MAPPING = {"!EXCLAMATIONMARK": ".PERIOD", ":COLON": ",COMMA", ";SEMICOLON": ".PERIOD", "-DASH": ",COMMA"}

EOS_TOKENS = {".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK"}
CRAP_TOKENS = {"<doc>", "<doc.>"} # punctuations that are not included in vocabulary nor mapping, must be added to CRAP_TOKENS

def write_processed_dataset(input_files, output_file):
    """
    data will consist of two sets of aligned subsequences (words and punctuations) of MAX_SEQUENCE_LEN tokens (actually punctuation sequence will be 1 element shorter).
    If a sentence is cut, then it will be added to next subsequence entirely (words before the cut belong to both sequences)
    """

    current_words = []
    current_punctuations = []

    last_eos_idx = 0 # if it's still 0 when MAX_SEQUENCE_LEN is reached, then the sentence is too long and skipped.
    last_token_was_punctuation = True # skipt first token if it's punctuation

    skip_until_eos = False # if a sentence does not fit into subsequence, then we need to skip tokens until we find a new sentence

    for input_file in input_files:

        with codecs.open(input_file, 'r', 'utf-8') as text, \
             codecs.open(output_file, 'w', 'utf-8') as text_out:

            for line in text:

                for token in line.split():

                    # First map oov punctuations to known punctuations
                    if token in PUNCTUATION_MAPPING:
                        token = PUNCTUATION_MAPPING[token]

                    if skip_until_eos:

                        if token in EOS_TOKENS:
                            skip_until_eos = False

                        continue

                    elif token in CRAP_TOKENS:
                        continue

                    elif token in PUNCTUATION_VOCABULARY:

                        if last_token_was_punctuation: # if we encounter sequences like: "... !EXLAMATIONMARK .PERIOD ...", then we only use the first punctuation and skip the ones that follow
                            continue

                        if token in EOS_TOKENS:
                            last_eos_idx = len(current_punctuations) # no -1, because the token is not added yet

                        punctuation = token

                        current_punctuations.append(punctuation)
                        last_token_was_punctuation = True

                    else:

                        if not last_token_was_punctuation:
                            current_punctuations.append(SPACE)

                        word = token

                        current_words.append(word)
                        last_token_was_punctuation = False

                    if len(current_words) == MAX_SEQUENCE_LEN: # this also means, that last token was a word
                        
                        assert len(current_words) == len(current_punctuations) + 1, "#words: %d; #punctuations: %d" % (len(current_words), len(current_punctuations))

                        # Sentence did not fit into subsequence - skip it
                        if last_eos_idx == 0: 
                            skip_until_eos = True

                            current_words = []
                            current_punctuations = []

                            last_token_was_punctuation = True # next sequence starts with a new sentence, so is preceded by eos which is punctuation

                        else:

                            for w, p in zip(current_words, current_punctuations):
                                text_out.write('%s\t%s\n' % (w, p))
                            text_out.write('\n')

                            # Carry unfinished sentence to next subsequence
                            current_words = current_words[last_eos_idx+1:]
                            current_punctuations = current_punctuations[last_eos_idx+1:]

                        last_eos_idx = 0 # sequence always starts with a new sentence

def create_dev_test_train_split(root_path, train_output, dev_output, test_output):

    train_txt_files = []
    dev_txt_files = []
    test_txt_files = []

    for root, dirnames, filenames in os.walk(root_path):
        for filename in fnmatch.filter(filenames, '*.txt'):

            path = os.path.join(root, filename)

            if filename.endswith(".test.txt"):
                test_txt_files.append(path)

            elif filename.endswith(".dev.txt"):
                dev_txt_files.append(path)

            else:
                train_txt_files.append(path)

    write_processed_dataset(train_txt_files, train_output)
    write_processed_dataset(dev_txt_files, dev_output)
    write_processed_dataset(test_txt_files, test_output)

if __name__ == "__main__":

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        sys.exit("The path to source data directory with txt files is missing")

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    else:
        sys.exit("Data already exists")

    create_dev_test_train_split(path, TRAIN_FILE, DEV_FILE, TEST_FILE)

