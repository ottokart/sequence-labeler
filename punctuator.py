import sys
import numpy

from collections import OrderedDict
from sequence_labeler import SequenceLabeler
from sequence_labeling_experiment import read_dataset, create_batches, parse_config, map_text_to_ids
from punctuation_data_converter import EOS_TOKENS, SPACE, MAX_SEQUENCE_LEN

def last_index_of(array, element):
    try:
        return len(array) -1 - array[::-1].index(element)
    except:
        return 0

def up_to_last_instance_of(array, elements):
    idx = max(last_index_of(array, element) for element in elements)
    if idx == 0:
        return array
    else:
        return array[:idx + 1]

def reverse_mapping(d):
    return OrderedDict([(v,k) for (k,v) in d.items()])

def convert_to_batch(word_sequence, lowercase_words, lowercase_chars, replace_digits, word2id, char2id):
    raw_word_ids = map_text_to_ids(" ".join(word_sequence), word2id, "<s>", "</s>", "<unk>", lowercase=lowercase_words, replace_digits=replace_digits)
    raw_char_ids = [map_text_to_ids("<s>", char2id, "<w>", "</w>", "<cunk>")] + \
                   [map_text_to_ids(" ".join(list(word)), char2id, "<w>", "</w>", "<cunk>", lowercase=lowercase_chars, replace_digits=replace_digits) for word in word_sequence] + \
                   [map_text_to_ids("</s>", char2id, "<w>", "</w>", "<cunk>")]

    assert(len(raw_char_ids) == len(raw_word_ids))

    # Mask and convert to numpy array
    batch_size = 1
    seq_len = len(raw_word_ids)

    max_word_length = numpy.array([len(c) for c in raw_char_ids]).max()

    word_ids = numpy.zeros((batch_size, seq_len), dtype=numpy.int32)
    char_ids = numpy.zeros((batch_size, seq_len, max_word_length), dtype=numpy.int32)
    char_mask = numpy.zeros((batch_size, seq_len, max_word_length), dtype=numpy.int32)

    for i in range(batch_size):
        for j in range(seq_len):
            word_ids[i][j] = raw_word_ids[j]
        for j in range(seq_len):
            for k in range(len(raw_char_ids[j])):
                char_ids[i][j][k] = raw_char_ids[j][k]
                char_mask[i][j][k] = 1

    return word_ids, char_ids, char_mask

def punctuate(config_path):
    config = parse_config("config", config_path)
    if config["path_test"] is None:
        print("No test data configured")
        return

    sequencelabeler = SequenceLabeler.load(config["save"])
    label2id = sequencelabeler.config["label2id"]
    word2id = sequencelabeler.config["word2id"]
    char2id = sequencelabeler.config["char2id"]

    config["word2id"] = word2id
    config["char2id"] = char2id
    config["label2id"] = label2id

    id2label = reverse_mapping(label2id)
    eos_labels = [label2id[l] for l in EOS_TOKENS if l in label2id]
    space_id = label2id[SPACE]

    all_predicted_labels = []

    for path_test in config["path_test"].strip().split(":"):

        with open(path_test + '.orig', 'r') as f:
            all_words = [w for w in f.read().split() if w not in label2id]

        last_eos_idx = 0
        
        while True:
            word_sequence = all_words[last_eos_idx:last_eos_idx+MAX_SEQUENCE_LEN]
            if len(word_sequence) == 0:
                break
            word_ids, char_ids, char_mask = convert_to_batch(word_sequence, False, False, True, word2id, char2id)
            predicted_labels = sequencelabeler.predict(word_ids, char_ids, char_mask)
            predicted_labels = up_to_last_instance_of(list(predicted_labels.flatten()), eos_labels)
            if len(predicted_labels) == 0:
                break
            all_predicted_labels += predicted_labels
            last_eos_idx += len(predicted_labels)

        with open(path_test + '.pred', 'w') as f:
            for w, l_id in zip(all_words, all_predicted_labels):
                f.write('%s %s ' % (w, '' if l_id == space_id else id2label[l_id]))

if __name__ == "__main__":
    punctuate(sys.argv[1])

