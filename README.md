A fork from https://github.com/marekrei/sequence-labeler to enable punctuation restoration in unsegmented text.

## Performance on English TED talks
(Training set size: 2.1M words)

PUNCTUATION      | PRECISION | RECALL    | F-SCORE
--- | --- | --- | ---
,COMMA           | 58.5 | 58.7 | 58.6
?QUESTIONMARK    | 71.4 | 54.3 | 61.7
.PERIOD          | 69.9 | 72.0 | 70.9
_Overall_        | _64.3_ | _64.9_ | _64.6_

Performance is very similar (even slightly better) to https://github.com/ottokart/punctuator2 although they are not directly comparable as punctuator2 used pretrained embeddings that were trained on much less data and had much smaller size. More details can be found [here](http://www.isca-speech.org/archive/Interspeech_2016/pdfs/1517.PDF).

Original README:
=========================

Sequence labeler
=========================

This is a neural network sequence labeling system. Given a sequence of tokens, it will learn to assign labels to each token. Can be used for named entity recognition, POS-tagging, error detection, chunking, CCG supertagging, etc.

The main model implements a bidirectional LSTM for sequence tagging. In addition, you can incorporate character-level information -- either by concatenating a character-based representation, or by using an attention/gating mechanism for combining it with a word embedding.

Run with:

    python sequence_labeling_experiment.py config.conf

Preferably with Theano set up to use CUDA, so the process can run on a GPU.

Requirements
-------------------------

* numpy
* theano
* lasagne

Configuration
-------------------------

Edit the values in config.conf as needed:

* **path_train** - Path to the training data, in CoNLL tab-separated format. One word per line, first column is the word, last column is the label. Empty lines between sentences.
* **path_dev** - Path to the development data, used for choosing the best epoch.
* **path_test** - Path to the test file. Can contain multiple files, colon separated.
* **main_label** - The output label for which precision/recall/F-measure are calculated.
* **conll_eval** - Whether the standard CoNLL NER evaluation should be run.
* **preload_vectors** - Path to the pretrained word embeddings, in word2vec plain text format. If your embeddings are in binary, you can use [convertvec](https://github.com/marekrei/convertvec) to convert them to plain text.
* **word_embedding_size** - Size of the word embeddings used in the model.
* **char_embedding_size** - Size of the character embeddings.
* **word_recurrent_size** - Size of the word-level LSTM hidden layers.
* **char_recurrent_size** - Size of the char-level LSTM hidden layers.
* **narrow_layer_size** - Size of the extra hidden layer on top of the bi-LSTM.
* **best_model_selector** - What is measured on the dev set for model selection: "dev_conll_f:high" for NER and chunking, "dev_acc:high" for POS-tagging, "dev_f05:high" for error detection.
* **epochs** - Maximum number of epochs to run.
* **stop_if_no_improvement_for_epochs** - Training will be stopped if there has been no improvement for n epochs.
* **learningrate** - Learning rate.
* **min_word_freq** - Minimal frequency of words to be included in the vocabulary. Others will be considered OOV.
* **max_batch_size** - Maximum batch size.
* **save** - Path to save the model.
* **load** - Path to load the model.
* **random_seed** - Random seed for initialisation and data shuffling. This can affect results, so for robust conclusions I recommend running multiple experiments with different seeds and averaging the metrics.
* **crf_on_top** - If True, use a CRF as the output layer. If False, use softmax instead.
* **char_integration_method** - How character information is integrated. Options are: "none" (not integrated), "input" (concatenated), "attention" (the method proposed in Rei et al. (2016)).


References
-------------------------

If you use the main sequence labeling code, please reference:

[**Compositional Sequence Labeling Models for Error Detection in Learner Writing**](http://aclweb.org/anthology/P/P16/P16-1112.pdf)  
Marek Rei and Helen Yannakoudakis  
*In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL-2016)*
  

If you use the character-level attention component, please reference:

[**Attending to characters in neural sequence labeling models**](https://aclweb.org/anthology/C/C16/C16-1030.pdf)  
Marek Rei, Sampo Pyysalo and Gamal K.O. Crichton  
*In Proceedings of the 26th International Conference on Computational Linguistics (COLING-2016)*
  

The CRF implementation is based on:

[**Neural Architectures for Named Entity Recognition**](https://arxiv.org/abs/1603.01360)  
Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami and Chris Dyer  
*In Proceedings of NAACL-HLT 2016*
  

The conlleval.py script is from: https://github.com/spyysalo/conlleval.py


License
---------------------------

MIT License

Copyright (c) 2016 Marek Rei

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
