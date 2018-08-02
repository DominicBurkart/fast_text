# fast_text

[![Build Status](https://travis-ci.org/DominicBurkart/fast_text.svg?branch=master)](https://travis-ci.org/DominicBurkart/fast_text)
[![Coverage Status](https://coveralls.io/repos/github/DominicBurkart/fast_text/badge.svg)](https://coveralls.io/github/DominicBurkart/fast_text)

FastText implements a series of word-embeddings based NLP utilities that
operate efficiently on large datasets. This library allows rust
programs to build and interface with Facebook's FastText library.

See the [FastText codebase](https://github.com/facebookresearch/fastText) for more information.

Currently implemented:

- skipgram : full access to the skipgram function for generating a new model.
- min_skipgram : ease-of-use function for generating a skipgram model from the
minimal parameters (input and output paths).
- cbow : full access to the cbow function for generating a new model.
- min_cbow : ease-of-use function for generating a cbow model from the
minimal parameters (input and output paths).
- nn : given a word and model, returns the k nearest neighbors and their
distance.