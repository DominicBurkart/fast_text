# fast_text

[![Build Status](https://travis-ci.org/DominicBurkart/fast_text.svg?branch=master)](https://travis-ci.org/DominicBurkart/fast_text)
[![Coveralls Coverage Status](https://coveralls.io/repos/github/DominicBurkart/fast_text/badge.svg)](https://coveralls.io/github/DominicBurkart/fast_text)
[![Codecov Coverage Status](https://codecov.io/gh/DominicBurkart/fast_text/branch/master/graphs/badge.svg)](https://codecov.io/gh/DominicBurkart/fast_text)
[![Crates.io](https://img.shields.io/crates/v/fast_text.svg)](https://crates.io/crates/fast_text)
[![Rust Documentation](https://docs.rs/fast_text/badge.svg)](https://docs.rs/fast_text)

FastText implements a series NLP utilities that operate efficiently on
large datasets. This library allows rust programs to build and interface
 with Facebook's fastText library. It's not great in terms of performance: it
 works as a simple wrapper to the 
 command instead of trying to bind to a shared C API. 

See the fastText [website](https://fasttext.cc) and [codebase](https://github.com/facebookresearch/fastText) for more information.

Implementation notes:
- Does not currently support the analogy function
