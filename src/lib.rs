#[macro_use(c)]
extern crate cute;

use std::{thread, time};
use std::collections::HashMap;
use std::path::Path;
use std::process::{Command, Output, Stdio};


const VERSION: &'static str = env!("CARGO_PKG_VERSION");
const DEBUG: bool = true;

fn s(v: &str) -> String {
    v.to_string()
}

/// installs fastText
pub fn install() -> Vec<Output> {
    if cfg!(target_os = "windows") {
        unimplemented!("Windows support not yet enabled")
    } else {
        let cmds = [
            s("wget https://github.com/facebookresearch/fastText/archive/v0.1.0.zip"),
            s("unzip v") + VERSION + ".zip",
            s("cd fastText-") + VERSION + "; make",
            s("mv fastText-") + VERSION + "/fasttext .",
            s("rm -r fastText-") + VERSION,
            s("rm v") + VERSION + ".zip"
        ];
        c![
            Command::new("sh")
                .arg("-c")
                .arg(c)
                .stdout(Stdio::piped())
                .output()
                .expect("failed to execute process"),

            for c in cmds.iter()
        ]
    }
}

/// runs a fastText command and, if it fails because fastText DNE, installs fastText and tries again.
fn wrap_install(cmds: &str) -> Output {
    let r = Command::new("sh")
        .arg("-c")
        .arg(s("./fasttext ") + cmds)
        .stdout(Stdio::piped())
        .output()
        .expect("failed to execute process");
    if !r.status.success() && !Path::new("fasttext").exists() {
        let inst_resps = install();
        for ir in inst_resps.iter() {
            assert!(ir.status.success()); // panic if installation failed
        }
        println!("Recursing in wrap_install with command: {}", cmds);
        return wrap_install(cmds); // this technique is inefficient and may fail if the working
        // directory is changed by other processes.
    }
    r
}

/// Interface for the supervised learning algorithm. Documentation from fastText:
///
///The following arguments are mandatory:
///  -input              training file path
///  -output             output file path
///
///The following arguments are optional:
///  -verbose            verbosity level [2]
///
///The following arguments for the dictionary are optional:
///  -minCount           minimal number of word occurences [1]
///  -minCountLabel      minimal number of label occurences [0]
///  -wordNgrams         max length of word ngram [1]
///  -bucket             number of buckets [2000000]
///  -minn               min length of char ngram [0]
///  -maxn               max length of char ngram [0]
///  -t                  sampling threshold [0.0001]
///  -label              labels prefix [__label__]
///
///The following arguments for training are optional:
///  -lr                 learning rate [0.1]
///  -lrUpdateRate       change the rate of updates for the learning rate [100]
///  -dim                size of word vectors [100]
///  -ws                 size of the context window [5]
///  -epoch              number of epochs [5]
///  -neg                number of negatives sampled [5]
///  -loss               loss function {ns, hs, softmax} [softmax]
///  -thread             number of threads [12]
///  -pretrainedVectors  pretrained word vectors for supervised learning []
///  -saveOutput         whether output params should be saved [0]
///
///The following arguments for quantization are optional:
///  -cutoff             number of words and ngrams to retain [0]
///  -retrain            finetune embeddings if a cutoff is applied [0]
///  -qnorm              quantizing the norm separately [0]
///  -qout               quantizing the classifier [0]
///  -dsub               size of each sub-vector [2]
pub fn supervised(args: &HashMap<&str, &str>) {
    gen_mod(s("supervised"), args);
}

/// Interface to shrink a model's memory requirements. Full interface from fastText:
///
///usage: fasttext quantize <args>
///
///The following arguments are mandatory:
///  -input              training file path
///  -output             output file path
///
///The following arguments are optional:
///  -verbose            verbosity level [2]
///
///The following arguments for the dictionary are optional:
///  -minCount           minimal number of word occurences [5]
///  -minCountLabel      minimal number of label occurences [0]
///  -wordNgrams         max length of word ngram [1]
///  -bucket             number of buckets [2000000]
///  -minn               min length of char ngram [3]
///  -maxn               max length of char ngram [6]
///  -t                  sampling threshold [0.0001]
///  -label              labels prefix [__label__]
///
///The following arguments for training are optional:
///  -lr                 learning rate [0.05]
///  -lrUpdateRate       change the rate of updates for the learning rate [100]
///  -dim                size of word vectors [100]
///  -ws                 size of the context window [5]
///  -epoch              number of epochs [5]
///  -neg                number of negatives sampled [5]
///  -loss               loss function {ns, hs, softmax} [ns]
///  -thread             number of threads [12]
///  -pretrainedVectors  pretrained word vectors for supervised learning []
///  -saveOutput         whether output params should be saved [0]
///
///The following arguments for quantization are optional:
///  -cutoff             number of words and ngrams to retain [0]
///  -retrain            finetune embeddings if a cutoff is applied [0]
///  -qnorm              quantizing the norm separately [0]
///  -qout               quantizing the classifier [0]
///  -dsub               size of each sub-vector [2]
pub fn quantize(args: &HashMap<&str, &str>) {
    gen_mod(s("quantize"), args);
}


/// Classify each line in an input file. Documentation from fastText:
///
/// usage: fasttext predict[-prob] <model> <test-data> [<k>]
///
///  <model>      model filename
///  <test-data>  test data filename (if -, read from stdin)
///  <k>          (optional; 1 by default) predict top k labels
pub fn predict(model: &str, inp: &str, k: u32) -> Vec<f64> {
    let s = s("predict ") + model + " " + inp + " " + &k.to_string();
    let r = wrap_install(&s);
    c![p.parse::<f64>().unwrap(),
     for p in String::from_utf8_lossy(&r.stdout).split("\n")]
}


/// Helper function used to unspool arguments. S is a string with the primary fastText command
/// (e.g. "skipgram") and args are the named arguments to be passed to it, with keys as the
/// argument tag and values as the argument value.
fn gen_mod<'a>(mut s: String, args: &HashMap<&str, &'a str>) {
    for k in args.keys() {
        s = s + " -" + k + " " + args.get(k).unwrap();
    }
    if !wrap_install(&s).status.success() {
        panic!("Gen_mod failed with given input: {}", s)
    }
}

/// Provides functionality for generating skipgrams.
///
/// Include argument names as HashMap keys and argument values as HashMap values, e.g.:
/// "input" : "sample_text.txt"
/// "output" : "sample"
///
/// Documentation from fastText:
///
///
/// The following arguments are mandatory:
///  -input              training file path
///  -output             output file path
///
///The following arguments are optional:
///  -verbose            verbosity level [2]
///
///The following arguments for the dictionary are optional:
///  -minCount           minimal number of word occurences [5]
///  -minCountLabel      minimal number of label occurences [0]
///  -wordNgrams         max length of word ngram [1]
///  -bucket             number of buckets [2000000]
///  -minn               min length of char ngram [3]
///  -maxn               max length of char ngram [6]
///  -t                  sampling threshold [0.0001]
///  -label              labels prefix [__label__]
///
///The following arguments for training are optional:
///  -lr                 learning rate [0.05]
///  -lrUpdateRate       change the rate of updates for the learning rate [100]
///  -dim                size of word vectors [100]
///  -ws                 size of the context window [5]
///  -epoch              number of epochs [5]
///  -neg                number of negatives sampled [5]
///  -loss               loss function {ns, hs, softmax} [ns]
///  -thread             number of threads [12]
///  -pretrainedVectors  pretrained word vectors for supervised learning []
///  -saveOutput         whether output params should be saved [0]
///
///The following arguments for quantization are optional:
///  -cutoff             number of words and ngrams to retain [0]
///  -retrain            finetune embeddings if a cutoff is applied [0]
///  -qnorm              quantizing the norm separately [0]
///  -qout               quantizing the classifier [0]
///  -dsub               size of each sub-vector [2]
pub fn skipgram(args: &HashMap<&str, &str>) {
    gen_mod(s("skipgram"), args);
}

/// Provides functionality for generating a continuous bag of words model.
///
/// Include argument names as HashMap keys and argument values as HashMap values, e.g.:
/// "input" : "sample_text.txt"
/// "output" : "sample"
///
/// Documentation from fastText:
///
/// The following arguments are mandatory:
///  -input              training file path
///  -output             output file path
///
///The following arguments are optional:
///  -verbose            verbosity level [2]
///
///The following arguments for the dictionary are optional:
///  -minCount           minimal number of word occurences [5]
///  -minCountLabel      minimal number of label occurences [0]
///  -wordNgrams         max length of word ngram [1]
///  -bucket             number of buckets [2000000]
///  -minn               min length of char ngram [3]
///  -maxn               max length of char ngram [6]
///  -t                  sampling threshold [0.0001]
///  -label              labels prefix [__label__]
///
///The following arguments for training are optional:
///  -lr                 learning rate [0.05]
///  -lrUpdateRate       change the rate of updates for the learning rate [100]
///  -dim                size of word vectors [100]
///  -ws                 size of the context window [5]
///  -epoch              number of epochs [5]
///  -neg                number of negatives sampled [5]
///  -loss               loss function {ns, hs, softmax} [ns]
///  -thread             number of threads [12]
///  -pretrainedVectors  pretrained word vectors for supervised learning []
///  -saveOutput         whether output params should be saved [0]
///
///The following arguments for quantization are optional:
///  -cutoff             number of words and ngrams to retain [0]
///  -retrain            finetune embeddings if a cutoff is applied [0]
///  -qnorm              quantizing the norm separately [0]
///  -qout               quantizing the classifier [0]
///  -dsub               size of each sub-vector [2]
pub fn cbow(args: &HashMap<&str, &str>) {
    gen_mod(s("cbow"), args);
}

/// Provides minimal functionality for generating skipgrams. Full documentation from fastText:
///
/// The following arguments are mandatory:
///  -input              training file path
///  -output             output file path
///
///The following arguments are optional:
///  -verbose            verbosity level [2]
///
///The following arguments for the dictionary are optional:
///  -minCount           minimal number of word occurences [5]
///  -minCountLabel      minimal number of label occurences [0]
///  -wordNgrams         max length of word ngram [1]
///  -bucket             number of buckets [2000000]
///  -minn               min length of char ngram [3]
///  -maxn               max length of char ngram [6]
///  -t                  sampling threshold [0.0001]
///  -label              labels prefix [__label__]
///
///The following arguments for training are optional:
///  -lr                 learning rate [0.05]
///  -lrUpdateRate       change the rate of updates for the learning rate [100]
///  -dim                size of word vectors [100]
///  -ws                 size of the context window [5]
///  -epoch              number of epochs [5]
///  -neg                number of negatives sampled [5]
///  -loss               loss function {ns, hs, softmax} [ns]
///  -thread             number of threads [12]
///  -pretrainedVectors  pretrained word vectors for supervised learning []
///  -saveOutput         whether output params should be saved [0]
///
///The following arguments for quantization are optional:
///  -cutoff             number of words and ngrams to retain [0]
///  -retrain            finetune embeddings if a cutoff is applied [0]
///  -qnorm              quantizing the norm separately [0]
///  -qout               quantizing the classifier [0]
///  -dsub               size of each sub-vector [2]
pub fn min_skipgram(input: &str, output: &str) -> String {
    let st = s("skipgram -input ") + input + " -output " + output;
    let o = wrap_install(&st);
    if o.status.success() {
        s(output) + ".bin"
    } else {
        panic!("Min_skipgram failed with given input: {} \noutput: {:?}", st, o)
    }
}


/// Provides minimal functionality for generating a continuous bag of words model. Documentation
/// from fastText:
///
/// The following arguments are mandatory:
///  -input              training file path
///  -output             output file path
///
///The following arguments are optional:
///  -verbose            verbosity level [2]
///
///The following arguments for the dictionary are optional:
///  -minCount           minimal number of word occurences [5]
///  -minCountLabel      minimal number of label occurences [0]
///  -wordNgrams         max length of word ngram [1]
///  -bucket             number of buckets [2000000]
///  -minn               min length of char ngram [3]
///  -maxn               max length of char ngram [6]
///  -t                  sampling threshold [0.0001]
///  -label              labels prefix [__label__]
///
///The following arguments for training are optional:
///  -lr                 learning rate [0.05]
///  -lrUpdateRate       change the rate of updates for the learning rate [100]
///  -dim                size of word vectors [100]
///  -ws                 size of the context window [5]
///  -epoch              number of epochs [5]
///  -neg                number of negatives sampled [5]
///  -loss               loss function {ns, hs, softmax} [ns]
///  -thread             number of threads [12]
///  -pretrainedVectors  pretrained word vectors for supervised learning []
///  -saveOutput         whether output params should be saved [0]
///
///The following arguments for quantization are optional:
///  -cutoff             number of words and ngrams to retain [0]
///  -retrain            finetune embeddings if a cutoff is applied [0]
///  -qnorm              quantizing the norm separately [0]
///  -qout               quantizing the classifier [0]
///  -dsub               size of each sub-vector [2]
pub fn min_cbow(input: &str, output: &str) -> String {
    let st = s("cbow -input ") + input + " -output " + output;
    if wrap_install(&st).status.success() {
        s(output) + ".bin"
    } else {
        panic!("Cbow failed with given input: {}", st)
    }
}


/// Nearest neighbors. Input of "words" are single words separated by spaces. Full documentation
/// from FastText:
///
/// usage: fasttext nn <model> <k>
///
///  <model>      model filename
///  <k>          (optional; 10 by default) predict top k labels
pub fn nn(words: &str, model: &str, k: u32) -> Vec<Vec<(String, f64)>> {
    if DEBUG { println!("NN begun") };
    let cmd = s("echo ") + words + " | ./fasttext nn " + model + " " + &k.to_string();
    if DEBUG { println!("cmd: {}", cmd); }
    let r = Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .stdout(Stdio::piped())
        .output()
        .expect("failed to execute process");
    if r.status.code() == Some(127) { // returns 127 if ./fasttext DNE
        let ir = install();
        for o in ir.iter() {
            if !r.status.success() { panic!("Missing files / executable in call to nn"); }
        }
    }
    if DEBUG { println!("{:?}", r); }
    let stdout = String::from_utf8_lossy(&r.stdout);
    let sm = "Query word? ";
    let mut v0 = Vec::new();
    if DEBUG {
        println!("Beginning match iteration");
        println!("stdout: {}", stdout);
    }
    for (start, _) in stdout.match_indices(sm) {
        if DEBUG { println!("Match found: {}", start); }
        let mut v1 = Vec::new();
        let mut first = true;
        for l in stdout[start..].split("\n") {
            let lar: Vec<&str> = l.split(" ").collect();
            if DEBUG { println!("{:?}", lar); }
            if lar.len() == 2 {
                v1.push((lar[0].to_string(), lar[1].parse::<f64>().unwrap()));
            } else if lar.len() == 4 && first {
                v1.push((lar[2].to_string(), lar[3].parse::<f64>().unwrap()));
                first = false;
            } else if l == "Query word? " || (lar.len() == 4 && !first) {
                break;
            } else {
                panic!("misformatted line in input: {}", l);
            }
        }
        if v1.len() > 0 {
            v0.push(v1);
        }
    }
    v0
}

/// Documentation from fastText:
///
/// usage: fasttext analogies <model> <k>
///
///  <model>      model filename
///  <k>          (optional; 10 by default) predict top k labels
pub fn analogies(args: &HashMap<&str, &str>) {
    gen_mod(s("analogies"), args);
}

#[cfg(test)]
mod tests {
    extern crate kolmogorov_smirnov as ks;

    use std::collections::HashSet;
    use std::panic;
    use super::*;

    fn check_exists(file: &str, or: fn()) {
        if !Path::new(file).exists() {
            thread::sleep(time::Duration::from_secs(60));
            if !Path::new(file).exists() {
                or()
            }
        }
    }

    fn rm(files: Vec<&str>) {
        for f in files.iter() {
            let cmd = s("rm -r ") + f;
            let r = Command::new("sh")
                .arg("-c")
                .arg(&cmd)
                .stdout(Stdio::piped())
                .output()
                .expect("failed to execute process");
        }
    }

    fn set(v: Vec<Vec<(String, f64)>>) -> HashSet<String> {
        let mut out = HashSet::new();
        for v0 in v.into_iter() {
            for t in v0.into_iter() {
                let (st, _) = t;
                out.insert(st);
            }
        }
        out
    }

    fn sim(a: &HashSet<String>, b: &HashSet<String>) -> usize {
        c![v, for v in a.intersection(b)].len()
    }

    fn inst() {
        check_exists("fasttext", || { install(); });
    }

    fn samp() {
        check_exists("sample.bin", sample_skipgram);
    }

    #[test]
    fn test_install() {
        let rv = install();
        for r in rv.iter() {
            println!("{}", String::from_utf8_lossy(&r.stdout));
            println!("{}", String::from_utf8_lossy(&r.stderr));
            assert!(r.status.success());
        }

        let r = Command::new("sh")
            .arg("-c")
            .arg("./fasttext")
            .stdout(Stdio::piped())
            .output()
            .expect("failed to execute process");
        println!("{}", String::from_utf8_lossy(&r.stdout));
        println!("{}", String::from_utf8_lossy(&r.stderr));
        assert_eq!(r.status.code(), Some(1)); // returns 127 if ./fasttext DNE
    }

    /// generate a skipgram model for testing things like the nearest neighbor function.
    fn sample_skipgram() {
        inst();
        let model = min_skipgram("sample_text.txt", "sample");
        println!("Generated skipgram model: {}", model);
    }

    #[test]
    fn test_nn() {
        samp();

        let out = nn("lesbian", "sample.bin", 10);
        println!("{:?}", out);
        assert_eq!(out.len(), 1); // number of words queried
        assert_eq!(out[0].len(), 10); // k

        let out = nn("lesbian gay", "sample.bin", 5);
        println!("{:?}", out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 5);

        let out = nn("lesbian gay bisexual", "sample.bin", 8);
        println!("{:?}", out);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), 8);

        let out = nn("lesbian gay bisexual transgender", "sample.bin", 1);
        println!("{:?}", out);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), 1);
    }


    /// Since model generation is stochastic, this function compares results statistically.
    /// It does so by finding the number of pairwise shared words for a nearest neighbor lookup
    /// across two models. By generating a series of models from identical versus different function
    /// calls, we can then see if the different function calls produce models that yield
    /// statistically different output. The comparison statistic is a two-sample KS test, and
    /// the test fails if more comparisons are significant than would be expected due to chance.
    fn test_embedding(min_fn: fn(&str, &str) -> String, reg_fn: fn(&HashMap<&str, &str>), min_name: &str, reg_name: &str) {
        inst();

        let mut failed = 0;
        let mut total = 0;
        let conf = 0.9; // fairly arbitrary if iters is large enough, since we're just testing
        // whether the ratio of significant results is greater than expected at the confidence level
        // (ie ratio sig > (1 - conf)).

        let input = "sample_text.txt";
        let mut args = HashMap::new();
        args.insert("input", input);
        args.insert("output", reg_name);

        // iterate through a set of arbitrary words to compare on.
        for w in ["friend", "tomorrow", "clear"].iter() {
            let mut v1 = Vec::new();
            let mut v2 = Vec::new();

            // since model generation is stochastic, we'll need to compare results statistically.
            for i in 0..12 {
                let m1 = min_fn(input, min_name);
                reg_fn(&args);
                let m2 = s(min_name) + ".bin";

                v1.push(set(nn(w, &m1, 10)));
                v2.push(set(nn(w, &m2, 10)));
                println!("model iteration #: {}", i);
            }

            let mut self1 = Vec::new();
            for s1 in v1.iter() {
                for s2 in v1.iter() {
                    self1.push(sim(s1, s2))
                }
            }

            let mut self2 = Vec::new();
            for s1 in v2.iter() {
                for s2 in v2.iter() {
                    self2.push(sim(s1, s2))
                }
            }

            let mut between = Vec::new();
            for s1 in v1.iter() {
                for s2 in v2.iter() {
                    between.push(sim(s1, s2))
                }
            }

            // comparison: if the distribution of result pairwise similarities is different from our
            // different methods (or from the pairwise similarity between them), then the wrappers are
            // not equivalent. Here we're describing similarity as the number of shared words in the
            // nearest neighbors response for an arbitrary word.
            let s1s2 = ks::test(&self1, &self2, conf);
            let bs1 = ks::test(&between, &self1, conf);
            let bs2 = ks::test(&between, &self2, conf);
            total += 3;
            if s1s2.is_rejected {
                println!("self1: {:?}\n\nself2: {:?}\n\nbetween: {:?}", self1, self2, between);
                println!("Self 1 and self 2 are dissimilar. P of difference: {}", s1s2.reject_probability);
                failed += 1;
            } else if bs1.is_rejected {
                println!("self1: {:?}\n\nself2: {:?}\n\nbetween: {:?}", self1, self2, between);
                println!("Between and self 1 are dissimilar. P of difference: {}", bs1.reject_probability);
                failed += 1;
            } else if bs2.is_rejected {
                println!("self1: {:?}\n\nself2: {:?}\n\nbetween: {:?}", self1, self2, between);
                println!("Between and self 2 are dissimilar. P of difference: {}", bs2.reject_probability);
                failed += 1;
            }
        }
        let r1 = s(min_name) + "*";
        let r2 = s(reg_name) + "*";
        rm(vec![&r1, &r2]);
        if (failed as f64 / total as f64) > (1. - conf) {
            panic!("Test failed")
        }
    }

    #[test]
    fn test_skipgram() {
        test_embedding(min_skipgram, skipgram, "test_min_skipgram", "test_skipgram");
    }

    #[test]
    fn test_cbow(){
        test_embedding(min_cbow, cbow, "test_min_cbow", "test_cbow");
    }
}