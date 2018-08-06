#[macro_use(c)]
extern crate cute;

use std::collections::HashMap;
use std::path::Path;
use std::process::{Command, Output, Stdio};


const VERSION: &'static str = "0.1.0";
// fastText archive version to pull.

const DEBUG: bool = true;

fn s(v: &str) -> String {
    v.to_string()
}

/// Installs fastText from the archive on Facebook's github.
pub fn install() -> Vec<Output> {
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

/// runs a fastText command and, if it fails because fastText DNE, installs fastText and tries again.
fn wrap_install(cmds: &str) -> Output {
    let st = s("./fasttext ") + cmds;
    run_cmd(&st)
}

/// Interface for fastText's supervised learning algorithm.
///
/// Each line of input should have labels included as such:
///
/// __label__sauce __label__cheese How much does potato starch affect a cheese sauce recipe?
/// __label__food-safety __label__acidity Dangerous pathogens capable of growing in acidic environments
/// __label__cast-iron __label__stove How do I cover up the white spots on my cast iron stove?
///
///
/// Documentation from fastText:
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

/// Interface to shrink a model's memory requirements.
///
/// Full interface from fastText:
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


/// Classify each line in an input file.
///
/// output: a vector of the same length as the number of lines in the input
/// file where each length is value is a  k-length vector of labels for the text from the line.
///
/// Documentation from fastText:
///
/// usage: fasttext predict[-prob] <model> <test-data> [<k>]
///
///  <model>      model filename
///  <test-data>  test data filename (if -, read from stdin)
///  <k>          (optional; 1 by default) predict top k labels
pub fn predict(model: &str, inp: &str, k: u32) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    let s = s("predict ") + model + " " + inp + " " + &k.to_string();
    let r = wrap_install(&s);
    for p in String::from_utf8_lossy(&r.stdout).split("\n") {
        let mut innerv = Vec::new();
        for v in p.split(" ") {
            if v != "" {
                innerv.push(v.to_string());
            }
        }
        if innerv.len() != 0 {
            out.push(innerv);
        }
    }
    out
}

/// Classify each line in an input file with probabilities of labels.
///
/// Documentation from fastText:
///
/// usage: fasttext predict[-prob] <model> <test-data> [<k>]
///
///  <model>      model filename
///  <test-data>  test data filename (if -, read from stdin)
///  <k>          (optional; 1 by default) predict top k labels
pub fn predict_prob(model: &str, inp: &str, k: u32) -> Vec<Vec<(String, f64)>> {
    fn ext(l: &str) -> Vec<(String, f64)> {
        let mut out = Vec::new();
        let mut f = true;
        let mut label = "";
        for u in l.split(" ") {
            if u != "" {
                if f {
                    label = u;
                } else {
                    out.push((label.to_string(), u.parse::<f64>().unwrap()));
                }
                f = !f;
            }
        }
        if DEBUG { assert!(f); } // last value is a prob, not a label
        out
    }
    let mut out = Vec::new();
    let s = s("predict-prob ") + model + " " + inp + " " + &k.to_string();
    let r = wrap_install(&s);
    for l in String::from_utf8_lossy(&r.stdout).split("\n") {
        let v = ext(l);
        if v.len() > 0 {
            out.push(v);
        }
    }
    out
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

/// Provides minimal functionality for generating skipgrams.
///
/// Full documentation from fastText:
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


/// Provides minimal functionality for generating a continuous bag of words model.
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
pub fn min_cbow(input: &str, output: &str) -> String {
    let st = s("cbow -input ") + input + " -output " + output;
    if wrap_install(&st).status.success() {
        s(output) + ".bin"
    } else {
        panic!("Cbow failed with given input: {}", st)
    }
}


fn resp(sm: &str, stdout: std::borrow::Cow<str>) -> Vec<Vec<(String, f64)>> {
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
            } else if l == sm || (lar.len() == 4 && !first) {
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

/// runs an arbitrary command and makes sure that the fasttext is set up locally.
fn run_cmd(cmd: &str) -> Output {
    if DEBUG { println!("cmd: {}", cmd); }
    let r = Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .stdout(Stdio::piped())
        .output()
        .expect("failed to execute process");
    if !r.status.success() && !Path::new("./fasttext").exists() {
        let ir = install();
        for o in ir.iter() {
            if !o.status.success() { panic!("Missing files / executable"); }
        }
    }
    if DEBUG { println!("{:?}", r); }
    r
}


/// Nearest neighbors. Input of "words" are single words separated by spaces.
///
/// Full documentation from FastText:
///
/// usage: fasttext nn <model> <k>
///
///  <model>      model filename
///  <k>          (optional; 10 by default) predict top k labels
pub fn nn(words: &str, model: &str, k: u32) -> Vec<Vec<(String, f64)>> {
    if DEBUG { println!("NN begun") };
    let cmd = s("echo ") + words + " | ./fasttext nn " + model + " " + &k.to_string();
    resp("Query word? ", String::from_utf8_lossy(&run_cmd(&cmd).stdout))
}

/// Access to the analogies function. Not supported.
///
/// Documentation from fastText:
///
/// usage: fasttext analogies <model> <k>
///
///  <model>      model filename
///  <k>          (optional; 10 by default) predict top k labels
pub fn analogies(analogies: &str, model: &str, k: u32) -> Vec<Vec<(String, f64)>> {
    unimplemented!();
    let cmd = s("echo \"") + analogies + "\" | ./fasttext analogies " + model + " " + &k.to_string();
    // just doing the "echo "cmd" | ./fasttext [...]" thing won't work here since it just keeps
    // checking stdin and re-outputting results.
    resp("Query triplet (A - B + C)? ", String::from_utf8_lossy(&run_cmd(&cmd).stdout))
}

fn parse_vec(cmd: &str, sentence: Option<&str>) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    let mut st = String::from_utf8_lossy(&run_cmd(cmd).stdout).to_string();
    match sentence {
        None => (),
        Some(sent) => {
            st = st.replace(sent, "");
        }
    };
    for l in st.split("\n") {
        let mut wordvec = Vec::new();
        let mut f = true;
        for t in l.split(" ") {
            if f {
                f = false;
            } else {
                if t != "" {
                    wordvec.push(t.parse::<f64>().unwrap());
                }
            }
        }
        if wordvec.len() > 0 {
            out.push(wordvec);
        }
    }
    out
}

/// access to the vectors for a given set of words.
///
/// Input: one or more words (separated by spaces)
/// Output: A vec of word vectors (one for each input word)
pub fn word_vector(words: &str, model: &str) -> Vec<Vec<f64>> {
    let cmd = s("echo \"") + words + "\" | ./fasttext print-word-vectors " + model;
    parse_vec(&cmd, None)
}


/// access to the vectors for a given sentence.
///
/// Input: sentence
/// Output: A vec of a sentence vector
pub fn sentence_vector(sentence: &str, model: &str) -> Vec<Vec<f64>> {
    let cmd = s("echo \"") + sentence + "\" | ./fasttext print-sentence-vectors " + model;
    parse_vec(&cmd, Some(sentence))
}


/// the objective for testing here is not to check that the fasttext binary is working as expected,
/// but that it can be install and that its output can be consistently read.

#[cfg(test)]
mod tests {
    extern crate kolmogorov_smirnov as ks;

    use std::{thread, time};
    use std::collections::HashSet;
    use super::*;

    fn check_exists(file: &str, or: fn()) {
        if !Path::new(file).exists() {
            thread::sleep(time::Duration::from_secs(30));
            if !Path::new(file).exists() {
                or()
            }
        }
    }

    fn rm(files: Vec<&str>) {
        for f in files.iter() {
            let cmd = s("rm -r ") + f;
            Command::new("sh")
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


    /// test nearest neighbors for two functions yields valid results.
    fn test_embedding(min_fn: fn(&str, &str) -> String, reg_fn: fn(&HashMap<&str, &str>), min_name: &str, reg_name: &str) {
        inst();

        let input = "sample_text.txt";
        let mut args = HashMap::new();
        args.insert("input", input);
        args.insert("output", reg_name);

        let k = 10;

        // Would iterate through a set of arbitrary words to compare on.
        for w in ["friend", "day", "door"].iter() {
            let m1 = min_fn(input, min_name);
            reg_fn(&args);
            let m2 = s(min_name) + ".bin";

            let r1 = nn(w, &m1, k);
            let r2 = nn(w, &m2, k);

            assert_eq!(r1.len(), r2.len());
            for i in 0..r1.len() {
                assert_eq!(r1[i].len(), r2[i].len());
                assert_eq!(r1[i].len(), k as usize);
            }

            assert!(sim(&set(r1), &set(r2)) > (0.9 * k as f64) as usize);
        }

        let r1 = s(min_name) + "*";
        let r2 = s(reg_name) + "*";
        rm(vec![&r1, &r2]);
    }

    #[test]
    fn test_skipgram() {
        test_embedding(min_skipgram, skipgram, "test_min_skipgram", "test_skipgram");
    }

    #[test]
    fn test_cbow() {
        test_embedding(min_cbow, cbow, "test_min_cbow", "test_cbow");
    }

    fn test_predict(model: String) {
        let p = predict(&model, "t.txt", 1);
        println!("test_predict output: {:?}", p);
        assert_eq!(p[0].len(), 1);
        assert_eq!(p.len(), 2);

        let p = predict(&model, "t.txt", 2);
        println!("test_predict output: {:?}", p);
        assert_eq!(p[0].len(), 2);
        assert_eq!(p.len(), 2);
    }

    fn test_predict_prob(model: String) {
        let p = predict_prob(&model, "t.txt", 1);
        println!("output of predict_prob: {:?}", p);
        assert_eq!(p[0].len(), 1);
        assert_eq!(p.len(), 2);

        let p = predict_prob(&model, "t.txt", 2);
        println!("output of predict_prob: {:?}", p);
        assert_eq!(p[0].len(), 2);
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn test_supervised_and_predicts() {
        inst();

        let model = "sup";

        let args: HashMap<_, _> = vec![
            ("input", "sample_text.txt"),
            ("output", model),
        ].into_iter().collect();

        supervised(&args);

        test_predict(s(model) + ".bin");
        test_predict_prob(s(model) + ".bin");

        let m = s(model) + "*";
        rm(vec![&m]);
    }

    #[test]
    fn test_word_vector() {
        samp();
        let v = word_vector("gay math queen", "sample.bin");
        assert_eq!(v.len(), 3); // three words go in, three wordvecs come out

        let mut hs = HashSet::new();
        for wv in v.iter() {
            hs.insert(wv.len());
        }
        assert_eq!(hs.len(), 1); // vectors are all the same length

        let v = word_vector("naps", "sample.bin");
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_sentence_vector() {
        samp();
        let v = sentence_vector("To die, to sleep – to sleep, perchance to dream – ay, there's the rub, for in this sleep of death what dreams may come…", "sample.bin");
        assert_eq!(v.len(), 1);
    }
}