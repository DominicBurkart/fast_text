use std::process::{Command, Output, Stdio};

#[macro_use(c)]
extern crate cute;

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
        Command::new("sh").arg("-c").arg("rm fasttext"); // if fasttext exists, remove it
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
    if r.status.code() == Some(127) { // returns 127 if ./fasttext DNE
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

/// Minimal interface for the supervised learning algorithm. Documentation from fastText:
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
pub fn supervised() {
    let r = wrap_install("todo");
    unimplemented!()
}

/// Minimal interface to shrink a model's memory requirements. Full interface from fastText:
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
pub fn quantize(input: &str, output: &str) {
    let s = s("quantize -input ") + input + " -output " + output;
    let r = wrap_install(&s);
    unimplemented!()
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
pub fn skipgram<'a>(input: &str, output: &'a str) -> &'a str {
    let s = s("skipgram -input ") + input + " -output " + output;
    if wrap_install(&s).status.success() {
        output
    } else {
        panic!("Skipgram failed with given input: {}", s)
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
pub fn cbow<'a>(input: &str, output: &'a str) -> &'a str {
    let s = s("cbow -input ") + input + " -output " + output;
    if wrap_install(&s).status.success() {
        output
    } else {
        panic!("Cbow failed with given input: {}", s)
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
pub fn analogies() {
    let r = wrap_install("todo");
    unimplemented!()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_install() {
        use std::process::{Command, Stdio};
        use install;
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
        sample_skipgram();
    }

    /// generate a skipgram model for testing things like the nearest neighbor function.
    fn sample_skipgram() {
        use std::process::{Command, Stdio};
        let r = Command::new("sh")
            .arg("-c")
            .arg("./fasttext skipgram -input sample_text.txt -output sample")
            .stdout(Stdio::piped())
            .output()
            .expect("failed to execute process");
        println!("{}", String::from_utf8_lossy(&r.stdout));
        println!("{}", String::from_utf8_lossy(&r.stderr));
        assert_eq!(r.status.code(), Some(0)); // returns 127 if ./fasttext DNE
    }

    #[test]
    fn test_nn() {
        use nn;

        sample_skipgram();

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
}