use std::process::{Command, Output, Stdio};

#[macro_use(c)]
extern crate cute;

const VERSION: &'static str = env!("CARGO_PKG_VERSION");

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

pub fn supervised() {
    let r = wrap_install("todo");
    unimplemented!()
}

pub fn quantize(model: &str) {
    let s = s("quantize -output ") + model;
    let r = wrap_install(&s);
    unimplemented!()
}

pub fn predict(model: &str, inp: &str, k: u32) {
    let s = s("predict ") + model + " " + inp + " " + &k.to_string();
    let r = wrap_install(&s);
    unimplemented!()
}

pub fn skipgram<'a>(input: &str, output: &'a str) -> &'a str {
    let s = s("skipgram -input ") + input + " -output " + output;
    if wrap_install(&s).status.success() {
        output
    } else {
        panic!("Skipgram failed with given input: {}", s)
    }
}

pub fn nn() {
    let r = wrap_install("todo");
    unimplemented!()
}

pub fn analogies() {
    let r = wrap_install("todo");
    unimplemented!()
}

pub fn dump() {
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
    }
}
