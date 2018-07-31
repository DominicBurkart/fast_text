use std::process::{Command, Stdio};

#[macro_use(c)]
extern crate cute;

fn s(v: &str) -> String {
    v.to_string()
}

pub fn install() -> Vec<std::process::Output> {
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");
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

pub fn supervised() {}

pub fn quantize() {}

pub fn predict() {}

pub fn skipgram() {}

pub fn nn() {}

pub fn analogies() {}

pub fn dump() {}

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
