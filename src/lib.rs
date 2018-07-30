use std::process::Command;

fn install() -> std::process::Output {
    if cfg!(target_os = "windows") {
        Command::new("cmd")
            .args(&["/C",
                "wget https://github.com/facebookresearch/fastText/archive/v0.1.0.zip",
                "unzip v0.1.0.zip",
                "cd fastText-0.1.0",
                "make"
            ])
            .output()
            .expect("failed to execute process")
    } else {
        Command::new("sh")
            .arg("-c")
            .args(&["wget https://github.com/facebookresearch/fastText/archive/v0.1.0.zip",
                "unzip v0.1.0.zip",
                "cd fastText-0.1.0",
                "make"])
            .output()
            .expect("failed to execute process")
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
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
