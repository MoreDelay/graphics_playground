use std::{env, path::PathBuf, process::Command};

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = PathBuf::from(out_dir);

    let mut cmd = Command::new("glslang");
    cmd.args([
        "-V100",
        "src/shaders/tutorial.vert",
        "-o",
        out_dir.join("vert.spv").to_str().unwrap(),
    ]);
    cmd.spawn().unwrap().wait().unwrap();

    let mut cmd = Command::new("glslang");
    cmd.args([
        "-V100",
        "src/shaders/tutorial.frag",
        "-o",
        out_dir.join("frag.spv").to_str().unwrap(),
    ]);
    cmd.spawn().unwrap().wait().unwrap();
}
