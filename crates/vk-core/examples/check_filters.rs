//! Runs every shipped filter on its own over a video, three frames each, and
//! reports which ones VapourSynth cannot run:
//! `cargo run -p vk-core --example check_filters -- VIDEO [NAME FILTER]`
//! (VK_VSPIPE overrides the vspipe setup installed).

use std::process::Command;
use vk_core::chain;
use vk_core::job::{self, Tools};
use vk_core::script::{self, Backend, ScriptConfig};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let Some(input) = a.first() else {
        eprintln!("usage: check_filters VIDEO [NAME FILTER]");
        std::process::exit(2);
    };
    let vspipe = std::env::var("VK_VSPIPE").map(Into::into).unwrap_or(Tools::default().vspipe);
    let path = job::script_path();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path.with_file_name("vk_runtime.py"), script::RUNTIME_PY).unwrap();
    let (mut ok, mut failed) = (0, Vec::new());
    for t in chain::bundled_filters().into_iter().filter(|t| a.get(1).is_none_or(|n| t.name.contains(n.as_str()))) {
        let cfg = ScriptConfig { input: input.into(), steps: vec![t.to_step()], backend: Backend::Ncnn, num_streams: 1 };
        std::fs::write(&path, script::render(&cfg, &path)).unwrap();
        let out = Command::new(&vspipe).args(["-s", "0", "-e", "2"]).arg(&path).arg("--").output().unwrap();
        if out.status.success() {
            ok += 1;
            println!("ok    {}", t.name);
        } else {
            let err = String::from_utf8_lossy(&out.stderr);
            let last = err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_string();
            println!("FAIL  {} [{}]: {last}", t.name, t.category());
            failed.push(t.name);
        }
    }
    println!("\n{ok} ran, {} failed", failed.len());
}
