//! Headless first-run setup: `cargo run -p vk-core --example setup -- [tensorrt|directml|ncnn]`.

use std::sync::mpsc::channel;
use vk_core::script::Backend;
use vk_core::setup;

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "ncnn".into());
    let Some(backend) = Backend::ALL.into_iter().find(|b| b.short().eq_ignore_ascii_case(&name) || b.label().to_lowercase().starts_with(&name)) else {
        eprintln!("usage: setup [tensorrt|directml|ncnn]");
        std::process::exit(2);
    };
    let (tx, rx) = channel();
    let printer = std::thread::spawn(move || rx.into_iter().for_each(|l: String| eprintln!("{l}")));
    let result = setup::run(backend, &tx);
    drop(tx);
    let _ = printer.join();
    match result {
        Ok(vspipe) => println!("{}", vspipe.display()),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
