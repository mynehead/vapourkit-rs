//! Headless run: `cargo run -p vk-core --example upscale -- IN MODEL OUT`
//! (NCNN, fp16; VK_VSPIPE / VK_FFMPEG override the tools on PATH).

use std::sync::mpsc::channel;
use vk_core::encoding::Encoding;
use vk_core::job::{self, Event, Tools};
use vk_core::chain::Step;
use vk_core::script::{self, Backend, ScriptConfig};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [input, model, output] = a.as_slice() else {
        eprintln!("usage: upscale IN MODEL OUT");
        std::process::exit(2);
    };
    let mut tools = Tools::default();
    if let Ok(p) = std::env::var("VK_VSPIPE") { tools.vspipe = p.into(); }
    if let Ok(p) = std::env::var("VK_FFMPEG") { tools.ffmpeg = p.into(); }
    let cfg = ScriptConfig { input: input.into(), steps: vec![Step::model(model.into())], backend: Backend::Ncnn, num_streams: 2 };
    let text = script::render(&cfg, &job::script_path());
    let (tx, rx) = channel();
    let _job = job::start(tools, text, input.into(), Encoding::default().args(), output.into(), tx);
    for ev in rx {
        match ev {
            Event::Log(l) => eprintln!("{l}"),
            Event::Progress { frame, total } => eprintln!("frame {frame}/{total}"),
            Event::Done(r) => {
                if let Err(e) = r { eprintln!("{e}"); std::process::exit(1); }
                return;
            }
        }
    }
}
