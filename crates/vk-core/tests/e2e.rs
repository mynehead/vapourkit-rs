//! A real run through vspipe and ffmpeg. Ignored by default; run with
//! `VK_E2E_INPUT=clip.mkv VK_E2E_MODEL=model.onnx cargo test -p vk-core -- --ignored`
//! (VK_VSPIPE / VK_FFMPEG override the tools on PATH).

use std::collections::BTreeMap;
use std::process::Command;
use std::sync::mpsc::channel;

use vk_core::chain::{bundled_filters, Param, Step, StepKind};
use vk_core::encoding::Encoding;
use vk_core::job::{self, Event, Tools};
use vk_core::script::{self, Backend, ScriptConfig};

#[test]
#[ignore]
fn model_then_crop_then_custom_code() {
    let input = std::env::var("VK_E2E_INPUT").expect("VK_E2E_INPUT");
    let model = std::env::var("VK_E2E_MODEL").expect("VK_E2E_MODEL");
    let mut crop = bundled_filters().into_iter().find(|t| t.name == "Crop").unwrap().to_step();
    if let StepKind::Custom { parameters, .. } = &mut crop.kind {
        parameters.insert("crop_left".into(), Param::Num(40.0));
    }
    let invert = Step {
        name: "Invert".into(),
        enabled: true,
        kind: StepKind::Custom { code: "clip = core.std.Invert(clip)".into(), variables: BTreeMap::new(), parameters: BTreeMap::new() },
    };
    let cfg = ScriptConfig { input: input.clone().into(), steps: vec![Step::model(model.into()), crop, invert], backend: Backend::Ncnn, num_streams: 1 };
    let mut tools = Tools::default();
    if let Ok(p) = std::env::var("VK_VSPIPE") { tools.vspipe = p.into(); }
    if let Ok(p) = std::env::var("VK_FFMPEG") { tools.ffmpeg = p.into(); }
    let output = std::env::temp_dir().join("vk-e2e-out.mkv");
    let (tx, rx) = channel();
    let enc = Encoding { codec: vk_core::encoding::Codec::H265, ..Default::default() };
    let _job = job::start(tools, script::render(&cfg, &job::script_path()), input.into(), enc.args(), output.clone(), tx);
    let result = rx.iter().find_map(|e| match e {
        Event::Done(r) => Some(r),
        Event::Log(l) => { eprintln!("{l}"); None }
        _ => None,
    });
    result.unwrap().expect("job failed");
    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width", "-of", "csv=p=0"])
        .arg(&output)
        .output()
        .unwrap();
    let probe = String::from_utf8_lossy(&probe.stdout);
    eprintln!("ffprobe: {probe}");
    assert!(probe.starts_with("hevc,"), "{probe}");
}
