//! One upscale run: `vspipe -c mkv script.vpy -` piped into ffmpeg, as in
//! electron/upscaleExecutor.ts and electron/ffmpegPipeArgs.ts.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::script;

#[derive(Clone, Debug)]
pub struct Tools {
    pub vspipe: PathBuf,
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

impl Default for Tools {
    /// vspipe from setup once it has run, otherwise all from PATH.
    fn default() -> Self {
        let vspipe = crate::setup::installed_vspipe().unwrap_or_else(|| "vspipe".into());
        Tools { vspipe, ffmpeg: "ffmpeg".into(), ffprobe: "ffprobe".into() }
    }
}

#[derive(Debug)]
pub enum Event {
    Log(String),
    Progress { frame: u64, total: u64 },
    /// A JPEG of the output, about once a second.
    Preview(Vec<u8>),
    Done(Result<(), String>),
}

/// A running job; `cancel` kills both processes.
pub struct Job {
    children: Arc<Mutex<Vec<Child>>>,
}

impl Job {
    pub fn cancel(&self) {
        for child in self.children.lock().unwrap().iter_mut() {
            let _ = child.kill();
        }
    }
}

/// `video_args` are the encoder arguments, e.g. from `Encoding::args`.
/// Where a run's script is written; render it for this path (`script::render`).
pub fn script_path() -> PathBuf {
    std::env::temp_dir().join(format!("vapourkit-{}", std::process::id())).join("run.vpy")
}

/// Runs `script` (rendered for `script_path()`, possibly edited by hand);
/// `input` is the source the audio and subtitles are copied from.
pub fn start(tools: Tools, script: String, input: PathBuf, video_args: Vec<String>, output: PathBuf, tx: Sender<Event>) -> Job {
    let children = Arc::new(Mutex::new(Vec::new()));
    let job = Job { children: children.clone() };
    thread::spawn(move || {
        let result = run(&tools, &script, &input, &video_args, &output, &tx, &children);
        let _ = tx.send(Event::Done(result));
    });
    job
}

/// The frame count `vspipe -i` prints as `Frames: N`.
fn parse_frames(info: &str) -> Option<u64> {
    info.lines()
        .find_map(|l| l.trim().strip_prefix("Frames:"))
        .and_then(|n| n.trim().parse().ok())
}

/// What is missing for a plugin namespace VapourSynth reported.
fn plugin_hint(namespace: &str) -> Option<&'static str> {
    Some(match namespace {
        "bs" => "the BestSource plugin, which reads the video",
        "trt" => "vs-mlrt's TensorRT plugin",
        "ort" => "vs-mlrt's ONNX Runtime plugin, which DirectML needs",
        "ncnn" => "vs-mlrt's NCNN plugin",
        _ => return None,
    })
}

/// How a failure caused by a missing known plugin starts, so the UI can offer setup.
pub const MISSING_PLUGIN: &str = "VapourSynth is missing";

/// A plain message for a script that failed because a plugin is missing.
fn explain_failure(info: &str) -> Option<String> {
    let rest = info.split("No attribute with the name ").nth(1)?;
    let namespace = rest.split_whitespace().next()?;
    Some(match plugin_hint(namespace) {
        Some(hint) => format!("{MISSING_PLUGIN} {hint}. Setup installs it for the chosen backend."),
        None => format!("VapourSynth has no plugin with namespace \"{namespace}\", which a filter step uses. Install it or remove that step."),
    })
}

/// ffmpeg's arguments: the Matroska stream from vspipe tagged BT.709 limited,
/// audio and subtitles copied from the source, progress on stderr and a
/// 1 fps MJPEG preview on stdout, as upstream does.
fn ffmpeg_args(input: &Path, video_args: &[String], output: &Path) -> Vec<String> {
    let mut a: Vec<String> = [
        "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:2",
        "-f", "matroska", "-color_range", "tv", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-i", "pipe:0",
    ]
    .map(String::from)
    .to_vec();
    a.push("-i".into());
    a.push(input.to_string_lossy().into_owned());
    for s in ["-map", "0:v:0", "-map", "1:a?", "-c:a", "copy", "-map", "1:s?", "-c:s", "copy"] {
        a.push(s.into());
    }
    a.extend_from_slice(video_args);
    for s in ["-fps_mode:v:0", "passthrough", "-y"] {
        a.push(s.into());
    }
    a.push(output.to_string_lossy().into_owned());
    for s in ["-map", "0:v:0", "-vf", "fps=1,scale=-2:'min(720,ih)'", "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "2", "pipe:1"] {
        a.push(s.into());
    }
    a
}

/// Takes the complete JPEGs (SOI 0xFFD8 to EOI 0xFFD9) off the front of `buf`.
fn take_jpegs(buf: &mut Vec<u8>) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let Some(start) = buf.windows(2).position(|w| w == [0xFF, 0xD8]) else { break };
        let Some(len) = buf[start..].windows(2).position(|w| w == [0xFF, 0xD9]) else { break };
        out.push(buf[start..start + len + 2].to_vec());
        buf.drain(..start + len + 2);
    }
    out
}

/// A line of ffmpeg's `-progress` output (`key=value`, lowercase key).
fn is_progress_line(line: &str) -> bool {
    line.split_once('=').is_some_and(|(k, _)| !k.is_empty() && k.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'))
}

fn forward_lines(r: impl Read + Send + 'static, tx: Sender<Event>, tag: &'static str) {
    thread::spawn(move || {
        for line in BufReader::new(r).lines().map_while(Result::ok) {
            let _ = tx.send(Event::Log(format!("[{tag}] {line}")));
        }
    });
}

fn run(
    tools: &Tools,
    script: &str,
    input: &Path,
    video_args: &[String],
    output: &Path,
    tx: &Sender<Event>,
    children: &Mutex<Vec<Child>>,
) -> Result<(), String> {
    let script_path = script_path();
    let work_dir = script_path.parent().expect("script_path has a folder");
    std::fs::create_dir_all(work_dir).map_err(|e| format!("Cannot create {}: {e}", work_dir.display()))?;
    std::fs::write(work_dir.join("vk_runtime.py"), script::RUNTIME_PY).map_err(|e| e.to_string())?;
    std::fs::write(&script_path, script).map_err(|e| e.to_string())?;
    let _ = tx.send(Event::Log(format!("Script: {}", script_path.display())));

    // Evaluating the script once also surfaces Python errors before ffmpeg starts.
    let info = Command::new(&tools.vspipe)
        .args(["-i".as_ref(), script_path.as_os_str(), "-".as_ref()])
        .output()
        .map_err(|e| format!("Cannot run {}: {e}", tools.vspipe.display()))?;
    let info_text = format!("{}{}", String::from_utf8_lossy(&info.stdout), String::from_utf8_lossy(&info.stderr));
    if !info.status.success() {
        return Err(match explain_failure(&info_text) {
            Some(plain) => format!("{plain}\n\n{info_text}"),
            None => format!("vspipe could not evaluate the script:\n{info_text}"),
        });
    }
    let total = parse_frames(&info_text).unwrap_or(0);
    let _ = tx.send(Event::Log(format!("Frames: {total}")));

    let mut vspipe = Command::new(&tools.vspipe)
        .args(["-c".as_ref(), "mkv".as_ref(), "--outputindex".as_ref(), "0".as_ref(), script_path.as_os_str(), "-".as_ref()])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Cannot run {}: {e}", tools.vspipe.display()))?;
    let pipe = vspipe.stdout.take().expect("piped");
    forward_lines(vspipe.stderr.take().expect("piped"), tx.clone(), "vspipe");
    let mut ffmpeg = match Command::new(&tools.ffmpeg)
        .args(ffmpeg_args(input, video_args, output))
        .stdin(Stdio::from(pipe))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = vspipe.kill();
            return Err(format!("Cannot run {}: {e}", tools.ffmpeg.display()));
        }
    };
    let mut preview = ffmpeg.stdout.take().expect("piped");
    let stderr = ffmpeg.stderr.take().expect("piped");
    children.lock().unwrap().extend([vspipe, ffmpeg]);

    let progress_tx = tx.clone();
    let progress = thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if let Some(frame) = line.strip_prefix("frame=").and_then(|n| n.trim().parse().ok()) {
                let _ = progress_tx.send(Event::Progress { frame, total });
            } else if !is_progress_line(&line) {
                let _ = progress_tx.send(Event::Log(format!("[ffmpeg] {line}")));
            }
        }
    });
    let mut buf = Vec::new();
    let mut chunk = [0u8; 64 * 1024];
    while let Ok(n) = preview.read(&mut chunk) {
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        for jpeg in take_jpegs(&mut buf) {
            let _ = tx.send(Event::Preview(jpeg));
        }
    }
    let _ = progress.join();

    let mut ok = true;
    for child in children.lock().unwrap().iter_mut() {
        ok &= child.wait().map(|s| s.success()).unwrap_or(false);
    }
    children.lock().unwrap().clear();
    if ok { Ok(()) } else { Err("vspipe or ffmpeg exited with an error (see log)".into()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explains_missing_plugins() {
        let err = "AttributeError: No attribute with the name bs exists. Did you mistype a plugin namespace";
        assert!(explain_failure(err).unwrap().contains("BestSource"));
        assert!(explain_failure("No attribute with the name ort exists.").unwrap().starts_with(MISSING_PLUGIN));
        assert!(explain_failure("No attribute with the name cas exists.").unwrap().contains("\"cas\""));
        assert_eq!(explain_failure("SyntaxError: invalid syntax"), None);
    }

    #[test]
    fn splits_jpegs() {
        let mut buf = vec![0xFF, 0xD8, 1, 2, 0xFF, 0xD9, 0xFF, 0xD8, 3];
        assert_eq!(take_jpegs(&mut buf), [vec![0xFF, 0xD8, 1, 2, 0xFF, 0xD9]]);
        assert_eq!(buf, [0xFF, 0xD8, 3]);
        buf.extend([0xFF, 0xD9]);
        assert_eq!(take_jpegs(&mut buf).len(), 1);
        assert!(buf.is_empty());
    }

    #[test]
    fn tells_progress_from_log_lines() {
        assert!(is_progress_line("out_time_us=1000"));
        assert!(is_progress_line("progress=continue"));
        assert!(!is_progress_line("x265 [info]: frame I: 1, Avg QP=19"));
        assert!(!is_progress_line("Error opening output file"));
    }

    #[test]
    fn reads_frame_count() {
        let info = "Width: 1920\nHeight: 1080\nFrames: 2400\nFPS: 24000/1001 (23.976 fps)\n";
        assert_eq!(parse_frames(info), Some(2400));
        assert_eq!(parse_frames("garbage"), None);
    }

    #[test]
    fn ffmpeg_reads_pipe_then_source() {
        let a = ffmpeg_args(Path::new("in.mkv"), &["-c:v".into(), "libx265".into()], Path::new("out.mp4"));
        let first = a.iter().position(|s| s == "pipe:0").unwrap();
        let second = a.iter().position(|s| s == "in.mkv").unwrap();
        let codec = a.iter().position(|s| s == "libx265").unwrap();
        assert!(first < second && second < codec);
        let out = a.iter().position(|s| s == "out.mp4").unwrap();
        assert!(codec < out && a.last().unwrap() == "pipe:1");
    }
}
