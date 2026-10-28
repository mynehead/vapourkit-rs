//! One upscale run: `vspipe -c mkv script.vpy -` piped into ffmpeg, as in
//! electron/upscaleExecutor.ts and electron/ffmpegPipeArgs.ts.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::script::{self, ScriptConfig};

#[derive(Clone, Debug)]
pub struct Tools {
    pub vspipe: PathBuf,
    pub ffmpeg: PathBuf,
}

impl Default for Tools {
    /// Both from PATH until first-run setup (milestone 3) installs its own.
    fn default() -> Self {
        Tools { vspipe: "vspipe".into(), ffmpeg: "ffmpeg".into() }
    }
}

#[derive(Debug)]
pub enum Event {
    Log(String),
    Progress { frame: u64, total: u64 },
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

pub fn start(tools: Tools, cfg: ScriptConfig, output: PathBuf, tx: Sender<Event>) -> Job {
    let children = Arc::new(Mutex::new(Vec::new()));
    let job = Job { children: children.clone() };
    thread::spawn(move || {
        let result = run(&tools, &cfg, &output, &tx, &children);
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

/// ffmpeg's arguments: the Matroska stream from vspipe tagged BT.709 limited,
/// audio and subtitles copied from the source, progress on stdout.
fn ffmpeg_args(input: &Path, output: &Path) -> Vec<String> {
    let mut a: Vec<String> = [
        "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1",
        "-f", "matroska", "-color_range", "tv", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-i", "pipe:0",
    ]
    .map(String::from)
    .to_vec();
    a.push("-i".into());
    a.push(input.to_string_lossy().into_owned());
    // shortcut: fixed x264 CRF 18 encode; encoder presets arrive with settings (milestone 2).
    for s in [
        "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy", "-map", "1:s?", "-c:s", "copy",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-fps_mode:v:0", "passthrough", "-y",
    ] {
        a.push(s.into());
    }
    a.push(output.to_string_lossy().into_owned());
    a
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
    cfg: &ScriptConfig,
    output: &Path,
    tx: &Sender<Event>,
    children: &Mutex<Vec<Child>>,
) -> Result<(), String> {
    let work_dir = std::env::temp_dir().join(format!("vapourkit-{}", std::process::id()));
    std::fs::create_dir_all(&work_dir).map_err(|e| format!("Cannot create {}: {e}", work_dir.display()))?;
    let script_path = work_dir.join("run.vpy");
    std::fs::write(work_dir.join("vk_runtime.py"), script::RUNTIME_PY).map_err(|e| e.to_string())?;
    std::fs::write(&script_path, script::render(cfg, &script_path)).map_err(|e| e.to_string())?;
    let _ = tx.send(Event::Log(format!("Script: {}", script_path.display())));

    // Evaluating the script once also surfaces Python errors before ffmpeg starts.
    let info = Command::new(&tools.vspipe)
        .args(["-i".as_ref(), script_path.as_os_str(), "-".as_ref()])
        .output()
        .map_err(|e| format!("Cannot run {}: {e}", tools.vspipe.display()))?;
    let info_text = format!("{}{}", String::from_utf8_lossy(&info.stdout), String::from_utf8_lossy(&info.stderr));
    if !info.status.success() {
        return Err(format!("vspipe could not evaluate the script:\n{info_text}"));
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
        .args(ffmpeg_args(&cfg.input, output))
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
    let progress = ffmpeg.stdout.take().expect("piped");
    forward_lines(ffmpeg.stderr.take().expect("piped"), tx.clone(), "ffmpeg");
    children.lock().unwrap().extend([vspipe, ffmpeg]);

    for line in BufReader::new(progress).lines().map_while(Result::ok) {
        if let Some(frame) = line.strip_prefix("frame=").and_then(|n| n.trim().parse().ok()) {
            let _ = tx.send(Event::Progress { frame, total });
        }
    }

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
    fn reads_frame_count() {
        let info = "Width: 1920\nHeight: 1080\nFrames: 2400\nFPS: 24000/1001 (23.976 fps)\n";
        assert_eq!(parse_frames(info), Some(2400));
        assert_eq!(parse_frames("garbage"), None);
    }

    #[test]
    fn ffmpeg_reads_pipe_then_source() {
        let a = ffmpeg_args(Path::new("in.mkv"), Path::new("out.mp4"));
        let first = a.iter().position(|s| s == "pipe:0").unwrap();
        let second = a.iter().position(|s| s == "in.mkv").unwrap();
        assert!(first < second);
        assert_eq!(a.last().unwrap(), "out.mp4");
    }
}
