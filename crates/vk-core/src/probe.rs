//! What the source is (ffprobe) and a still of it (ffmpeg), for the UI.

use std::path::Path;
use std::process::Command;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration: f64,
    pub size_bytes: u64,
}

fn parse(text: &str) -> VideoInfo {
    let mut info = VideoInfo::default();
    for (key, value) in text.lines().filter_map(|l| l.split_once('=')) {
        match key {
            "width" => info.width = value.parse().unwrap_or(0),
            "height" => info.height = value.parse().unwrap_or(0),
            "avg_frame_rate" => {
                if let Some((n, d)) = value.split_once('/') {
                    let (n, d): (f64, f64) = (n.parse().unwrap_or(0.0), d.parse().unwrap_or(0.0));
                    info.fps = if d > 0.0 { n / d } else { 0.0 };
                }
            }
            "duration" => info.duration = value.parse().unwrap_or(0.0),
            _ => {}
        }
    }
    info
}

pub fn probe(ffprobe: &Path, input: &Path) -> Result<VideoInfo, String> {
    let out = Command::new(ffprobe)
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate:format=duration", "-of", "default=nw=1"])
        .arg(input)
        .output()
        .map_err(|e| format!("Cannot run {}: {e}", ffprobe.display()))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    let mut info = parse(&String::from_utf8_lossy(&out.stdout));
    info.size_bytes = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);
    Ok(info)
}

/// One frame at `at` seconds as a JPEG, at most 1080 lines tall.
pub fn still(ffmpeg: &Path, input: &Path, at: f64) -> Result<Vec<u8>, String> {
    let out = Command::new(ffmpeg)
        .args(["-v", "error", "-ss", &format!("{at:.3}"), "-i"])
        .arg(input)
        .args(["-frames:v", "1", "-vf", "scale=-2:'min(1080,ih)'", "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "2", "pipe:1"])
        .output()
        .map_err(|e| format!("Cannot run {}: {e}", ffmpeg.display()))?;
    if !out.status.success() || out.stdout.is_empty() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(out.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ffprobe_output() {
        let i = parse("width=3840\nheight=2160\navg_frame_rate=24000/1001\nduration=9.160000\n");
        assert_eq!((i.width, i.height), (3840, 2160));
        assert!((i.fps - 23.976).abs() < 0.001);
        assert!((i.duration - 9.16).abs() < 1e-9);
        assert_eq!(parse("avg_frame_rate=0/0").fps, 0.0);
    }
}
