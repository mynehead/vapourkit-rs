//! First-run setup, as electron/dependencyManager.ts: a private Python with
//! VapourSynth, BestSource and the vs-mlrt plugin for a backend, all from pip.
//! On Windows that is the embedded Python from python.org, elsewhere a venv
//! of the host Python.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::Sender;
use std::thread;

use crate::script::Backend;

/// Newer cores no longer load the ort and trt plugins (upstream vapoursynthCore.ts).
pub const VAPOURSYNTH: &str = "vapoursynth==79";
const PYTHON_ZIP: &str = "https://www.python.org/ftp/python/3.13.0/python-3.13.0-embed-amd64.zip";
const GET_PIP: &str = "https://bootstrap.pypa.io/get-pip.py";
/// pypi.nvidia.com has TensorRT, the JET index the plugin wheels not on PyPI.
const EXTRA_INDEX: [&str; 4] = [
    "--extra-index-url", "https://pypi.nvidia.com/",
    "--extra-index-url", "https://jaded-encoding-thaumaturgy.github.io/vs-wheels/simple",
];
const NETWORK: [&str; 6] = ["--retries", "5", "--timeout", "30", "--progress-bar", "off"];

/// The vs-mlrt wheel a backend needs, at upstream's pins.
pub fn backend_package(backend: Backend) -> &'static str {
    match backend {
        Backend::TensorRt => "vapoursynth-mlrt-trt==16.1",
        Backend::DirectMl => "vapoursynth-mlrt-ort==16.1",
        Backend::Ncnn => "vapoursynth-mlrt-ncnn==15.16",
    }
}

/// Beside the exe on Windows, as upstream's portable layout; the user's data
/// folder elsewhere.
pub fn data_dir() -> PathBuf {
    if cfg!(windows) {
        let exe = std::env::current_exe().unwrap_or_default();
        return exe.parent().unwrap_or(Path::new(".")).join("data");
    }
    let base = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local/share")
    });
    base.join("vapourkit")
}

pub fn env_dir() -> PathBuf {
    data_dir().join("vapoursynth-portable")
}

fn python(env: &Path) -> PathBuf {
    if cfg!(windows) { env.join("python.exe") } else { env.join("bin/python3") }
}

fn vspipe_in(env: &Path) -> PathBuf {
    if cfg!(windows) { env.join(r"Lib\site-packages\vapoursynth\vspipe.exe") } else { env.join("bin/vspipe") }
}

/// The vspipe setup installed, if setup has run.
pub fn installed_vspipe() -> Option<PathBuf> {
    Some(vspipe_in(&env_dir())).filter(|p| p.is_file())
}

/// Installs the environment, or tops it up with `backend`'s plugin; safe to
/// run again. Each output line goes to `log`. Returns the vspipe to use.
pub fn run(backend: Backend, log: &Sender<String>) -> Result<PathBuf, String> {
    let env = env_dir();
    std::fs::create_dir_all(&env).map_err(|e| format!("Cannot create {}: {e}", env.display()))?;
    let py = python(&env);

    if cfg!(windows) {
        // curl.exe and tar.exe ship with Windows 10 and later; named by path so
        // a Git or MSYS tar on PATH is not picked up.
        let sys32 = PathBuf::from(std::env::var_os("SystemRoot").unwrap_or("C:\\Windows".into())).join("System32");
        if !py.is_file() || !env.join("python313.zip").is_file() {
            let zip = data_dir().join("python-embed.zip");
            let _ = log.send("Downloading Python 3.13…".into());
            exec(Command::new(sys32.join("curl.exe")).args(["-fL", "-o"]).arg(&zip).arg(PYTHON_ZIP), log)?;
            let _ = log.send("Extracting Python…".into());
            exec(Command::new(sys32.join("tar.exe")).arg("-xf").arg(&zip).arg("-C").arg(&env), log)?;
            let _ = std::fs::remove_file(&zip);
        }
        // The embedded Python reads its import paths from this file only.
        std::fs::write(env.join("python313._pth"), "python313.zip\r\n.\r\nLib\\site-packages\r\n")
            .map_err(|e| format!("Cannot write python313._pth: {e}"))?;
        if !env.join(r"Lib\site-packages\pip").is_dir() {
            let get_pip = data_dir().join("get-pip.py");
            let _ = log.send("Installing pip…".into());
            exec(Command::new(sys32.join("curl.exe")).args(["-fL", "-o"]).arg(&get_pip).arg(GET_PIP), log)?;
            exec(Command::new(&py).arg(&get_pip).arg("--no-warn-script-location").args(NETWORK), log)?;
            let _ = std::fs::remove_file(&get_pip);
        }
    } else if !py.is_file() {
        let _ = log.send("Creating a Python environment…".into());
        exec(Command::new("python3").args(["-m", "venv"]).arg(&env), log)
            .map_err(|e| format!("{e}\nPython 3.12 or newer with venv is needed (on Debian and Ubuntu: apt install python3-venv)."))?;
    }

    let _ = log.send(format!("Installing VapourSynth, BestSource and {}…", backend_package(backend)));
    exec(
        Command::new(&py)
            .args(["-m", "pip", "install", "--upgrade", "--no-warn-script-location"])
            .args(NETWORK)
            .args(EXTRA_INDEX)
            .args([VAPOURSYNTH, "vapoursynth-bestsource", backend_package(backend)]),
        log,
    )?;
    if !cfg!(windows) {
        // Tells vspipe where libpython is (~/.config/vapoursynth/vapoursynth.toml);
        // Windows finds python.exe beside the package instead.
        exec(Command::new(&py).args(["-m", "vapoursynth", "config"]), log)?;
    }
    let vspipe = vspipe_in(&env);
    if !vspipe.is_file() {
        return Err(format!("Setup finished but {} is missing.", vspipe.display()));
    }
    let _ = log.send("Setup done.".into());
    Ok(vspipe)
}

/// Runs `cmd`, sending stdout and stderr line by line; an error if it fails.
fn exec(cmd: &mut Command, log: &Sender<String>) -> Result<(), String> {
    let name = format!("{:?}", cmd.get_program());
    let _ = log.send(format!("> {name} {}", cmd.get_args().map(|a| a.to_string_lossy()).collect::<Vec<_>>().join(" ")));
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Cannot start {name}: {e}"))?;
    let readers = [
        forward(child.stdout.take().unwrap(), log.clone()),
        forward(child.stderr.take().unwrap(), log.clone()),
    ];
    let status = child.wait().map_err(|e| e.to_string())?;
    for r in readers {
        let _ = r.join();
    }
    if status.success() { Ok(()) } else { Err(format!("{name} failed ({status}); the console has its output.")) }
}

fn forward(r: impl Read + Send + 'static, tx: Sender<String>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        for line in BufReader::new(r).lines().map_while(Result::ok) {
            let _ = tx.send(line);
        }
    })
}
