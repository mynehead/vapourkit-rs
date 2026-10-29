//! First-run setup, as upstream's dependencyManager.ts and pluginInstaller.ts:
//! a private Python with VapourSynth, the vs-mlrt plugin for a backend and
//! everything the bundled filters use (the PyPI plugin ecosystem, PyTorch, the
//! Hybrid and bundled scripts, the vs-mlrt model zoo, and on Windows the
//! plugin DLLs upstream ships in plugins.7z). On Windows the Python is the
//! embedded one from python.org, elsewhere a venv of the host Python.

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
/// Upstream's include folder at the commit this port follows: plugins.7z and extra_scripts.7z.
const UPSTREAM: &str = "https://github.com/Kim2091/vapourkit/raw/2c515ccb7e77094c81278f3c52b2c10473a33ea8/include";
/// Selur's Hybrid scripts (havsfunc and friends), at upstream's pin.
const HYBRID_SCRIPTS: &str =
    "https://github.com/Selur/VapoursynthScriptsInHybrid/archive/d430e1973a78c2dc52a6e4aa58e5f89cc0093ae9.zip";
/// The vs-mlrt model zoo the RIFE and DPIR filters use, with a file each pack has.
const MODEL_PACKS: [(&str, &str); 2] = [
    ("https://github.com/AmusementClub/vs-mlrt/releases/download/external-models/rife_v4.10.7z", "rife/rife_v4.10.onnx"),
    ("https://github.com/AmusementClub/vs-mlrt/releases/download/model-20211209/dpir_v3.7z", "dpir/drunet_color.onnx"),
];
/// Bundled DLLs that PyPI wheels now install under another name, plus
/// fft3dfilter, which aborts R79 at autoload (upstream legacyCleanup.ts).
const SUPERSEDED_PLUGINS: [&str; 25] = [
    "vsort.dll", "vsort", "vstrt.dll", "vsmlrt-cuda", "bestsource.dll", "akarin.dll", "vszip.dll", "zsmooth.dll",
    "manipmv.dll", "hysteresis.dll", "dfttest2_cuda.dll", "bm3dcuda_rtc.dll", "BilateralGPU_RTC.dll",
    "adaptivegrain_rs.dll", "libfillborders.dll", "vsznedi3.dll", "libsangnom.dll", "libawarpsharp2.dll",
    "libresize2.dll", "EEDI3m.dll", "libmvtools.dll", "libscxvid.dll", "fft3dfilter.dll", "bm3dhip", "dfttest2_hip",
];
/// Script modules PyPI now provides; copies in vs-scripts would shadow them.
const SUPERSEDED_SCRIPTS: [&str; 10] = [
    "vs_temporalfix", "vs_undistort", "vs_colorfix", "vs_colorfix.py", "vs_grain", "vs_grain.py", "vs_tiletools",
    "vs_tiletools.py", "dfttest2.py", "__pycache__",
];

/// The vs-mlrt wheel a backend needs, at upstream's pins.
pub fn backend_package(backend: Backend) -> &'static str {
    match backend {
        Backend::TensorRt => "vapoursynth-mlrt-trt==16.1",
        Backend::DirectMl => "vapoursynth-mlrt-ort==16.1",
        Backend::Ncnn => "vapoursynth-mlrt-ncnn==15.16",
    }
}

/// What pip installs: the core, the backend wheel and the plugin ecosystem
/// (upstream vendorPackages.ts `getPypiPackages`, without vsview, whose
/// previewer this app does not use). TensorRT stands in for an NVIDIA card.
fn packages(backend: Backend) -> Vec<&'static str> {
    let nvidia = backend == Backend::TensorRt;
    let mut p = vec![
        VAPOURSYNTH,
        "vapoursynth-bestsource",
        backend_package(backend),
        if nvidia { "vsjetpack[full,nvidia]" } else { "vsjetpack[full,cl,vulkan]" },
        "vs_temporalfix",
        "vs_grain",
        "vs_tiletools>=1.0.0.dev0",
        if nvidia { "vs_colorfix[tensorrt]" } else { "vs_colorfix" },
        "vapoursynth-mvtools",
        "vapoursynth-cas",
        "vapoursynth-adaptivegrain",
        "vapoursynth-wnnm",
        "vapoursynth-scxvid",
        "vapoursynth-dctfilter",
        "vapoursynth-timecube",
        "positional-encodings",
        "einops",
        "timm",
    ];
    if nvidia {
        p.extend(["vs_undistort>=2.3.1", "vapoursynth-nlm-cuda"]);
    }
    p
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

/// The vs-mlrt model zoo, for `vsmlrt.models_path`.
pub fn models_dir() -> PathBuf {
    data_dir().join("vsmlrt-models")
}

fn python(env: &Path) -> PathBuf {
    if cfg!(windows) { env.join("python.exe") } else { env.join("bin/python3") }
}

fn vspipe_in(env: &Path) -> PathBuf {
    if cfg!(windows) { env.join(r"Lib\site-packages\vapoursynth\vspipe.exe") } else { env.join("bin/vspipe") }
}

fn site_packages(env: &Path) -> Result<PathBuf, String> {
    if cfg!(windows) {
        return Ok(env.join(r"Lib\site-packages"));
    }
    // A venv's lib/pythonX.Y follows the host Python's version.
    let lib = std::fs::read_dir(env.join("lib")).map_err(|e| format!("Cannot read {}: {e}", env.display()))?;
    lib.filter_map(Result::ok)
        .map(|e| e.path().join("site-packages"))
        .find(|p| p.is_dir())
        .ok_or_else(|| format!("No site-packages in {}", env.display()))
}

/// The vspipe setup installed, if setup has run.
pub fn installed_vspipe() -> Option<PathBuf> {
    Some(vspipe_in(&env_dir())).filter(|p| p.is_file())
}

/// Installs everything, or tops it up with `backend`'s plugin; safe to run
/// again. Each output line goes to `log`. Returns the vspipe to use.
pub fn run(backend: Backend, log: &Sender<String>) -> Result<PathBuf, String> {
    let env = env_dir();
    let tmp = data_dir().join("tmp");
    let _ = std::fs::remove_dir_all(&tmp);
    for d in [&env, &tmp, &env.join("vs-scripts")] {
        std::fs::create_dir_all(d).map_err(|e| format!("Cannot create {}: {e}", d.display()))?;
    }
    let py = python(&env);
    let say = |s: &str| {
        let _ = log.send(s.to_string());
    };

    if cfg!(windows) {
        if !py.is_file() || !env.join("python313.zip").is_file() {
            let zip = tmp.join("python-embed.zip");
            say("Downloading Python 3.13…");
            download(PYTHON_ZIP, &zip, log)?;
            say("Extracting Python…");
            // tar.exe ships with Windows 10 and later; named by path so a Git or MSYS tar on PATH is not used.
            exec(Command::new(system32("tar.exe")).arg("-xf").arg(&zip).arg("-C").arg(&env), log)?;
        }
        // The embedded Python reads its import paths from this file only.
        std::fs::write(env.join("python313._pth"), "python313.zip\r\n.\r\nLib\\site-packages\r\nvs-scripts\r\n")
            .map_err(|e| format!("Cannot write python313._pth: {e}"))?;
        if !env.join(r"Lib\site-packages\pip").is_dir() {
            let get_pip = tmp.join("get-pip.py");
            say("Installing pip…");
            download(GET_PIP, &get_pip, log)?;
            exec(Command::new(&py).arg(&get_pip).arg("--no-warn-script-location").args(NETWORK), log)?;
        }
    } else if !py.is_file() {
        say("Creating a Python environment…");
        exec(Command::new("python3").args(["-m", "venv"]).arg(&env), log)
            .map_err(|e| format!("{e}\nPython 3.12 or newer with venv is needed (on Debian and Ubuntu: apt install python3-venv)."))?;
    }
    let site = site_packages(&env)?;
    if !cfg!(windows) {
        // A venv reads .pth files from site-packages; this one puts vs-scripts on the import path.
        let line = format!("{}\n", env.join("vs-scripts").display());
        std::fs::write(site.join("vapourkit-vs-scripts.pth"), line).map_err(|e| format!("Cannot write the .pth file: {e}"))?;
    }

    // PyTorch first, from its own index: the CUDA build for NVIDIA, otherwise
    // the CPU build (PyPI's default on Windows, the CPU index on Linux).
    say("Installing PyTorch (used by a few filters)…");
    let torch_index: &[&str] = match backend {
        Backend::TensorRt => &["--index-url", "https://download.pytorch.org/whl/cu130"],
        _ if !cfg!(windows) => &["--index-url", "https://download.pytorch.org/whl/cpu"],
        _ => &[],
    };
    pip(&py, &["torch", "torchvision"], torch_index, log)?;

    // Plugin DLLs without a PyPI wheel. Before pip, so a wheel's same-named DLL replaces the bundled one.
    let plugins = site.join("vapoursynth").join("plugins");
    if cfg!(windows) {
        say("Downloading the plugin bundle…");
        let archive = tmp.join("plugins.7z");
        download(&format!("{UPSTREAM}/plugins/plugins.7z"), &archive, log)?;
        let staging = tmp.join("plugins");
        extract_7z(&archive, &staging)?;
        copy_tree(&staging, &plugins, false)?;
    }

    say("Installing VapourSynth and its plugins…");
    pip(&py, &packages(backend), &EXTRA_INDEX, log)?;
    if backend != Backend::TensorRt {
        // Its declared TensorRT stack (3.8 GB) is only usable on NVIDIA; the PyTorch path works without it.
        exec(Command::new(&py).args(["-m", "pip", "install", "--upgrade", "--no-deps", "--no-warn-script-location"]).args(NETWORK).arg("vs_undistort>=2.3.1"), log)?;
    }
    remove_all(&plugins, &SUPERSEDED_PLUGINS);

    say("Installing the filter scripts…");
    let scripts = env.join("vs-scripts");
    let hybrid = tmp.join("hybrid.zip");
    download(HYBRID_SCRIPTS, &hybrid, log)?;
    exec(Command::new(&py).args(["-m", "zipfile", "-e"]).arg(&hybrid).arg(tmp.join("hybrid")), log)?;
    // The Hybrid repository keeps its modules in folders; vs-scripts takes them flat, as upstream.
    for f in files_in(&tmp.join("hybrid")).into_iter().filter(|f| f.extension().is_some_and(|e| e == "py")) {
        std::fs::copy(&f, scripts.join(f.file_name().unwrap())).map_err(|e| format!("Cannot copy {}: {e}", f.display()))?;
    }
    let extra = tmp.join("extra_scripts.7z");
    download(&format!("{UPSTREAM}/scripts/extra_scripts.7z"), &extra, log)?;
    extract_7z(&extra, &tmp.join("extra"))?;
    copy_tree(&tmp.join("extra"), &scripts, true)?;
    remove_all(&scripts, &SUPERSEDED_SCRIPTS);

    for (url, marker) in MODEL_PACKS {
        if !models_dir().join(marker).is_file() {
            say(&format!("Downloading models: {}…", url.rsplit('/').next().unwrap_or(url)));
            let archive = tmp.join("models.7z");
            download(url, &archive, log)?;
            extract_7z(&archive, &models_dir())?;
        }
    }

    if !cfg!(windows) {
        // Tells vspipe where libpython is (~/.config/vapoursynth/vapoursynth.toml);
        // Windows finds python.exe beside the package instead.
        exec(Command::new(&py).args(["-m", "vapoursynth", "config"]), log)?;
    }
    let _ = std::fs::remove_dir_all(&tmp);
    let vspipe = vspipe_in(&env);
    if !vspipe.is_file() {
        return Err(format!("Setup finished but {} is missing.", vspipe.display()));
    }
    say("Setup done.");
    Ok(vspipe)
}

fn system32(exe: &str) -> PathBuf {
    PathBuf::from(std::env::var_os("SystemRoot").unwrap_or("C:\\Windows".into())).join("System32").join(exe)
}

fn download(url: &str, dest: &Path, log: &Sender<String>) -> Result<(), String> {
    // curl.exe ships with Windows 10 and later.
    let curl = if cfg!(windows) { system32("curl.exe") } else { "curl".into() };
    exec(Command::new(curl).args(["-fsSL", "--retry", "3", "-o"]).arg(dest).arg(url), log)
}

fn pip(py: &Path, packages: &[&str], index: &[&str], log: &Sender<String>) -> Result<(), String> {
    exec(
        Command::new(py)
            .args(["-m", "pip", "install", "--upgrade", "--no-warn-script-location"])
            .args(NETWORK)
            .args(index)
            .args(packages),
        log,
    )
}

fn extract_7z(archive: &Path, dest: &Path) -> Result<(), String> {
    sevenz_rust2::decompress_file(archive, dest).map_err(|e| format!("Cannot extract {}: {e}", archive.display()))
}

/// Every file under `dir`.
fn files_in(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() { out.extend(files_in(&p)) } else { out.push(p) }
    }
    out
}

/// Copies the files under `from` to the same places under `to`; existing
/// ones are kept unless `overwrite`.
fn copy_tree(from: &Path, to: &Path, overwrite: bool) -> Result<(), String> {
    for f in files_in(from) {
        let target = to.join(f.strip_prefix(from).unwrap());
        if overwrite || !target.exists() {
            std::fs::create_dir_all(target.parent().unwrap()).map_err(|e| e.to_string())?;
            std::fs::copy(&f, &target).map_err(|e| format!("Cannot copy to {}: {e}", target.display()))?;
        }
    }
    Ok(())
}

fn remove_all(dir: &Path, names: &[&str]) {
    for n in names {
        let p = dir.join(n);
        let _ = if p.is_dir() { std::fs::remove_dir_all(&p) } else { std::fs::remove_file(&p) };
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_tree_keeps_existing_unless_told() {
        let base = std::env::temp_dir().join(format!("vk-copy-{}", std::process::id()));
        let (from, to) = (base.join("from"), base.join("to"));
        std::fs::create_dir_all(from.join("sub")).unwrap();
        std::fs::create_dir_all(&to).unwrap();
        std::fs::write(from.join("a.dll"), "new").unwrap();
        std::fs::write(from.join("sub/b.dll"), "new").unwrap();
        std::fs::write(to.join("a.dll"), "pip").unwrap();
        copy_tree(&from, &to, false).unwrap();
        assert_eq!(std::fs::read_to_string(to.join("a.dll")).unwrap(), "pip");
        assert_eq!(std::fs::read_to_string(to.join("sub/b.dll")).unwrap(), "new");
        copy_tree(&from, &to, true).unwrap();
        assert_eq!(std::fs::read_to_string(to.join("a.dll")).unwrap(), "new");
        let _ = std::fs::remove_dir_all(base);
    }
}
