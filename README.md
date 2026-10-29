# vapourkit-rs

A native Rust port of [Vapourkit](https://github.com/Kim2091/vapourkit), the VapourSynth video upscaling app, without Electron. GUI is [egui](https://github.com/emilk/egui).

Status: early. You pick a video and an output, build a filter chain from AI models (TensorRT, DirectML or NCNN) and VapourSynth filters (161 bundled `.vkfilter` templates, your own `.vkfilter` files, or code typed in), optionally edit the generated script by hand, set up the encode (H.264, H.265, AV1 or ProRes on CPU, NVENC, AMF or QSV, or your own ffmpeg arguments, saved as named presets), and it runs `vspipe` into `ffmpeg` with progress and a live preview of the output. On first start the Setup window installs a private Python with VapourSynth R79, the vs-mlrt plugin for the chosen backend and everything the shipped filters need (vsjetpack and the other PyPI plugins, PyTorch, the Hybrid and bundled scripts, the RIFE/DPIR models, and on Windows upstream's plugins.7z), as upstream does (embedded Python 3.13 beside the exe on Windows, a venv under `~/.local/share/vapourkit` elsewhere; headless: `cargo run -p vk-core --example setup -- ncnn`; `--example check_filters -- VIDEO` then runs every shipped filter once). ffmpeg/ffprobe must be on `PATH`. Files can be dropped onto the window or passed on the command line (`vapourkit video.mkv flow.vkworkflow`).

```sh
cargo run --release -p vk-gui            # the app
cargo run -p vk-core --example upscale -- in.mkv model.onnx out.mkv   # headless
```

## Layout

- `crates/vk-core`: filter chain and .vkfilter (`chain.rs`), script generation (`script.rs`), encoding (`encoding.rs`), .vkworkflow (`workflow.rs`) and the vspipe → ffmpeg job (`job.rs`)
- `crates/vk-gui`: the egui app
- `assets/`: `vapoursynth_template.vpy`, `vk_runtime.py` and `filters/*.vkfilter`, taken unchanged from upstream

## Roadmap

1. ~~Skeleton: video + model → vspipe | ffmpeg with progress~~
2. ~~Encoding settings and saved presets~~ (MIGraphX still to do)
3. ~~First-run setup (Python, VapourSynth, plugins)~~ done; bundled models and ffmpeg still to come
4. ~~Filter chains, `.vkfilter` / `.vkworkflow`, editable script~~ (stage references and Load Video side chains still to do)
5. ~~Live preview~~, side-by-side compare
6. Queue/batch, TensorRT engine builds, update check
7. Colour grading, LUTs, crop, DLSS, Discord presence, installers

## License

GPL-3.0, as upstream.
