# vapourkit-rs

A native Rust port of [Vapourkit](https://github.com/Kim2091/vapourkit), the VapourSynth video upscaling app, without Electron. GUI is [egui](https://github.com/emilk/egui).

Status: skeleton. You pick a video, an ONNX/engine model, a backend (TensorRT, DirectML or NCNN) and an output, and it runs `vspipe` into `ffmpeg` with a progress bar. VapourSynth, vs-mlrt, BestSource and ffmpeg must already be installed and on `PATH`; first-run setup comes later.

```sh
cargo run --release -p vk-gui            # the app
cargo run -p vk-core --example upscale -- in.mkv model.onnx out.mkv   # headless
```

## Layout

- `crates/vk-core`: script generation (`script.rs`) and the vspipe → ffmpeg job (`job.rs`)
- `crates/vk-gui`: the egui app
- `assets/`: `vapoursynth_template.vpy` and `vk_runtime.py`, taken unchanged from upstream

## Roadmap

1. ~~Skeleton: video + model → vspipe | ffmpeg with progress~~
2. Settings, encoder presets, backend choice incl. MIGraphX
3. First-run setup (Python venv, VapourSynth, plugins, models)
4. Filter chains, `.vkfilter` / `.vkworkflow` templates
5. Live preview and side-by-side compare
6. Queue/batch, TensorRT engine builds, update check
7. Colour grading, LUTs, crop, DLSS, Discord presence, installers

## License

GPL-3.0, as upstream.
