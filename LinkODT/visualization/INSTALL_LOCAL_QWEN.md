# Local Qwen for LinkNex Visualization Studio

AI Director runs fully locally from a GGUF file. No API key and no network request are required after dependencies and the model are present.

## Models

The model picker accepts any compatible `.gguf`, including for example:

- `Qwen3-4B-Q4_K_M.gguf`
- `Qwen3-8B-Q4_K_M.gguf`
- `Qwen3-14B-Q4_K_M.gguf`

The model file can live anywhere. LinkNex remembers the last selected path.

## Runtime

LinkNex uses `llama-cpp-python` to load the GGUF directly.

Install the normal Visualization dependencies first:

```bash
pip install -r requirements_visualization.txt
```

`llama-cpp-python` can run on CPU. For NVIDIA GPU acceleration, install/build a CUDA-enabled llama-cpp-python appropriate for the CUDA toolchain on that workstation. The AI Director has a `GPU layers` setting; `-1` asks llama.cpp to offload all possible layers.

## Recommended LinkNex settings

For the camera-action translation task, a long context is not necessary. A practical starting point is:

- Context: 8192
- GPU layers: -1 (all possible) when using a compatible NVIDIA build
- CPU threads: 0 (auto)
- Max output tokens: 2048
- Temperature: 0.2

The first translation loads the model and is slower. Later translations reuse the loaded model.

## Privacy

The user's prompt, ROI coordinates, camera state and object names stay on the local machine. They are not sent to a cloud API.
