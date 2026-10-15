from pathlib import Path
import sys

print("Python:", sys.version.split()[0])
try:
    import llama_cpp
    print("llama_cpp: OK", getattr(llama_cpp, "__version__", ""))
except Exception as exc:
    print("llama_cpp: MISSING / FAILED")
    print(exc)
    raise SystemExit(1)

if len(sys.argv) > 1:
    p = Path(sys.argv[1]).expanduser()
    print("GGUF:", p)
    print("Exists:", p.exists())
    if p.suffix.lower() != ".gguf":
        print("Warning: file does not end in .gguf")
print("Local Qwen runtime check complete.")
