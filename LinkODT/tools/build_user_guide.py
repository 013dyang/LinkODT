"""Build the bilingual user guide from any working directory."""

import argparse
from pathlib import Path
import subprocess
import sys
import webbrowser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-dependencies", action="store_true",
                        help="Install the guide's Python dependencies before building.")
    parser.add_argument("--open", action="store_true", dest="open_browser",
                        help="Open the English guide after a successful build.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    candidates = [path for path in root.iterdir() if path.is_dir()
                  and (path / "source" / "conf.py").is_file()
                  and (path / "source_en" / "conf.py").is_file()
                  and (path / "requirements-docs.txt").is_file()]
    if len(candidates) != 1:
        parser.exit(1, "Could not identify a unique user guide directory. "
                    "Keep the guide sources beside the LinkODT application folder.\n")
    guide = candidates[0]
    if args.install_dependencies:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                        str(guide / "requirements-docs.txt")], check=True)
    for source, output in (("source", "html"), ("source_en", "en/html")):
        subprocess.run([sys.executable, "-m", "sphinx", "-b", "html", "-W",
                        "--keep-going", str(guide / source), str(guide / output)],
                       check=True)
    print("English and Chinese user guides built successfully.")
    if args.open_browser:
        webbrowser.open((guide / "en" / "html" / "index.html").as_uri())


if __name__ == "__main__":
    main()
