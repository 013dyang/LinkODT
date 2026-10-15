"""LinkODT application entry point. Legacy module names preserve compatibility."""
import sys

if '--visualization' in sys.argv:
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'visualization'))
    from visualization_app import main
else:
    from LinkNex import main

if __name__ == '__main__':
    main()
