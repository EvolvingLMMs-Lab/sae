from pathlib import Path

from utils import run_unittest_files

current_dir = Path(__file__).parent.resolve()

if __name__ == "__main__":
    files = [p for p in current_dir.glob("**/test_*.py")]

    exit_code = run_unittest_files(files)
    exit(exit_code)
