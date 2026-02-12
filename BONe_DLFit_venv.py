import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VENV_DIR = SCRIPT_DIR / "BONe"
PYTHON_EXE = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

REQUIREMENTS = {
    "1": "requirements/cuda129.txt",
    "2": "requirements/cuda126.txt",
    "3": "requirements/cpu.txt"
}


def run_cmd(cmd: list[str], check=True) -> int:
    """Run a subprocess and return the exit code."""
    try:
        subprocess.run(cmd, check=check)
        return 0
    except subprocess.CalledProcessError as e:
        return e.returncode


def find_python312() -> Path:
    """
    Try to locate a valid Python 3.12 interpreter on the system.
    Returns its path if found, or exits if not.
    """
    candidates = ["python3.12", "py -3.12", "C:\\Python312\\python.exe"]

    for cmd in candidates:
        try:
            result = subprocess.run(
                cmd.split() + ["--version"],
                capture_output=True,
                text=True,
                check=True
            )
            if "3.12" in result.stdout or "3.12" in result.stderr:
                print(f"[INFO] Found Python 3.12: {cmd}")
                return cmd.split()
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    print("[ERROR] Python 3.12 not found on this system.")
    print("Please install it from https://www.python.org/downloads/release/python-3120/")
    sys.exit(1)


def create_venv():
    if not PYTHON_EXE.exists():
        python_cmd = find_python312()
        print("[INFO] Creating BONe virtual environment using Python 3.12...")
        result = run_cmd(python_cmd + ["-m", "venv", str(VENV_DIR)])
        if result != 0:
            print("[ERROR] Failed to create virtual environment.")
            sys.exit(1)
    else:
        print("[INFO] BONe virtual environment already exists.")

    # Ensure pip is upgraded
    print("[INFO] Upgrading pip in the virtual environment...")
    run_cmd([str(PYTHON_EXE), "-m", "ensurepip", "--upgrade"])
    run_cmd([str(PYTHON_EXE), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

def verify_venv_python():
    print("[INFO] Verifying virtual environment Python interpreter...")
    run_cmd([str(PYTHON_EXE), "--version"])


def is_torch_installed() -> bool:
    result = subprocess.run(
        [str(PYTHON_EXE), "-c", "import torch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def prompt_cuda_choice() -> str:
    print("\nSelect a PyTorch version to install:")
    print("1. CUDA 12.9 (Maxwell to Blackwell GPUs)")
    print("2. CUDA 12.6 (Kepler to Hopper GPUs)")
    print("3. CPU only")
    while True:
        choice = input("Enter your choice (1/2/3): ").strip()
        if choice in REQUIREMENTS:
            return REQUIREMENTS[choice]
        print("[ERROR] Invalid choice. Please try again.")


def install_requirements(req_file: Path):
    if not req_file.exists():
        print(f"[ERROR] Requirements file not found: {req_file}")
        sys.exit(1)

    print(f"[INFO] Installing packages from {req_file.name}...")
    run_cmd([str(PYTHON_EXE), "-m", "pip", "install", "-r", str(req_file)])


def run_main_script():
    script_path = SCRIPT_DIR / "BONe_DLFit/main.py"
    if not script_path.exists():
        print("[ERROR] BONe_DLFit/main.py not found!")
        sys.exit(1)

    print("\n[INFO] Running BONe_DLFit...\n")
    run_cmd([str(PYTHON_EXE), "-m", "BONe_DLFit.main"])


def main():
    os.chdir(SCRIPT_DIR)

    create_venv()
    verify_venv_python()

    if not is_torch_installed():
        print("\n[INFO] PyTorch not detected.")
        req_file = SCRIPT_DIR / prompt_cuda_choice()
        install_requirements(req_file)
    else:
        print("[INFO] PyTorch 2.8.0 already installed. Skipping installation.")

    run_main_script()

    print("\n[INFO] Script finished.")


if __name__ == "__main__":
    main()