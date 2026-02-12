# BONe_DLFit/main.py
from pathlib import Path
import sys
from tkinter import Tk


# Allow running `python BONe_DLFit/main.py` directly by adding the repo root to sys.path.
if __package__ in (None, ''):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


from BONe_DLFit.app.gui import BONe_DLFit

def main():
    root = Tk()
    root.title('BONe Deep Learning Fitting')
    root.geometry('')
    root.minsize(600, 820)
    app = BONe_DLFit(root)
    root.mainloop()

if __name__ == '__main__':
    main()
