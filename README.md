Standalone versions of BONe DLFit, BONe DLPred, and TorchCUDA IoU (Amira-Avizo is not needed)

Citation: Lee et al. in prep.

To fit a 2D, 2.5D, and 3D models, use:   BONe_DLFit_venv.py
To perform AI segmentation, use:         BONe_DLPred_venv.py
To calculate IoU Score, use:             TorchCUDA_IoU_venv.py

Installation and operation: Windows (tested on 10/11 Pro)
=========================================================
1.  Install Python 3.12 (tested 12.0, 12.3, 12.10) for full PyTorch support (as of 10/14/2025).

2.  Copy BONe_apps folder to computer.

3.  Double-click to run:
    BONe_DLFit_venv.py        (train a model) or 
    BONe_DLPred_venv.py       (apply AI model to segment a micro-CT scan) or
    BONe_IoU_venv.py          (evaluate predicted segmentation)

    Note: step 3 works as intended on your computer if py files are associated with Python 3.12. If not,
          right-click the desired py file and Open with Python 3.12. Alternatively, open a Terminal or 
          Powershell window from the BONe_apps folder.
          [if Python 3.12 is the only version on the computer] Type: py BONe_DLFit_venv.py
          [if there are different versions of Python on the computer] Type: python3.12 BONe_DLFit_venv.py

4.  Each app will check for Python 3.12, pip, and PyTorch (2.8 as of 10/14/2025) before running. Upon initial installation, user will be asked to choose a CUDA version of PyTorch (12.9 for support up to 50-series/Blackwell Nvidia GPUs; 12.6 for support up to 40-series/Ada Lovelace/Hopper Nvidia GPUs; or CPU for systems without an Nvidia GPU).

5.  Close app window to exit.


Installation and operation: Ubuntu Linux (tested on 22.04 LTS)
==============================================================
1.  * Install Python 3.12 (tested 12.0, 12.3, 12.10) for full PyTorch support (as of 7/12/2025)

2.  Copy BONe_apps folder to computer (drive must be in ext4 format).

3.  Open Terminal at location of BONe_apps folder, and type:
    python3.12 BONe_DLFit_venv.py        (training a model)  or
    python3.12 BONe_DLPred_venv.py       (apply AI model to segment a micro-CT scan) or
    python3.12 BONe_IoU_venv.py          (evaluate predicted segmentation)

4.  Each app will check for Python 3.12, pip, and Pytorch (2.8 as of 10/14/2025) before running. Upon initial installation, user will be asked to choose a CUDA version of PyTorch (12.9 for support up to 50-series/Blackwell Nvidia GPUs; 12.6 for support up to 40-series/Ada Lovelace/Hopper Nvidia GPUs; or CPU for systems without an Nvidia GPU).

5.  Close app window to exit.

* (Recommended method of installing Python using Ubuntu 22.04 LTS)
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.12{,-venv,-tk, -dev, -distutils}
