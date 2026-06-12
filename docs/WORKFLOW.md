# Project Workflow

This document describes how to set up the project locally and defines the team workflow.

--------------------------------------------------

## 1. Repository purpose

GitHub is used for:

- code
- configuration
- documentation
- project structure

GitHub is **not** used for:

- raw data
- large image files
- local environments
- secrets or credentials

--------------------------------------------------

## 2. Clone the repository

git clone <repository-url>  
cd Rakuten_Data_Science

--------------------------------------------------

## 3. Install the environment

We use:

- Python 3.12
- uv for dependency management

Run:

uv sync

This installs all dependencies defined in `pyproject.toml`.

--------------------------------------------------

## 4. Configure Kaggle

To download the dataset you need the Kaggle API.

Place your `kaggle.json` file in:

C:\Users\<username>\.kaggle\

Example:

C:\Users\felix\.kaggle\kaggle.json

--------------------------------------------------

## 5. Download and prepare the dataset

Use the setup script that matches your operating system:

Windows:
.\scripts\setup_data.ps1

macOS:
./scripts/setup_data.sh

The script will:

- create the required folder structure
- download the dataset (if enabled)
- extract the files
- organize everything in `data/raw/`

--------------------------------------------------

## 6. Data structure

data/  
└── raw/  
&nbsp;&nbsp;&nbsp;&nbsp;├── images/  
&nbsp;&nbsp;&nbsp;&nbsp;├── x_train.csv  
&nbsp;&nbsp;&nbsp;&nbsp;├── y_train.csv  
&nbsp;&nbsp;&nbsp;&nbsp;└── x_test.csv  

processed/ and splits/ are used for derived data.

### Important rule

Files inside `data/raw` must **never be modified**.

--------------------------------------------------

## 7. Folder responsibilities

- `notebooks/` → exploration and experiments  
- `src/` → reusable project code  
- `scripts/` → setup and helper scripts  
- `tests/` → validation and testing  
- `docs/` → documentation  

--------------------------------------------------

## 8. Git rules

- Do not commit raw data  
- Do not commit image datasets  
- Do not commit `.venv`  
- Do not commit Kaggle credentials  
- Keep notebooks for exploration only  
- Move reusable logic into `src/`  

--------------------------------------------------

## 9. Recommended team workflow

1. Pull the latest repository version  
2. Run `uv sync`  
3. Set up local data  
4. Work in a separate branch if needed  
5. Commit only code and documentation  
6. Push changes to GitHub  

--------------------------------------------------

## 10. Notes

- The repository should stay lightweight  
- Data and code are intentionally separated  
- The structure should remain consistent across the team  