# 🚀 CHEATSHEET – Git & uv Commands

**Project: Rakuten Data Science**

This cheat sheet contains the most important commands for working with:
- Git
- Python dependencies using `uv`

---

# 📁 Project Setup

Go to the project folder:


cd <path-to-project>


---

# 🔧 Git – Daily Workflow

## Check status


git status


## Pull latest changes


git pull


If your push was rejected:


git pull --rebase origin main


## Add changes

Single file:


git add scripts/setup_data.ps1


Multiple files:


git add README.md docs/WORKFLOW.md scripts/setup_data.ps1


All files:


git add .


## Commit


git commit -m "Describe your change"


## Push


git push


## Check differences


git diff
git diff --cached


## Show commit history


git log --oneline


---

# 🌿 Git – Branching

## Create new branch


git checkout -b feature/my-feature


## Switch branch


git checkout main


## Push branch


git push -u origin feature/my-feature


---

# ⚠️ Git – Common Issues

## Push rejected


git pull --rebase origin main
git push


## Unstage file


git restore --staged <file>


## Discard changes


git restore <file>


---

# 🐍 uv – Environment & Dependencies

## Install environment


uv sync


## Add new library


uv add <package>


Example:


uv add matplotlib


## Add multiple libraries


uv add numpy pandas seaborn


## Run Python in project environment


uv run python script.py


## Check installed packages


uv pip list


## Check Python version


python --version
uv run python --version


---

# 📏 Dependency Rules

## Correct way


uv add <package>


## Avoid


!pip install <package>


Reason:
- not reproducible
- not synced with team
- not stored in `pyproject.toml`

## Team workflow

1. Add dependency:


uv add <package>


2. Commit:
- `pyproject.toml`
- `uv.lock`

3. Push to GitHub

4. Others update with:


uv sync


---

# 📊 Data Setup

## Windows


.\scripts\setup_data.ps1


## macOS


./scripts/setup_data.sh


---

# 🔁 Daily Workflow

## Start working


git pull
uv sync


## After changes


git status
git add .
git commit -m "Describe your change"
git push


## If push fails


git pull --rebase origin main
git push


---

# 📦 What Goes Into Git

## Include

- `README.md`
- `docs/`
- `scripts/`
- `src/`
- `notebooks/`
- `pyproject.toml`
- `uv.lock`

## Exclude

- `.venv/`
- `data/`
- `private/`
- `kaggle.json`

---

# ⚡ Quick Reference


git status
git pull
git add .
git commit -m "message"
git push

uv sync
uv add seaborn

.\scripts\setup_data.ps1


---

# 🧠 Summary

## Git
- `git status` → current state
- `git add` → stage changes
- `git commit` → save locally
- `git push` → upload changes
- `git pull --rebase` → integrate remote changes cleanly

## uv
- `uv sync` → install environment
- `uv add` → add dependency
- `uv run` → execute inside project environment