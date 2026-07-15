#!/usr/bin/env bash
# Simple git helper for people without the VS Code git panel.
# Usage: ./scripts/git-helper.sh <command> [args]
set -euo pipefail

cmd="${1:-help}"
shift || true

case "$cmd" in

  status)
    echo "== branch =="
    git status -sb
    echo
    echo "== changed files =="
    git status --porcelain=v1 --untracked-files=all
    ;;

  diff)
    if [ $# -eq 0 ]; then
      git diff
    else
      git diff -- "$@"
    fi
    ;;

  add)
    if [ $# -eq 0 ]; then
      echo "Usage: $0 add <file> [file...]   (use 'add-all' to stage everything)"
      exit 1
    fi
    git add -- "$@"
    git status -s
    ;;

  add-all)
    echo "Staging ALL changes (tracked + untracked). Review below before committing:"
    git add -A
    git status -s
    ;;

  commit)
    if [ $# -eq 0 ]; then
      echo "Usage: $0 commit \"message\""
      exit 1
    fi
    git commit -m "$1"
    ;;

  push)
    branch="$(git branch --show-current)"
    echo "Pushing '$branch' to origin..."
    git push origin "$branch"
    ;;

  pull)
    branch="$(git branch --show-current)"
    git pull origin "$branch"
    ;;

  fetch)
    git fetch origin
    git status -sb
    ;;

  save)
    # All-in-one: stage everything, commit, push. Good for a quick end-of-task save.
    if [ $# -eq 0 ]; then
      echo "Usage: $0 save \"commit message\""
      exit 1
    fi
    git add -A
    git status -s
    git commit -m "$1"
    branch="$(git branch --show-current)"
    git push origin "$branch"
    ;;

  branch)
    if [ $# -eq 0 ]; then
      git branch -vv
    else
      git switch -c "$1" 2>/dev/null || git switch "$1"
    fi
    ;;

  log)
    n="${1:-10}"
    git log --oneline --graph --decorate -n "$n"
    ;;

  help|*)
    cat <<EOF
git-helper.sh — plain-terminal git shortcuts

  status              show branch + changed files
  diff [file]          show unstaged diff (optionally for one file)
  add <file...>        stage specific files
  add-all              stage everything (tracked + untracked)
  commit "msg"          commit currently staged changes
  push                 push current branch to origin
  pull                 pull current branch from origin
  fetch                fetch + show branch status vs origin
  save "msg"            add-all + commit + push in one step
  branch [name]        list branches, or create/switch to <name>
  log [n]              short graph log (default 10 entries)
EOF
    ;;
esac
