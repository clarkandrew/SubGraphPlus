import subprocess
import os
from collections import defaultdict

def get_git_tracked_files():
    """Returns a list of all files tracked by git."""
    try:
        output = subprocess.check_output(['git', 'ls-files'], text=True)
        return output.strip().split('\n')
    except subprocess.CalledProcessError:
        print("⚠️ Not a git repository or git not installed.")
        return []

def build_tree(files):
    """Builds a nested dictionary representing the file tree."""
    tree = lambda: defaultdict(tree)
    root = tree()
    for file_path in files:
        parts = file_path.split(os.sep)
        current = root
        for part in parts:
            current = current[part]
    return root

def print_tree(d, prefix=''):
    """Recursively prints the tree structure."""
    entries = sorted(d.keys())
    for idx, entry in enumerate(entries):
        connector = "└── " if idx == len(entries) - 1 else "├── "
        print(f"{prefix}{connector}{entry}")
        if d[entry]:
            extension = "    " if idx == len(entries) - 1 else "│   "
            print_tree(d[entry], prefix + extension)

if __name__ == "__main__":
    files = get_git_tracked_files()
    if files:
        tree = build_tree(files)
        print(".")
        print_tree(tree)
