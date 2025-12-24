import shutil
import subprocess
from pathlib import Path
from typing import List, Dict
from urllib.parse import urlparse

# ---------------- CONFIG ----------------
BASE_REPO_DIR = Path("backend/data/repos")

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".java",
    ".md", ".txt", ".html", ".css"
}

IGNORE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    "dist",
    "build"
}

def extract_repo_name(repo_url: str) -> str:
    # Convert GitHub URL to safe folder name (https://github.com/user/repo -> user_repo)
    parsed = urlparse(repo_url)
    path = parsed.path.strip("/").replace(".git", "") # strip() removes start/end slashes
    return path.replace("/", "_")

def clone_repository(repo_url: str) -> Path:
    # Clone GitHub repository locally and return local path
    repo_name = extract_repo_name(repo_url)
    local_repo_path = BASE_REPO_DIR / repo_name

    # Remove existing repo (re-ingestion)
    if local_repo_path.exists():
        shutil.rmtree(local_repo_path)

    BASE_REPO_DIR.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(local_repo_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to clone GitHub repository")

    return local_repo_path

def _should_ignore(path: Path) -> bool:
    # Check if path should be ignored based on directory names
    return any(part.lower() in IGNORE_DIRS for part in path.parts)


def load_repository(repo_path: Path) -> List[Dict]:
    # Recursively load valid source files from a GitHub repository
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    documents: List[Dict] = []

    for file_path in repo_path.rglob("*"):
        if file_path.is_dir():
            continue

        if _should_ignore(file_path):
            continue

        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            if content.strip():
                documents.append({
                    "file_path": str(file_path.relative_to(repo_path)),
                    "content": content
                })

        except Exception as e:
            print(f"⚠️ Failed to read {file_path}: {e}")

    print(f"✅ Loaded {len(documents)} files from repository")

    return documents


def ingest_repository(repo_url: str) -> List[Dict]:
    # Main function to ingest a GitHub repository
    repo_path = clone_repository(repo_url)
    return load_repository(repo_path)
