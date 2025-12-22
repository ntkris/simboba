"""JSON file storage for simboba.

All data is stored as JSON files in the boba-evals/ directory:
- datasets/{name}.json - Dataset metadata and cases (contains stable UUID)
- baselines/{dataset_id}.json - Committed run results (git tracked, by UUID)
- runs/{dataset_id}/{timestamp}.json - All runs (gitignored, by UUID)
- files/ - Uploaded attachments
- settings.json - App settings

Datasets are identified by both name (human-readable, file name) and id (UUID, stable).
Runs and baselines use the dataset UUID so they survive renames.
"""

import json
import secrets
import string
from datetime import datetime
from pathlib import Path
from typing import Optional

from simboba.config import find_boba_evals_dir


def get_evals_dir() -> Path:
    """Get the boba-evals directory, creating if needed."""
    evals_dir = find_boba_evals_dir()
    if evals_dir:
        return evals_dir
    # Fallback to current directory
    return Path.cwd() / "boba-evals"


def ensure_dirs(evals_dir: Path) -> None:
    """Ensure all required subdirectories exist."""
    (evals_dir / "datasets").mkdir(parents=True, exist_ok=True)
    (evals_dir / "baselines").mkdir(parents=True, exist_ok=True)
    (evals_dir / "runs").mkdir(parents=True, exist_ok=True)
    (evals_dir / "files").mkdir(parents=True, exist_ok=True)


def generate_id(length: int = 8) -> str:
    """Generate a random alphanumeric ID."""
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def generate_timestamp() -> str:
    """Generate a timestamp string for run filenames."""
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def atomic_write(path: Path, data: dict) -> None:
    """Write JSON data atomically using temp file + rename."""
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(data, indent=2, default=str))
    temp_path.rename(path)


def safe_read(path: Path) -> Optional[dict]:
    """Read JSON file, returning None if not found."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return None


# --- Dataset Operations ---

def list_datasets(evals_dir: Optional[Path] = None) -> list[dict]:
    """List all datasets."""
    evals_dir = evals_dir or get_evals_dir()
    datasets_dir = evals_dir / "datasets"
    if not datasets_dir.exists():
        return []

    datasets = []
    for path in sorted(datasets_dir.glob("*.json")):
        data = safe_read(path)
        if data:
            # Add computed fields
            data["case_count"] = len(data.get("cases", []))
            datasets.append(data)

    # Sort by updated_at descending
    datasets.sort(key=lambda d: d.get("updated_at", ""), reverse=True)
    return datasets


def get_dataset(name: str, evals_dir: Optional[Path] = None) -> Optional[dict]:
    """Get a dataset by name."""
    evals_dir = evals_dir or get_evals_dir()
    path = evals_dir / "datasets" / f"{name}.json"
    data = safe_read(path)
    if data:
        data["case_count"] = len(data.get("cases", []))
    return data


def save_dataset(dataset: dict, evals_dir: Optional[Path] = None) -> dict:
    """Save a dataset. Creates new if no 'name' exists, updates if it does."""
    evals_dir = evals_dir or get_evals_dir()
    ensure_dirs(evals_dir)

    name = dataset["name"]
    path = evals_dir / "datasets" / f"{name}.json"

    now = datetime.now().isoformat()

    # Ensure dataset has a stable ID (UUID)
    if "id" not in dataset:
        dataset["id"] = generate_id()

    # Ensure required fields
    if "created_at" not in dataset:
        dataset["created_at"] = now
    dataset["updated_at"] = now

    # Ensure all cases have IDs
    for case in dataset.get("cases", []):
        if "id" not in case:
            case["id"] = generate_id()
        if "created_at" not in case:
            case["created_at"] = now
        case["updated_at"] = now

    atomic_write(path, dataset)
    dataset["case_count"] = len(dataset.get("cases", []))
    return dataset


def get_dataset_by_id(dataset_id: str, evals_dir: Optional[Path] = None) -> Optional[dict]:
    """Get a dataset by its UUID."""
    evals_dir = evals_dir or get_evals_dir()
    datasets_dir = evals_dir / "datasets"
    if not datasets_dir.exists():
        return None

    for path in datasets_dir.glob("*.json"):
        data = safe_read(path)
        if data and data.get("id") == dataset_id:
            data["case_count"] = len(data.get("cases", []))
            return data
    return None


def rename_dataset(old_name: str, new_name: str, evals_dir: Optional[Path] = None) -> Optional[dict]:
    """Rename a dataset. Updates the file name but keeps the same UUID.

    Returns the updated dataset, or None if old dataset not found.
    Raises ValueError if new_name already exists.
    """
    evals_dir = evals_dir or get_evals_dir()

    # Check old exists
    old_path = evals_dir / "datasets" / f"{old_name}.json"
    if not old_path.exists():
        return None

    # Check new doesn't exist (unless same name)
    if old_name != new_name:
        new_path = evals_dir / "datasets" / f"{new_name}.json"
        if new_path.exists():
            raise ValueError(f"Dataset '{new_name}' already exists")

    # Load, update name, save to new path
    dataset = safe_read(old_path)
    if not dataset:
        return None

    dataset["name"] = new_name
    dataset["updated_at"] = datetime.now().isoformat()

    if old_name != new_name:
        # Write to new path, delete old
        new_path = evals_dir / "datasets" / f"{new_name}.json"
        atomic_write(new_path, dataset)
        old_path.unlink()
    else:
        # Just update in place
        atomic_write(old_path, dataset)

    dataset["case_count"] = len(dataset.get("cases", []))
    return dataset


def delete_dataset(name: str, evals_dir: Optional[Path] = None) -> bool:
    """Delete a dataset. Returns True if deleted, False if not found."""
    evals_dir = evals_dir or get_evals_dir()
    path = evals_dir / "datasets" / f"{name}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def dataset_exists(name: str, evals_dir: Optional[Path] = None) -> bool:
    """Check if a dataset with the given name exists."""
    evals_dir = evals_dir or get_evals_dir()
    path = evals_dir / "datasets" / f"{name}.json"
    return path.exists()


# --- Case Operations ---

def get_case(dataset_name: str, case_id: str, evals_dir: Optional[Path] = None) -> Optional[dict]:
    """Get a case by ID from a dataset."""
    dataset = get_dataset(dataset_name, evals_dir)
    if not dataset:
        return None
    for case in dataset.get("cases", []):
        if case.get("id") == case_id:
            case["dataset_name"] = dataset_name
            return case
    return None


def add_case(dataset_name: str, case: dict, evals_dir: Optional[Path] = None) -> dict:
    """Add a case to a dataset."""
    evals_dir = evals_dir or get_evals_dir()
    dataset = get_dataset(dataset_name, evals_dir)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_name}' not found")

    # Ensure case has ID
    if "id" not in case:
        case["id"] = generate_id()

    now = datetime.now().isoformat()
    case["created_at"] = now
    case["updated_at"] = now

    dataset.setdefault("cases", []).append(case)
    save_dataset(dataset, evals_dir)

    case["dataset_name"] = dataset_name
    return case


def update_case(dataset_name: str, case_id: str, updates: dict, evals_dir: Optional[Path] = None) -> Optional[dict]:
    """Update a case in a dataset."""
    evals_dir = evals_dir or get_evals_dir()
    dataset = get_dataset(dataset_name, evals_dir)
    if not dataset:
        return None

    for case in dataset.get("cases", []):
        if case.get("id") == case_id:
            case.update(updates)
            case["updated_at"] = datetime.now().isoformat()
            save_dataset(dataset, evals_dir)
            case["dataset_name"] = dataset_name
            return case
    return None


def delete_case(dataset_name: str, case_id: str, evals_dir: Optional[Path] = None) -> bool:
    """Delete a case from a dataset."""
    evals_dir = evals_dir or get_evals_dir()
    dataset = get_dataset(dataset_name, evals_dir)
    if not dataset:
        return False

    original_count = len(dataset.get("cases", []))
    dataset["cases"] = [c for c in dataset.get("cases", []) if c.get("id") != case_id]

    if len(dataset["cases"]) < original_count:
        save_dataset(dataset, evals_dir)
        return True
    return False


# --- Run Operations ---
# Runs are stored by dataset ID (UUID), not name, so they survive renames.

def get_run_dir(dataset_id: str, evals_dir: Optional[Path] = None) -> Path:
    """Get the runs directory for a dataset (by ID)."""
    evals_dir = evals_dir or get_evals_dir()
    return evals_dir / "runs" / dataset_id


def list_runs(dataset_id: Optional[str] = None, evals_dir: Optional[Path] = None) -> list[dict]:
    """List all runs, optionally filtered by dataset ID.

    Args:
        dataset_id: Filter by dataset UUID. If None, returns all runs.
    """
    evals_dir = evals_dir or get_evals_dir()
    runs_dir = evals_dir / "runs"
    if not runs_dir.exists():
        return []

    runs = []

    if dataset_id:
        # List runs for specific dataset
        dataset_runs_dir = runs_dir / dataset_id
        if dataset_runs_dir.exists():
            for path in dataset_runs_dir.glob("*.json"):
                data = safe_read(path)
                if data:
                    data["dataset_id"] = dataset_id
                    data["filename"] = path.stem
                    runs.append(data)
    else:
        # List all runs
        for dataset_dir in runs_dir.iterdir():
            if dataset_dir.is_dir():
                for path in dataset_dir.glob("*.json"):
                    data = safe_read(path)
                    if data:
                        data["dataset_id"] = dataset_dir.name
                        data["filename"] = path.stem
                        runs.append(data)

    # Sort by started_at descending
    runs.sort(key=lambda r: r.get("started_at", ""), reverse=True)
    return runs


def get_run(dataset_id: str, filename: str, evals_dir: Optional[Path] = None) -> Optional[dict]:
    """Get a specific run by dataset ID and filename."""
    evals_dir = evals_dir or get_evals_dir()
    path = evals_dir / "runs" / dataset_id / f"{filename}.json"
    data = safe_read(path)
    if data:
        data["dataset_id"] = dataset_id
        data["filename"] = filename
    return data


def save_run(dataset_id: str, run: dict, evals_dir: Optional[Path] = None) -> dict:
    """Save a run. Creates timestamp-based filename if not provided.

    Args:
        dataset_id: The dataset UUID (not name)
        run: Run data dict
    """
    evals_dir = evals_dir or get_evals_dir()
    ensure_dirs(evals_dir)

    dataset_runs_dir = evals_dir / "runs" / dataset_id
    dataset_runs_dir.mkdir(parents=True, exist_ok=True)

    # Use existing filename or generate new one
    filename = run.get("filename") or generate_timestamp()
    path = dataset_runs_dir / f"{filename}.json"

    # Ensure started_at
    if "started_at" not in run:
        run["started_at"] = datetime.now().isoformat()

    # Don't store filename in the JSON itself, but do store dataset_id
    run_data = {k: v for k, v in run.items() if k not in ("filename",)}
    run_data["dataset_id"] = dataset_id

    atomic_write(path, run_data)

    run["filename"] = filename
    run["dataset_id"] = dataset_id
    return run


def delete_run(dataset_id: str, filename: str, evals_dir: Optional[Path] = None) -> bool:
    """Delete a run by dataset ID and filename."""
    evals_dir = evals_dir or get_evals_dir()
    path = evals_dir / "runs" / dataset_id / f"{filename}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def clear_runs(evals_dir: Optional[Path] = None) -> int:
    """Delete all runs. Returns count of deleted runs."""
    evals_dir = evals_dir or get_evals_dir()
    runs_dir = evals_dir / "runs"
    if not runs_dir.exists():
        return 0

    count = 0
    for dataset_dir in runs_dir.iterdir():
        if dataset_dir.is_dir():
            for path in dataset_dir.glob("*.json"):
                path.unlink()
                count += 1
            # Remove empty dataset directory
            if not any(dataset_dir.iterdir()):
                dataset_dir.rmdir()

    return count


# --- Baseline Operations ---
# Baselines are stored by dataset ID (UUID), not name, so they survive renames.

def get_baseline(dataset_id: str, evals_dir: Optional[Path] = None) -> Optional[dict]:
    """Get the baseline for a dataset by ID."""
    evals_dir = evals_dir or get_evals_dir()
    path = evals_dir / "baselines" / f"{dataset_id}.json"
    data = safe_read(path)
    if data:
        data["dataset_id"] = dataset_id
    return data


def save_baseline(dataset_id: str, baseline: dict, evals_dir: Optional[Path] = None) -> dict:
    """Save a baseline for a dataset.

    Args:
        dataset_id: The dataset UUID (not name)
        baseline: Baseline data dict
    """
    evals_dir = evals_dir or get_evals_dir()
    ensure_dirs(evals_dir)

    path = evals_dir / "baselines" / f"{dataset_id}.json"

    # Add metadata
    baseline["dataset_id"] = dataset_id
    baseline["saved_at"] = datetime.now().isoformat()

    atomic_write(path, baseline)
    return baseline


def list_baselines(evals_dir: Optional[Path] = None) -> list[dict]:
    """List all baselines."""
    evals_dir = evals_dir or get_evals_dir()
    baselines_dir = evals_dir / "baselines"
    if not baselines_dir.exists():
        return []

    baselines = []
    for path in baselines_dir.glob("*.json"):
        data = safe_read(path)
        if data:
            data["dataset_id"] = path.stem
            baselines.append(data)

    return baselines


# --- Settings Operations ---

def get_settings(evals_dir: Optional[Path] = None) -> dict:
    """Get all settings."""
    evals_dir = evals_dir or get_evals_dir()
    path = evals_dir / "settings.json"

    defaults = {
        "model": "anthropic/claude-haiku-4-5-20251001",
    }

    data = safe_read(path) or {}
    return {**defaults, **data}


def save_settings(settings: dict, evals_dir: Optional[Path] = None) -> dict:
    """Save settings."""
    evals_dir = evals_dir or get_evals_dir()
    ensure_dirs(evals_dir)

    path = evals_dir / "settings.json"
    atomic_write(path, settings)
    return settings


def get_setting(key: str, evals_dir: Optional[Path] = None) -> str:
    """Get a single setting value."""
    settings = get_settings(evals_dir)
    return settings.get(key, "")


def set_setting(key: str, value: str, evals_dir: Optional[Path] = None) -> None:
    """Set a single setting value."""
    settings = get_settings(evals_dir)
    settings[key] = value
    save_settings(settings, evals_dir)


# --- File Operations ---

def get_files_dir(evals_dir: Optional[Path] = None) -> Path:
    """Get the files directory."""
    evals_dir = evals_dir or get_evals_dir()
    return evals_dir / "files"


def save_file(filename: str, content: bytes, evals_dir: Optional[Path] = None) -> str:
    """Save a file and return its relative path."""
    evals_dir = evals_dir or get_evals_dir()
    ensure_dirs(evals_dir)

    files_dir = evals_dir / "files"
    path = files_dir / filename

    # Handle duplicates by adding suffix
    if path.exists():
        stem = path.stem
        suffix = path.suffix
        counter = 1
        while path.exists():
            path = files_dir / f"{stem}-{counter}{suffix}"
            counter += 1

    path.write_bytes(content)
    return path.name


def get_file_path(filename: str, evals_dir: Optional[Path] = None) -> Optional[Path]:
    """Get the full path to a file."""
    evals_dir = evals_dir or get_evals_dir()
    path = evals_dir / "files" / filename
    if path.exists():
        return path
    return None


# --- Regression Detection ---

def compare_run_to_baseline(run: dict, baseline: Optional[dict]) -> dict:
    """Compare a run to its baseline and return regression info.

    Returns dict with:
    - regressions: list of case IDs that were passing, now failing
    - fixes: list of case IDs that were failing, now passing
    - new_cases: list of case IDs not in baseline
    - removed_cases: list of case IDs in baseline but not in run
    """
    if not baseline:
        return {
            "regressions": [],
            "fixes": [],
            "new_cases": list(run.get("results", {}).keys()),
            "removed_cases": [],
            "has_baseline": False,
        }

    run_results = run.get("results", {})
    baseline_results = baseline.get("results", {})

    regressions = []
    fixes = []
    new_cases = []
    removed_cases = []

    # Check each case in current run
    for case_id, result in run_results.items():
        if case_id not in baseline_results:
            new_cases.append(case_id)
        else:
            baseline_passed = baseline_results[case_id].get("passed", False)
            current_passed = result.get("passed", False)

            if baseline_passed and not current_passed:
                regressions.append(case_id)
            elif not baseline_passed and current_passed:
                fixes.append(case_id)

    # Check for removed cases
    for case_id in baseline_results:
        if case_id not in run_results:
            removed_cases.append(case_id)

    return {
        "regressions": regressions,
        "fixes": fixes,
        "new_cases": new_cases,
        "removed_cases": removed_cases,
        "has_baseline": True,
    }
