"""CLI entry point for simboba."""

import logging
import click
import uvicorn
from dotenv import load_dotenv

# Load .env file from current directory
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s - %(message)s"
)


BOBA_ART = r"""
     ( )
   .-~~~-.
  /  o o  \
  |  ===  |
  |  .:.  |
  | ::::: |
  |_:::::_|
    '---'
"""


def _maybe_exec_in_docker():
    """Check if we should exec into Docker, called before Click processing."""
    import sys

    # Don't redirect these commands - they need to run locally or handle Docker themselves
    if len(sys.argv) > 1 and sys.argv[1] in ("init", "setup", "magic", "run"):
        return

    from simboba.config import maybe_exec_in_docker
    maybe_exec_in_docker()


@click.group()
@click.version_option(version="0.1.7")
def main():
    """
    \b
     ( )
   .-~~~-.
  /       \\
  |  ===  |
  | ::::: |
  |_:::::_|
    '---'

  boba - Simple eval tracking

  Run evals and track results with AI-powered judging.
    """
    pass


# Call docker exec wrapper early
_maybe_exec_in_docker()


@main.command()
@click.option("--docker", "use_docker", is_flag=True, help="Quick setup for Docker Compose environments")
@click.option("--local", "use_local", is_flag=True, help="Quick setup for local Python environments")
def init(use_docker: bool, use_local: bool):
    """Initialize boba-evals folder with starter templates."""
    from pathlib import Path
    from simboba.config import BobaConfig, save_config, CONFIG_FILENAME

    evals_dir = Path("boba-evals")
    if evals_dir.exists():
        click.echo("boba-evals/ folder already exists")
        return

    # Determine runtime mode
    if use_docker and use_local:
        click.echo("Cannot specify both --docker and --local", err=True)
        raise SystemExit(1)

    if use_docker:
        runtime = "docker-compose"
        service = click.prompt("Docker Compose service name", default="api")
    elif use_local:
        runtime = "local"
        service = None
    else:
        # Interactive mode
        click.echo("")
        click.echo("Welcome to Boba! Let's set up your eval environment.")
        click.echo("")

        runtime = click.prompt(
            "How do you run your app?",
            type=click.Choice(["local", "docker-compose"], case_sensitive=False),
            default="local",
            show_choices=True,
        )

        if runtime == "docker-compose":
            service = click.prompt("Docker Compose service name", default="api")
        else:
            service = None

    # Create the evals directory and subdirectories
    evals_dir.mkdir()
    (evals_dir / "datasets").mkdir()
    (evals_dir / "baselines").mkdir()
    (evals_dir / "runs").mkdir()
    (evals_dir / "files").mkdir()

    # Save configuration
    config = BobaConfig(runtime=runtime, service=service)
    save_config(config, evals_dir / CONFIG_FILENAME)

    # Copy sample files from package
    import simboba
    import shutil
    package_dir = Path(simboba.__file__).parent
    samples_dir = package_dir / "samples"

    if samples_dir.exists():
        # Copy setup.py
        shutil.copy(samples_dir / "setup.py", evals_dir / "setup.py")
        # Copy test.py
        shutil.copy(samples_dir / "test.py", evals_dir / "test.py")
    else:
        # Fallback if samples not found
        (evals_dir / "setup.py").write_text("# Setup file - see boba docs\ndef get_context():\n    return {}\n\ndef cleanup():\n    pass\n")
        (evals_dir / "test.py").write_text("# Eval file - see boba docs\nfrom simboba import Boba\n\nboba = Boba()\n")

    # Create settings.json with defaults
    import json
    settings = {"model": "anthropic/claude-haiku-4-5-20251001"}
    (evals_dir / "settings.json").write_text(json.dumps(settings, indent=2))

    # Add .gitignore for runs folder
    gitignore_content = """# Boba eval runs (ephemeral, not committed)
runs/
"""
    (evals_dir / ".gitignore").write_text(gitignore_content)

    # Success message
    click.echo("")
    click.echo(click.style("Created boba-evals/ folder", fg="green"))
    click.echo("  - datasets/      (your eval datasets)")
    click.echo("  - baselines/     (committed run results)")
    click.echo("  - runs/          (ephemeral run history)")
    click.echo("  - files/         (uploaded attachments)")
    click.echo("  - setup.py       (shared test fixtures)")
    click.echo("  - test.py        (example eval script)")
    click.echo("")

    if runtime == "docker-compose":
        click.echo("Next steps:")
        click.echo("")
        click.echo("  1. Add simboba to your container's dependencies")
        click.echo("  2. Expose port 8787 in docker-compose.yml")
        click.echo("  3. docker compose build && docker compose up -d")
        click.echo("  4. Run 'boba magic' and paste into your AI coding tool")
        click.echo("  5. boba serve")
    else:
        click.echo("Next steps:")
        click.echo("  1. Run 'boba magic' and paste into your AI coding tool")
        click.echo("  2. boba serve")


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8787, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool):
    """Start the web UI server."""
    from simboba.config import find_boba_evals_dir

    # Check boba-evals exists
    if not find_boba_evals_dir():
        click.echo("No boba-evals/ folder found. Run 'boba init' first.", err=True)
        raise SystemExit(1)

    click.echo(f"Starting server at http://{host}:{port}")
    click.echo("Press Ctrl+C to stop")

    uvicorn.run(
        "simboba.server:app",
        host=host,
        port=port,
        reload=reload,
    )


@main.command()
def setup():
    """Print basic setup instructions (use 'boba magic' for guided AI setup)."""
    from simboba.config import find_boba_evals_dir

    if not find_boba_evals_dir():
        click.echo("No boba-evals/ folder found. Run 'boba init' first.", err=True)
        raise SystemExit(1)

    prompt = '''I need help setting up boba eval scripts for this project.

Please analyze this codebase and update the files in boba-evals/:

1. **boba-evals/setup.py** - Create test fixtures:
   - Look at the app's data models (User, Project, etc.)
   - Update get_context() to create test data needed for the agent
   - Return a dict with IDs and tokens the agent will need

2. **boba-evals/test.py** - Connect the agent:
   - Find how to call the agent/API in this codebase
   - Update the agent() function to call it with the test context
   - The function receives a message string and should return a response string

Look at the instructions in each file for more details.

After you're done, I'll run: boba run'''

    click.echo("")
    click.echo("Copy this prompt into your AI coding tool (Claude Code, Cursor, etc.):")
    click.echo("")
    click.echo(click.style("-" * 60, fg="bright_black"))
    click.echo(prompt)
    click.echo(click.style("-" * 60, fg="bright_black"))
    click.echo("")
    click.echo("Tip: Use 'boba magic' for a more guided AI setup experience.")


@main.command()
def magic():
    """Print a detailed prompt for AI coding tools to set up your evals."""
    from simboba.config import find_boba_evals_dir

    evals_dir = find_boba_evals_dir()
    if not evals_dir:
        click.echo("No boba-evals/ folder found. Run 'boba init' first.", err=True)
        raise SystemExit(1)

    prompt = '''# Setting Up Boba Evals

I need help setting up boba eval scripts for this project. Boba is a lightweight eval tracking tool that runs Python scripts and tracks results in a web UI.

## Your Task

Please analyze this codebase and configure the files in `boba-evals/`:

### 1. boba-evals/setup.py - Create Test Fixtures

This file creates any test data your agent needs to run. Look at the app's data models and determine what context is required.

```python
def get_context():
    """Create test fixtures and return context dict.

    Examples of what you might create:
    - Test user account with API token
    - Test project/workspace
    - Sample documents or data

    Return a dict with IDs and credentials the agent will need.
    """
    # Example:
    # user = create_test_user(email="eval@test.com")
    # return {"user_id": user.id, "token": user.api_token}
    return {}

def cleanup():
    """Optional: clean up test data after evals."""
    pass
```

### 2. boba-evals/test.py - Connect Your Agent

This file runs the actual evals. You need to update the `agent()` function to call your agent/API.

```python
from simboba import Boba
from setup import get_context, cleanup

boba = Boba()

def agent(message: str) -> str:
    """Call your agent and return its response.

    This function receives a user message and should:
    1. Get test context (user IDs, tokens, etc.)
    2. Call your agent/API with the message
    3. Return the response as a string
    """
    ctx = get_context()
    # TODO: Call your agent here
    # response = your_agent.chat(message, user_id=ctx["user_id"])
    # return response
    return "Not implemented"

if __name__ == "__main__":
    try:
        # Single eval example
        boba.eval(
            input="Hello, how can you help me?",
            output=agent("Hello, how can you help me?"),
            expected="Should greet the user and explain capabilities",
        )

        # Or run against a dataset:
        # boba.run(agent, dataset="my-dataset")

        print("Done! Run 'boba serve' to view results.")
    finally:
        cleanup()
```

## What to Look For

1. **How is the agent called?** Look for:
   - API endpoints (e.g., `/api/chat`, `/api/agent`)
   - Python functions (e.g., `agent.run()`, `chat()`)
   - SDK clients

2. **What context does it need?** Look for:
   - User authentication (user ID, API token)
   - Session or conversation IDs
   - Project/workspace context

3. **What's the response format?**
   - Extract the text response from whatever the agent returns

## After Setup

Once you've configured both files, tell me to run:
```bash
boba run
```

This will execute the eval script and track results. Then run `boba serve` to view results in the web UI.'''

    click.echo("")
    click.echo(click.style(BOBA_ART, fg="magenta"))
    click.echo(click.style("Copy the prompt below into your AI coding tool:", bold=True))
    click.echo(click.style("(Claude Code, Cursor, Windsurf, etc.)", fg="bright_black"))
    click.echo("")
    click.echo(click.style("=" * 60, fg="magenta"))
    click.echo(prompt)
    click.echo(click.style("=" * 60, fg="magenta"))
    click.echo("")
    click.echo("After your AI sets up the files, run:")
    click.echo(click.style("  boba run", fg="cyan"))
    click.echo("")


@main.command()
@click.argument("script", default="test.py")
def run(script: str):
    """Run an eval script.

    Automatically handles Docker vs local execution based on your config.

    \b
    Examples:
        boba run                    # Runs boba-evals/test.py
        boba run test.py            # Same as above
        boba run my_eval.py         # Runs boba-evals/my_eval.py
    """
    import subprocess
    import sys
    from pathlib import Path
    from simboba.config import load_config, inside_container, find_boba_evals_dir

    # Find boba-evals directory
    evals_dir = find_boba_evals_dir()
    if not evals_dir:
        click.echo("No boba-evals/ folder found. Run 'boba init' first.", err=True)
        raise SystemExit(1)

    # Resolve script path
    script_path = evals_dir / script
    if not script_path.exists():
        # Maybe they passed a full path
        script_path = Path(script)
        if not script_path.exists():
            click.echo(f"Script not found: {script}", err=True)
            click.echo("Make sure the file exists in boba-evals/ folder.")
            raise SystemExit(1)

    config = load_config()

    # If local or already in container, just run python
    if config.runtime == "local" or inside_container():
        cmd = [sys.executable, str(script_path)]
    else:
        # Docker mode - exec into container
        # Use relative path for Docker (container has different filesystem)
        service = config.service or "api"
        relative_script = f"boba-evals/{script}"
        cmd = ["docker", "compose", "exec", service, "python", relative_script]

    click.echo(f"Running: {' '.join(cmd)}")
    click.echo("")

    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except FileNotFoundError:
        if "docker" in cmd[0]:
            click.echo("Docker not found. Is Docker installed and running?", err=True)
        else:
            click.echo("Python not found.", err=True)
        raise SystemExit(1)
    except KeyboardInterrupt:
        click.echo("\nInterrupted.")
        raise SystemExit(130)


@main.command()
def datasets():
    """List all datasets."""
    from simboba import storage
    from simboba.config import find_boba_evals_dir

    if not find_boba_evals_dir():
        click.echo("No boba-evals/ folder found. Run 'boba init' first.", err=True)
        raise SystemExit(1)

    all_datasets = storage.list_datasets()
    if not all_datasets:
        click.echo("No datasets found. Run 'boba generate' to create one.")
        return

    click.echo(f"{'Name':<30} {'Cases':<10} {'Description'}")
    click.echo("-" * 70)
    for ds in all_datasets:
        desc = (ds.get("description") or "")[:30]
        click.echo(f"{ds['name']:<30} {ds.get('case_count', 0):<10} {desc}")


@main.command()
@click.argument("description")
def generate(description: str):
    """Generate a dataset from a description.

    Example: boba generate "A customer support chatbot for an e-commerce site"
    """
    from simboba import storage
    from simboba.config import find_boba_evals_dir
    from simboba.utils import LLMClient
    from simboba.prompts import build_dataset_generation_prompt

    if not find_boba_evals_dir():
        click.echo("No boba-evals/ folder found. Run 'boba init' first.", err=True)
        raise SystemExit(1)

    # Get model from settings
    model = storage.get_setting("model")
    click.echo(f"Using model: {model}")
    click.echo("Generating dataset...")

    # Generate the dataset
    prompt = build_dataset_generation_prompt(description)
    client = LLMClient(model=model)

    try:
        response = client.generate(prompt)
        result = client.parse_json_response(response)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    # Validate response
    if not isinstance(result, dict) or not result.get("name") or not result.get("cases"):
        click.echo("Error: Invalid response from LLM", err=True)
        raise SystemExit(1)

    # Handle duplicate names
    name = result["name"]
    if storage.dataset_exists(name):
        i = 1
        while storage.dataset_exists(f"{name}-{i}"):
            i += 1
        name = f"{name}-{i}"

    # Create the dataset
    dataset = {
        "name": name,
        "description": result.get("description", ""),
        "cases": result.get("cases", []),
    }
    storage.save_dataset(dataset)

    click.echo("")
    click.echo(click.style(f"Created dataset: {name}", fg="green"))
    click.echo(f"Description: {dataset['description']}")
    click.echo(f"Cases: {len(dataset['cases'])}")
    click.echo("")
    click.echo("Test cases:")
    for i, case_data in enumerate(dataset["cases"], 1):
        case_name = case_data.get("name", f"Case {i}")
        click.echo(f"  {i}. {case_name}")

    click.echo("")
    click.echo("Next: Update boba-evals/test.py to use this dataset:")
    click.echo(f'  boba.run(agent, dataset="{name}")')


@main.command()
def baseline():
    """Save a run as baseline for regression detection.

    Shows recent runs and lets you select one to save as the baseline.
    The baseline will be committed to git for tracking.
    """
    from simboba import storage
    from simboba.config import find_boba_evals_dir

    if not find_boba_evals_dir():
        click.echo("No boba-evals/ folder found. Run 'boba init' first.", err=True)
        raise SystemExit(1)

    # Get all runs
    runs = storage.list_runs()
    if not runs:
        click.echo("No runs found. Run 'boba run' first to create some runs.")
        return

    # Group by dataset ID, enrich with dataset name
    runs_by_dataset = {}
    for run in runs:
        ds_id = run.get("dataset_id", "_unknown")
        if ds_id not in runs_by_dataset:
            # Look up dataset name
            if ds_id == "_adhoc":
                ds_name = "_adhoc"
            else:
                ds = storage.get_dataset_by_id(ds_id)
                ds_name = ds["name"] if ds else f"(deleted: {ds_id[:8]})"
            runs_by_dataset[ds_id] = {"name": ds_name, "runs": []}
        runs_by_dataset[ds_id]["runs"].append(run)

    # Display runs
    click.echo("")
    click.echo(click.style("Recent runs:", bold=True))
    click.echo("")

    all_runs = []
    idx = 1
    for dataset_id, ds_info in runs_by_dataset.items():
        if dataset_id == "_adhoc":
            continue  # Skip ad-hoc single evals

        dataset_name = ds_info["name"]
        dataset_runs = ds_info["runs"]

        # Get existing baseline info (by dataset ID)
        existing_baseline = storage.get_baseline(dataset_id)
        baseline_info = ""
        if existing_baseline:
            baseline_info = click.style(f" (baseline: {existing_baseline.get('source_run', 'unknown')})", fg="bright_black")

        click.echo(click.style(f"  {dataset_name}", fg="cyan") + baseline_info)

        # Show last 3 runs for this dataset
        for run in dataset_runs[:3]:
            passed = run.get("passed", 0)
            failed = run.get("failed", 0)
            total = run.get("total", 0)
            score = run.get("score", 0)
            filename = run.get("filename", "unknown")
            started_at = run.get("started_at", "")[:16].replace("T", " ")

            status_color = "green" if failed == 0 else "yellow" if passed > failed else "red"
            status = click.style(f"{passed}/{total}", fg=status_color)

            click.echo(f"    {idx}. [{filename}] {status} ({score:.0f}%) - {started_at}")
            # Store dataset info with the run for later use
            run["_dataset_name"] = dataset_name
            all_runs.append(run)
            idx += 1

        click.echo("")

    if not all_runs:
        click.echo("No dataset runs found. Run 'boba run' with a dataset first.")
        return

    # Prompt for selection
    selection = click.prompt(
        "Select a run to save as baseline (enter number)",
        type=click.IntRange(1, len(all_runs)),
    )

    selected_run = all_runs[selection - 1]
    dataset_id = selected_run["dataset_id"]
    dataset_name = selected_run.get("_dataset_name", dataset_id)
    filename = selected_run["filename"]

    # Create baseline from run
    new_baseline = {
        "source_run": filename,
        "dataset_name": dataset_name,  # For display purposes
        "passed": selected_run.get("passed", 0),
        "failed": selected_run.get("failed", 0),
        "total": selected_run.get("total", 0),
        "score": selected_run.get("score"),
        "results": selected_run.get("results", {}),
    }

    # Save baseline using dataset ID (survives renames)
    storage.save_baseline(dataset_id, new_baseline)

    click.echo("")
    click.echo(click.style(f"Saved baseline for '{dataset_name}'", fg="green"))
    click.echo(f"  Source run: {filename}")
    click.echo(f"  Results: {new_baseline['passed']}/{new_baseline['total']} passed")
    click.echo("")
    click.echo("The baseline is saved to:")
    click.echo(click.style(f"  boba-evals/baselines/{dataset_id}.json", fg="cyan"))
    click.echo("")
    click.echo("To commit, run:")
    click.echo(click.style(f"  git add boba-evals/baselines/{dataset_id}.json", fg="bright_black"))
    click.echo(click.style("  git commit -m 'Update eval baseline'", fg="bright_black"))


@main.command()
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
def reset(force: bool):
    """Clear all run history (keeps datasets and baselines)."""
    from simboba import storage
    from simboba.config import find_boba_evals_dir

    evals_dir = find_boba_evals_dir()
    if not evals_dir:
        click.echo("No boba-evals/ folder found. Nothing to reset.")
        return

    runs_dir = evals_dir / "runs"
    if not runs_dir.exists() or not any(runs_dir.iterdir()):
        click.echo("No runs found. Nothing to reset.")
        return

    if not force:
        click.echo("This will delete all run history from boba-evals/runs/")
        click.echo("Datasets and baselines will be preserved.")
        if not click.confirm("Are you sure?"):
            click.echo("Aborted.")
            return

    count = storage.clear_runs()
    click.echo(f"Deleted {count} run(s).")


if __name__ == "__main__":
    main()
