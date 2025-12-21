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
@click.version_option(version="0.1.2")
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

    # Create the evals directory
    evals_dir.mkdir()

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
        # Copy test_chat.py
        shutil.copy(samples_dir / "test_chat.py", evals_dir / "test_chat.py")
    else:
        # Fallback if samples not found
        (evals_dir / "setup.py").write_text("# Setup file - see boba docs\ndef get_context():\n    return {}\n\ndef cleanup():\n    pass\n")
        (evals_dir / "test_chat.py").write_text("# Eval file - see boba docs\nfrom simboba import Boba\n\nboba = Boba()\n")

    # Add .gitignore for database
    (evals_dir / ".gitignore").write_text("# boba database\n*.db\n")

    # Success message
    click.echo("")
    click.echo(click.style("Created boba-evals/ folder", fg="green"))
    click.echo("  - setup.py       (shared test fixtures)")
    click.echo("  - test_chat.py   (example eval script)")
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
    from simboba.database import init_db

    # Initialize database
    init_db()

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

2. **boba-evals/test_chat.py** - Connect the agent:
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

### 2. boba-evals/test_chat.py - Connect Your Agent

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
@click.argument("script", default="test_chat.py")
def run(script: str):
    """Run an eval script.

    Automatically handles Docker vs local execution based on your config.

    \b
    Examples:
        boba run                    # Runs boba-evals/test_chat.py
        boba run test_chat.py       # Same as above
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
    from simboba.database import init_db, get_session_factory
    from simboba.models import Dataset

    init_db()
    Session = get_session_factory()
    db = Session()

    try:
        all_datasets = db.query(Dataset).order_by(Dataset.name).all()
        if not all_datasets:
            click.echo("No datasets found. Run 'boba generate' to create one.")
            return

        click.echo(f"{'Name':<30} {'Cases':<10} {'Description'}")
        click.echo("-" * 70)
        for ds in all_datasets:
            desc = (ds.description or "")[:30]
            click.echo(f"{ds.name:<30} {len(ds.cases):<10} {desc}")
    finally:
        db.close()


@main.command()
@click.argument("description")
def generate(description: str):
    """Generate a dataset from a description.

    Example: boba generate "A customer support chatbot for an e-commerce site"
    """
    from simboba.database import init_db, get_session_factory
    from simboba.models import Dataset, EvalCase, Settings
    from simboba.utils.models import LLMClient
    from simboba.prompts import build_dataset_generation_prompt

    init_db()
    Session = get_session_factory()
    db = Session()

    try:
        # Get model from settings
        model = Settings.get(db, "model")
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
        existing = db.query(Dataset).filter(Dataset.name == name).first()
        if existing:
            i = 1
            while db.query(Dataset).filter(Dataset.name == f"{name}-{i}").first():
                i += 1
            name = f"{name}-{i}"

        # Create the dataset
        ds = Dataset(name=name, description=result.get("description", ""))
        db.add(ds)
        db.flush()

        # Create the cases
        for case_data in result["cases"]:
            case = EvalCase(
                dataset_id=ds.id,
                name=case_data.get("name"),
                inputs=case_data.get("inputs", []),
                expected_outcome=case_data.get("expected_outcome", ""),
                expected_source=case_data.get("expected_source"),
            )
            db.add(case)

        db.commit()

        click.echo("")
        click.echo(click.style(f"Created dataset: {name}", fg="green"))
        click.echo(f"Description: {ds.description}")
        click.echo(f"Cases: {len(result['cases'])}")
        click.echo("")
        click.echo("Test cases:")
        for i, case_data in enumerate(result["cases"], 1):
            case_name = case_data.get("name", f"Case {i}")
            click.echo(f"  {i}. {case_name}")

        click.echo("")
        click.echo("Next: Update evals/test_chat.py to use this dataset:")
        click.echo(f'  boba.run(agent, dataset="{name}")')

    finally:
        db.close()


@main.command()
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
def reset(force: bool):
    """Reset the database (deletes all data)."""
    import os
    from pathlib import Path
    from simboba.database import DEFAULT_DB_PATH

    db_path = Path(os.environ.get("SIMBOBA_DB_PATH", DEFAULT_DB_PATH))

    if not db_path.exists():
        click.echo("No database file found. Nothing to reset.")
        return

    if not force:
        click.echo(f"This will delete: {db_path}")
        click.echo("All datasets, cases, and eval runs will be permanently removed.")
        if not click.confirm("Are you sure?"):
            click.echo("Aborted.")
            return

    db_path.unlink()
    click.echo("Database reset successfully.")


if __name__ == "__main__":
    main()
