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

    # Don't redirect 'init' or 'setup' command - they need to run locally
    if len(sys.argv) > 1 and sys.argv[1] in ("init", "setup"):
        return

    from simboba.config import maybe_exec_in_docker
    maybe_exec_in_docker()


@click.group()
@click.version_option(version="0.1.0")
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
    """Initialize evals folder with starter templates."""
    from pathlib import Path
    from simboba.config import BobaConfig, save_config, CONFIG_FILENAME

    evals_dir = Path("evals")
    if evals_dir.exists():
        click.echo("evals/ folder already exists")
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
    click.echo(click.style("Created evals/ folder", fg="green"))
    click.echo("  - setup.py       (shared test fixtures)")
    click.echo("  - test_chat.py   (example eval script)")
    click.echo("")

    if runtime == "docker-compose":
        click.echo("Next steps:")
        click.echo("")
        click.echo("  1. Add simboba to your container's dependencies")
        click.echo("  2. Expose port 8787 in docker-compose.yml")
        click.echo("  3. docker compose build && docker compose up -d")
        click.echo("  4. Edit evals/setup.py and evals/test_chat.py")
        click.echo("     (or run 'boba setup' for AI assistance)")
        click.echo("  5. python evals/test_chat.py")
        click.echo("  6. boba serve")
    else:
        click.echo("Next steps:")
        click.echo("  1. Edit evals/setup.py and evals/test_chat.py")
        click.echo("     (or run 'boba setup' for AI assistance)")
        click.echo("  2. python evals/test_chat.py")
        click.echo("  3. boba serve")


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
    """Print instructions for AI-assisted setup.

    Outputs a prompt you can paste into Claude Code, Cursor, or other AI coding tools
    to automatically configure your eval scripts.
    """
    from pathlib import Path

    if not Path("evals").is_dir():
        click.echo("No evals/ folder found. Run 'boba init' first.", err=True)
        raise SystemExit(1)

    prompt = '''I need help setting up boba eval scripts for this project.

Please analyze this codebase and update the files in evals/:

1. **evals/setup.py** - Create test fixtures:
   - Look at the app's data models (User, Project, etc.)
   - Update get_context() to create test data needed for the agent
   - Return a dict with IDs and tokens the agent will need

2. **evals/test_chat.py** - Connect the agent:
   - Find how to call the agent/API in this codebase
   - Update the agent() function to call it with the test context
   - The function receives a message string and should return a response string

Look at the instructions in each file for more details.

After you're done, I'll run: python evals/test_chat.py'''

    click.echo("")
    click.echo("Copy this prompt into your AI coding tool (Claude Code, Cursor, etc.):")
    click.echo("")
    click.echo(click.style("-" * 60, fg="bright_black"))
    click.echo(prompt)
    click.echo(click.style("-" * 60, fg="bright_black"))
    click.echo("")
    click.echo("The AI will analyze your codebase and update the eval scripts.")


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
