"""CLI entry point for simboba."""

import json
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

    # Don't redirect 'init' command - it needs to run locally to create config
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        return

    # Don't redirect 'config' command (if we add one)
    if len(sys.argv) > 1 and sys.argv[1] == "config":
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

  boba - LLM eval datasets & judge

  Generate test cases, run evals, and judge with AI.
    """
    pass


# Call docker exec wrapper early
_maybe_exec_in_docker()


@main.command()
@click.option("--docker", "use_docker", is_flag=True, help="Quick setup for Docker Compose environments")
@click.option("--local", "use_local", is_flag=True, help="Quick setup for local Python environments")
def init(use_docker: bool, use_local: bool):
    """Initialize evals folder with a starter template."""
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

    template = '''"""Example eval configuration.

Edit this file to connect your agent/API for evaluation.

Tip: boba automatically loads .env files from the current directory.
"""

import os
import requests
from simboba import Eval


def my_agent(messages):
    """Call your agent and return its response.

    Args:
        messages: List of conversation messages, each with:
            - role: "user" or "assistant"
            - message: The message content
            - attachments: Optional list of attachments

    Returns:
        The agent's response as a string.
    """
    # TIP: Validate env vars inside the function, not at module level.
    # This gives clear errors instead of silent import failures.
    api_url = os.environ.get("MY_API_URL", "http://localhost:8000")
    api_key = os.environ.get("MY_API_KEY")
    # if not api_key:
    #     raise ValueError("MY_API_KEY environment variable not set")

    # Example: Call your API
    # response = requests.post(f"{api_url}/chat",
    #     headers={"Authorization": f"Bearer {api_key}"},
    #     json={"messages": [{"role": m["role"], "content": m["message"]} for m in messages]}
    # )
    # return response.json()["response"]

    # Placeholder - replace with your actual agent call
    last_message = messages[-1]["message"]
    return f"Echo: {last_message}"


# Register the eval
my_eval = Eval(
    name="my-agent",
    fn=my_agent,
)
'''

    (evals_dir / "example.py").write_text(template)

    # Create setup.py template
    setup_template = '''"""Setup script for eval test fixtures.

This script creates any test data needed before running evals.
Run with: boba setup

The setup() function should return a dict of credentials/context
that will be printed to the terminal and can be used in your evals.
"""


def setup():
    """Create test fixtures and return credentials.

    Returns:
        dict: Credentials and context for evals. Example:
            {
                "user_id": 123,
                "project_id": 456,
                "api_token": "test-token-xxx",
            }
    """
    # Example: Create a test user
    # user = create_user(email="eval-test@example.com")
    #
    # Example: Create a test project
    # project = create_project(user_id=user.id, name="Eval Test Project")
    #
    # Return credentials for use in evals
    return {
        "message": "Edit setup.py to create your test fixtures",
    }


def teardown(ctx: dict):
    """Optional: Clean up test fixtures.

    Args:
        ctx: The dict returned by setup()
    """
    # Example: Delete test user
    # delete_user(ctx["user_id"])
    pass
'''

    (evals_dir / "setup.py").write_text(setup_template)

    # Copy README from package
    import simboba
    package_dir = Path(simboba.__file__).parent.parent
    readme_src = package_dir / "README.md"
    if readme_src.exists():
        (evals_dir / "README.md").write_text(readme_src.read_text())
    else:
        # Fallback if README not found
        (evals_dir / "README.md").write_text("# Evals\n\nSee https://github.com/your-repo/simboba for documentation.\n")

    # Add .gitignore for database
    (evals_dir / ".gitignore").write_text("# boba database\n*.db\n")

    # Success message
    click.echo("")
    click.echo(click.style("✓ Created evals/ folder", fg="green"))
    click.echo(f"  - {CONFIG_FILENAME} (runtime: {runtime})")
    click.echo("  - example.py")
    click.echo("  - setup.py")
    click.echo("  - README.md")
    click.echo("")

    if runtime == "docker-compose":
        click.echo("Next steps:")
        click.echo("")
        click.echo("  1. Add simboba to your container's dependencies:")
        click.echo(click.style("     # requirements.txt, Pipfile, or pyproject.toml", fg="bright_black"))
        click.echo("     simboba")
        click.echo("")
        click.echo("  2. Expose port 8787 in docker-compose.yml:")
        click.echo(click.style(f"     services:", fg="bright_black"))
        click.echo(click.style(f"       {service}:", fg="bright_black"))
        click.echo(click.style(f"         ports:", fg="bright_black"))
        click.echo(click.style(f"           - \"8787:8787\"", fg="bright_black"))
        click.echo("")
        click.echo("  3. Rebuild and start:")
        click.echo(click.style("     docker compose build && docker compose up -d", fg="cyan"))
        click.echo("")
        click.echo("  4. Run boba:")
        click.echo(click.style("     boba serve", fg="cyan"))
    else:
        click.echo("Next steps:")
        click.echo("  1. Edit evals/example.py to connect your agent")
        click.echo("  2. Run: boba serve")


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8787, help="Port to bind to")
@click.option("--config", "-c", "config_path", help="Path to eval config file or folder (default: ./evals)")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, config_path: str, reload: bool):
    """Start the web UI server."""
    from pathlib import Path
    from simboba.database import init_db
    from simboba.server import set_loaded_evals, set_load_errors

    # Initialize database
    init_db()

    # Load eval config - try folder first, then file, then default evals/ folder
    from simboba.runner import load_config, load_evals_folder

    evals = None
    load_errors = []

    if config_path:
        path = Path(config_path)
        try:
            if path.is_dir():
                result = load_evals_folder(config_path, return_details=True)
                evals = result.evals
                load_errors = result.failed
            else:
                evals = load_config(config_path)
        except Exception as e:
            click.echo(click.style(f"✗ Could not load config: {e}", fg="red"), err=True)
    else:
        # Try default evals/ folder
        if Path("evals").is_dir():
            try:
                result = load_evals_folder("evals", return_details=True)
                evals = result.evals
                load_errors = result.failed
            except Exception as e:
                click.echo(click.style(f"✗ No evals loaded: {e}", fg="red"), err=True)

    # Show what loaded
    if evals:
        set_loaded_evals(evals)
        for name in evals.keys():
            click.echo(click.style(f"  ✓ {name}", fg="green"))

    # Show what failed
    if load_errors:
        set_load_errors(load_errors)
        for filename, error in load_errors:
            click.echo(click.style(f"  ✗ {filename}", fg="red") + f": {error}")

    # Summary
    if evals or load_errors:
        total = len(evals) if evals else 0
        msg = f"Loaded {total} eval(s)"
        if load_errors:
            msg += click.style(f", {len(load_errors)} failed", fg="red")
        click.echo(msg)
    else:
        click.echo("No evals loaded. Create evals in ./evals/ folder.")

    click.echo("")
    click.echo(f"Starting server at http://{host}:{port}")
    click.echo("Press Ctrl+C to stop")

    uvicorn.run(
        "simboba.server:app",
        host=host,
        port=port,
        reload=reload,
    )


@main.command()
@click.option("--config", "-c", "config_path", help="Path to eval config file or folder (default: ./evals)")
@click.option("--dataset", "-d", required=True, help="Dataset name to run evals against")
@click.option("--eval", "-e", "eval_name", help="Specific eval to run (default: all)")
@click.option("--no-judge", is_flag=True, help="Skip LLM judging, just run the functions")
def run(config_path: str, dataset: str, eval_name: str, no_judge: bool):
    """Run evals headlessly (for CI)."""
    from pathlib import Path
    from simboba.database import init_db, get_session_factory
    from simboba.models import Dataset
    from simboba.runner import load_config, load_evals_folder, run_eval

    init_db()
    Session = get_session_factory()
    db = Session()

    try:
        # Load config - try specified path, then default evals/ folder
        evals = None
        if config_path:
            path = Path(config_path)
            try:
                if path.is_dir():
                    result = load_evals_folder(config_path, return_details=True)
                    evals = result.evals
                    if result.failed:
                        for filename, error in result.failed:
                            click.echo(click.style(f"  ✗ {filename}", fg="red") + f": {error}")
                else:
                    evals = load_config(config_path)
            except Exception as e:
                click.echo(f"Error loading config: {e}", err=True)
                raise SystemExit(1)
        else:
            # Try default evals/ folder
            if Path("evals").is_dir():
                try:
                    result = load_evals_folder("evals", return_details=True)
                    evals = result.evals
                    if result.failed:
                        for filename, error in result.failed:
                            click.echo(click.style(f"  ✗ {filename}", fg="red") + f": {error}")
                except Exception as e:
                    click.echo(f"Error loading evals: {e}", err=True)
                    raise SystemExit(1)
            else:
                click.echo("Error: No evals/ folder found. Run 'boba init' first or specify --config.", err=True)
                raise SystemExit(1)

        if not evals:
            click.echo("Error: No evals loaded.", err=True)
            raise SystemExit(1)

        # Get dataset
        ds = db.query(Dataset).filter(Dataset.name == dataset).first()
        if not ds:
            click.echo(f"Error: Dataset '{dataset}' not found", err=True)
            raise SystemExit(1)

        cases = [c.to_dict() for c in ds.cases]
        if not cases:
            click.echo(f"Error: Dataset '{dataset}' has no cases", err=True)
            raise SystemExit(1)

        # Determine which evals to run
        if eval_name:
            if eval_name not in evals:
                click.echo(f"Error: Eval '{eval_name}' not found in config", err=True)
                click.echo(f"Available: {', '.join(evals.keys())}")
                raise SystemExit(1)
            evals_to_run = {eval_name: evals[eval_name]}
        else:
            evals_to_run = evals

        # Setup judge
        judge_fn = None
        if not no_judge:
            try:
                from simboba.judge import create_judge
                judge_fn = create_judge()
                click.echo("Using LLM judge")
            except (ImportError, ValueError) as e:
                click.echo(f"Warning: Could not create LLM judge ({e}), using simple judge")
                from simboba.judge import create_simple_judge
                judge_fn = create_simple_judge()

        # Run evals
        all_passed = True
        for name, eval_config in evals_to_run.items():
            click.echo(f"\nRunning eval: {name}")
            click.echo(f"  Dataset: {dataset} ({len(cases)} cases)")
            click.echo("-" * 40)

            result = run_eval(eval_config, cases, judge_fn)

            click.echo(f"  Passed: {result.passed}/{result.total}")
            click.echo(f"  Failed: {result.failed}/{result.total}")
            click.echo(f"  Score:  {result.score:.1f}%")

            if result.failed > 0:
                all_passed = False
                click.echo("\n  Failed cases:")
                for case_result in result.results:
                    if not case_result.passed:
                        click.echo(f"    - Case {case_result.case_id}: {case_result.error_message or case_result.reasoning}")

        # Exit with error code if any failed
        if not all_passed:
            raise SystemExit(1)

    finally:
        db.close()


@main.command()
@click.option("--dataset", required=True, help="Dataset name to export")
@click.option("--output", "-o", required=True, help="Output file path")
def export(dataset: str, output: str):
    """Export a dataset to JSON file."""
    from simboba.database import init_db, get_session_factory
    from simboba.models import Dataset

    init_db()
    Session = get_session_factory()
    db = Session()

    try:
        ds = db.query(Dataset).filter(Dataset.name == dataset).first()
        if not ds:
            click.echo(f"Error: Dataset '{dataset}' not found", err=True)
            raise SystemExit(1)

        data = {
            "name": ds.name,
            "description": ds.description,
            "cases": [case.to_dict() for case in ds.cases],
        }

        with open(output, "w") as f:
            json.dump(data, f, indent=2)

        click.echo(f"Exported {len(ds.cases)} cases to {output}")
    finally:
        db.close()


@main.command("import")
@click.option("--input", "-i", "input_file", required=True, help="Input JSON file path")
@click.option("--dataset", help="Override dataset name (default: use name from file)")
def import_dataset(input_file: str, dataset: str):
    """Import a dataset from JSON file."""
    from simboba.database import init_db, get_session_factory
    from simboba.models import Dataset, EvalCase

    init_db()
    Session = get_session_factory()
    db = Session()

    try:
        with open(input_file) as f:
            data = json.load(f)

        name = dataset or data.get("name")
        if not name:
            click.echo("Error: No dataset name provided", err=True)
            raise SystemExit(1)

        existing = db.query(Dataset).filter(Dataset.name == name).first()
        if existing:
            click.echo(f"Error: Dataset '{name}' already exists", err=True)
            raise SystemExit(1)

        ds = Dataset(name=name, description=data.get("description"))
        db.add(ds)
        db.flush()

        for case_data in data.get("cases", []):
            case = EvalCase(
                dataset_id=ds.id,
                name=case_data.get("name"),
                inputs=case_data.get("inputs", []),
                expected_outcome=case_data.get("expected_outcome", ""),
            )
            db.add(case)

        db.commit()
        click.echo(f"Imported {len(data.get('cases', []))} cases to dataset '{name}'")
    finally:
        db.close()


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
            click.echo("No datasets found")
            return

        click.echo(f"{'Name':<30} {'Cases':<10} {'Description'}")
        click.echo("-" * 70)
        for ds in all_datasets:
            desc = (ds.description or "")[:30]
            click.echo(f"{ds.name:<30} {len(ds.cases):<10} {desc}")
    finally:
        db.close()


@main.command()
def evals():
    """List all loaded evals and show any import errors."""
    from pathlib import Path
    from simboba.runner import load_evals_folder

    if not Path("evals").is_dir():
        click.echo("No evals/ folder found. Run 'boba init' first.")
        return

    try:
        result = load_evals_folder("evals", return_details=True)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        return

    # Show loaded evals
    if result.loaded:
        click.echo("Loaded evals:")
        for filename, eval_names in result.loaded:
            for name in eval_names:
                click.echo(click.style(f"  ✓ {name}", fg="green") + click.style(f"  ({filename})", fg="bright_black"))
    else:
        click.echo("No evals loaded.")

    # Show failed imports
    if result.failed:
        click.echo("")
        click.echo("Failed to load:")
        for filename, error in result.failed:
            click.echo(click.style(f"  ✗ {filename}", fg="red") + f": {error}")

    # Summary
    total_evals = sum(len(names) for _, names in result.loaded)
    click.echo("")
    click.echo(f"Total: {total_evals} eval(s) from {len(result.loaded)} file(s)", nl=False)
    if result.failed:
        click.echo(click.style(f", {len(result.failed)} failed", fg="red"))
    else:
        click.echo("")


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


@main.command()
def config():
    """Show current boba configuration."""
    from pathlib import Path
    from simboba.config import load_config, find_config, inside_container

    config_path = find_config()

    if not config_path:
        click.echo("No configuration found.")
        click.echo("")
        click.echo("Run 'boba init' to create a configuration.")
        return

    cfg = load_config()
    if not cfg:
        click.echo(f"Found {config_path} but could not parse it.")
        return

    click.echo(f"Configuration: {config_path}")
    click.echo("")
    click.echo(f"  runtime: {cfg.runtime}")
    if cfg.service:
        click.echo(f"  service: {cfg.service}")

    click.echo("")
    if inside_container():
        click.echo(click.style("Currently running inside a container", fg="green"))
    else:
        if cfg.runtime == "docker-compose":
            click.echo(f"Commands will exec into Docker service '{cfg.service}'")
        else:
            click.echo("Commands will run locally")


@main.command()
@click.option("--eval", "-e", "eval_name", help="Specific eval to test (default: first one found)")
@click.option("--message", "-m", default="Hello, this is a test message.", help="Test message to send")
def test(eval_name: str, message: str):
    """Test connection to your agent.

    Sends a test message through your eval function to verify it can reach your app.

    Example: boba test
             boba test --eval my-agent
             boba test -m "What is the status of order 123?"
    """
    from pathlib import Path
    from simboba.runner import load_evals_folder, load_config
    import time

    # Load evals
    evals = None
    if Path("evals").is_dir():
        try:
            evals = load_evals_folder("evals")
        except Exception as e:
            click.echo(f"Error loading evals: {e}", err=True)
            raise SystemExit(1)
    else:
        click.echo("No evals/ folder found. Run 'boba init' first.", err=True)
        raise SystemExit(1)

    if not evals:
        click.echo("No evals found in evals/ folder.", err=True)
        raise SystemExit(1)

    # Select eval to test
    if eval_name:
        if eval_name not in evals:
            click.echo(f"Eval '{eval_name}' not found.", err=True)
            click.echo(f"Available: {', '.join(evals.keys())}")
            raise SystemExit(1)
        selected_eval = evals[eval_name]
    else:
        # Use the first one
        eval_name = list(evals.keys())[0]
        selected_eval = evals[eval_name]

    click.echo(f"Testing eval: {eval_name}")
    click.echo(f"Message: {message}")
    click.echo("-" * 40)

    # Build test input
    test_input = [{"role": "user", "message": message, "attachments": []}]

    # Run the test
    start_time = time.time()
    try:
        result = selected_eval.run(test_input)
        elapsed = time.time() - start_time

        click.echo(f"Response ({elapsed:.2f}s):")
        click.echo(result)
        click.echo("-" * 40)
        click.echo(click.style("Connection successful!", fg="green"))
    except Exception as e:
        elapsed = time.time() - start_time
        click.echo(click.style(f"Error ({elapsed:.2f}s): {e}", fg="red"), err=True)
        raise SystemExit(1)


@main.command()
def setup():
    """Run setup script and print credentials.

    Executes evals/setup.py and prints the returned credentials.
    Use this to create test fixtures before running evals.
    """
    from pathlib import Path
    import importlib.util
    import json

    setup_path = Path("evals/setup.py")
    if not setup_path.exists():
        click.echo("No evals/setup.py found. Run 'boba init' first.", err=True)
        raise SystemExit(1)

    # Load the setup module
    try:
        spec = importlib.util.spec_from_file_location("setup", setup_path)
        setup_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup_module)
    except Exception as e:
        click.echo(click.style(f"Error loading setup.py: {e}", fg="red"), err=True)
        raise SystemExit(1)

    if not hasattr(setup_module, "setup"):
        click.echo("setup.py must define a setup() function", err=True)
        raise SystemExit(1)

    # Run setup
    click.echo("Running setup...")
    try:
        ctx = setup_module.setup()
    except Exception as e:
        click.echo(click.style(f"Setup failed: {e}", fg="red"), err=True)
        raise SystemExit(1)

    if not isinstance(ctx, dict):
        click.echo("setup() must return a dict", err=True)
        raise SystemExit(1)

    # Print credentials
    click.echo("")
    click.echo(click.style("Setup complete!", fg="green"))
    click.echo("")
    click.echo("Credentials:")
    for key, value in ctx.items():
        click.echo(f"  {key}: {value}")

    # Also print as JSON for easy copying
    click.echo("")
    click.echo("JSON:")
    click.echo(click.style(json.dumps(ctx, indent=2, default=str), fg="cyan"))


@main.command()
@click.argument("description")
def generate(description: str):
    """Generate a dataset from a product description.

    Example: simboba generate "A customer support chatbot for an e-commerce site"
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
        click.echo(f"Created dataset: {name}")
        click.echo(f"Description: {ds.description}")
        click.echo(f"Cases: {len(result['cases'])}")
        click.echo("")
        click.echo("Test cases:")
        for i, case_data in enumerate(result["cases"], 1):
            case_name = case_data.get("name", f"Case {i}")
            click.echo(f"  {i}. {case_name}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
