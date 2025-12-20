# simboba

A lightweight tool for generating annotated eval datasets and running LLM-as-judge evaluations.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Start the web UI
simboba serve

# With eval config
simboba serve --config evals.py
```

## Project Structure

```
simboba/
├── simboba/               # Main package
│   ├── __init__.py       # Exports Eval class
│   ├── cli.py            # Click CLI commands
│   ├── config.py         # Configuration (.boba.yaml) handling
│   ├── database.py       # SQLite/SQLAlchemy setup
│   ├── models.py         # Dataset, EvalCase, EvalRun, EvalResult
│   ├── server.py         # FastAPI REST API
│   ├── runner.py         # Eval execution logic
│   ├── judge.py          # LLM judge implementation
│   └── static/           # Web UI
│       ├── index.html    # Main page structure + CSS
│       └── app.js        # Frontend JavaScript
├── tests/
│   ├── conftest.py       # Pytest fixtures
│   └── test_core_flows.py # Core functionality tests
├── pyproject.toml        # Package config
└── REQUIREMENTS.md       # Product requirements
```

## Architecture

### Data Model

- **Dataset**: Named collection of eval cases
- **EvalCase**: Single test case with inputs, expected outcome, and optional source reference
- **EvalRun**: One execution of an eval against a dataset
- **EvalResult**: Per-case result with actual output, judgment, reasoning

### EvalCase Structure

```python
{
    "inputs": [
        {"role": "user", "message": "...", "attachments": [{"file": "doc.pdf"}]}
    ],
    "expected_outcome": "Agent should...",
    "expected_source": {        # optional - for verification
        "file": "doc.pdf",      # which file
        "page": 12,             # which page
        "excerpt": "..."        # optional snippet
    }
}
```

### Eval Class Pattern

```python
from simboba import Eval

def my_function(messages):
    return "response"

my_eval = Eval(
    name="my-eval",
    fn=my_function,
    transform_inputs=lambda msgs: {"messages": msgs},  # optional
    transform_output=lambda r: str(r),                 # optional
)
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/datasets` | GET, POST | List/create datasets |
| `/api/datasets/{id}` | GET, DELETE | Get/delete dataset |
| `/api/datasets/{id}/export` | GET | Export dataset as JSON |
| `/api/datasets/import` | POST | Import dataset from JSON |
| `/api/cases` | GET, POST | List/create cases |
| `/api/cases/{id}` | GET, PUT, DELETE | CRUD for single case |
| `/api/generate` | POST | Generate synthetic cases |
| `/api/generate/with-files` | POST | Generate cases from PDF files (multipart form) |
| `/api/generate/accept` | POST | Accept generated cases |
| `/api/evals` | GET | List loaded evals |
| `/api/evals/errors` | GET | List eval import errors |
| `/api/evals/{name}/test` | POST | Test connection to an eval |
| `/api/runs` | GET, POST | List runs / start new run |
| `/api/runs/{id}` | GET, DELETE | Get/delete run |

### CLI Commands

```bash
boba init                                        # Interactive setup wizard
boba init --docker                               # Quick setup for Docker Compose
boba init --local                                # Quick setup for local Python
boba config                                      # Show current configuration
boba serve [--config evals.py] [--port 8787]     # Start web UI
boba test [--eval name] [-m "message"]           # Test connection to your agent
boba evals                                       # List loaded evals and show errors
boba run --dataset name [--eval name]            # Run evals headlessly
boba setup                                       # Run setup script, print credentials
boba export --dataset name -o file.json          # Export dataset
boba import -i file.json                         # Import dataset
boba datasets                                    # List all datasets
boba generate "description"                      # Generate dataset from CLI
boba reset                                       # Delete database (all data)
```

### Docker Integration

Boba supports transparent Docker Compose integration. When configured for Docker, all commands automatically exec into your container.

**Setup:**
```bash
$ boba init
? How do you run your app? [local / docker-compose]
> docker-compose
? Docker Compose service name [api]
> api
```

This creates `evals/.boba.yaml`:
```yaml
runtime: docker-compose
service: api
```

**How it works:**
- `boba serve`, `boba generate`, etc. automatically run inside your container
- The `init` command always runs locally (to create the config)
- Set `BOBA_NO_DOCKER=1` to bypass Docker exec temporarily

**Configuration file:** `evals/.boba.yaml`
```yaml
runtime: local           # or "docker-compose"
service: api             # Docker Compose service name (if docker-compose)
```

### Setup Script

The `evals/setup.py` file creates test fixtures before running evals. Use it to set up test users, projects, or any data your evals need.

**Structure:**
```python
# evals/setup.py

def setup():
    """Create test fixtures, return credentials dict."""
    # Create test data using your app's models/APIs
    user = create_user(email="eval@test.com")
    project = create_project(user_id=user.id)

    return {
        "user_id": user.id,
        "project_id": project.id,
    }

def teardown(ctx: dict):
    """Optional: clean up test data."""
    delete_user(ctx["user_id"])
```

**Run setup:**
```bash
$ boba setup

Running setup...

Setup complete!

Credentials:
  user_id: 123
  project_id: 456

JSON:
{
  "user_id": 123,
  "project_id": 456
}
```

**Using credentials in evals:** Import your setup module or hardcode the IDs after running setup once.

## Design System

### Colors (Tailwind Zinc + Taro accent)

| Token | Value | Usage |
|-------|-------|-------|
| `--zinc-50` | #fafafa | Page background |
| `--zinc-100` | #f4f4f5 | Subtle backgrounds |
| `--zinc-200` | #e4e4e7 | Borders |
| `--zinc-400` | #a1a1aa | Muted text, icons |
| `--zinc-500` | #71717a | Secondary text |
| `--zinc-600` | #52525b | Labels |
| `--zinc-900` | #18181b | Primary text |
| `--taro` | #8B7BA5 | Primary accent (buttons, links, active states) |
| `--taro-dark` | #7A6B94 | Hover state |
| `--taro-light` | #F4F2F7 | Active item backgrounds |
| `--green-500` | #22c55e | Pass states |
| `--red-500` | #ef4444 | Fail/error states |

### Typography

- **Body/UI**: DM Sans (400, 500, 600)
- **Data/Code**: JetBrains Mono (400, 500)
- Use `.mono` class for numeric data, scores, counts

### Components

- **Buttons**: `.btn`, `.btn-primary`, `.btn-secondary`, `.btn-danger`, `.btn-sm`
- **Forms**: `.input-group` wrapper with `label`, `input`/`textarea`/`select`
- **Tabs**: `.tabs` container with `.tab` items
- **Cards**: `.case-item`, `.run-item`, `.result-item`
- **Status**: `.status-pearl.pass`, `.status-pearl.fail`, `.status-pearl.running`
- **Badges**: `.result-badge.pass`, `.result-badge.fail`
- **Empty states**: `.empty-state` centered container

### Design Principles

1. **Sharp corners** - max 4px border-radius
2. **Minimal** - zinc grays, single taro accent
3. **Status pearls** - small colored dots as visual motif
4. **Monospace data** - scores, counts, timestamps use JetBrains Mono
5. **Dense but readable** - 14px base, compact spacing

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Key Test Patterns

Tests use an isolated in-memory SQLite database via the `client` fixture in `conftest.py`. This ensures each test runs independently.

### Adding New Features

1. Add model to `models.py` if needed
2. Add API endpoint to `server.py`
3. Update UI in `app.js` (render functions)
4. Add styles to `index.html` if needed
5. Write tests in `test_core_flows.py`

### Common Modifications

**Adding a new eval option:**
1. Update `Eval` class in `eval.py`
2. Update `run_case` in `runner.py` to handle it

**Adding a new API field:**
1. Update Pydantic models in `server.py`
2. Update SQLAlchemy model in `models.py`
3. Update `to_dict()` method

**Changing UI styling:**
1. CSS variables and classes are in `index.html`
2. Dynamic elements use inline styles with `var(--token)` in `app.js`

## File-Based Generation

Generate test cases from PDF documents:

1. **Upload PDFs** in the Generate tab (drag & drop or click to browse)
2. **Describe your agent** - e.g., "A document Q&A assistant"
3. **Generate** - Claude reads the PDFs and creates test cases based on actual content
4. **Review** - each generated case includes:
   - Question about the document
   - Expected outcome
   - Source reference (file + page number) for quick verification

The source reference (`expected_source`) helps you verify that the expected outcome is correct without opening the PDF.

```python
# Example generated case
{
    "name": "Cancellation policy question",
    "inputs": [{"role": "user", "message": "Can I cancel early?", "attachments": [{"file": "contract.pdf"}]}],
    "expected_outcome": "Agent should explain the 30-day notice requirement",
    "expected_source": {"file": "contract.pdf", "page": 12, "excerpt": "30 days written notice required"}
}
```

## Judge Configuration

The LLM judge uses Claude by default. Set `ANTHROPIC_API_KEY` environment variable.

```python
# Custom judge function signature
def judge(inputs, expected, actual) -> tuple[bool, str]:
    """Returns (passed, reasoning)"""
    pass
```

A simple keyword-matching fallback (`create_simple_judge()`) is used when no API key is available.

## Documentation

**IMPORTANT: Keep README.md updated!**

When adding new features (CLI commands, API endpoints, UI features):
1. Update the Commands table in README.md
2. Add usage examples if the feature is user-facing
3. Update this CLAUDE.md file with technical details

The README.md is copied to users' `evals/` folders on `boba init` and serves as documentation for AI tools helping users write eval configs.
