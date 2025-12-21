# simboba

Lightweight eval tracking with LLM-as-judge. Users write Python scripts to run evals, results are tracked in SQLite and viewable in a web UI.

## Quick Start

```bash
pip install -e .
boba init                    # Create evals/ folder
python evals/test_chat.py    # Run evals
boba serve                   # View results
```

## Project Structure

```
simboba/
├── simboba/
│   ├── __init__.py       # Exports Boba class
│   ├── boba.py           # Core Boba class (eval, run methods)
│   ├── cli.py            # Click CLI (init, setup, serve, datasets, generate, reset)
│   ├── config.py         # Configuration (.boba.yaml) and Docker exec
│   ├── database.py       # SQLite/SQLAlchemy setup
│   ├── models.py         # Dataset, EvalCase, EvalRun, EvalResult, Settings
│   ├── server.py         # FastAPI REST API
│   ├── judge.py          # LLM judge implementation
│   ├── prompts/          # LLM prompts for generation and judging
│   ├── utils/            # LLM client utilities
│   ├── samples/          # Template files copied by `boba init`
│   │   ├── setup.py      # Test fixtures template
│   │   └── test_chat.py  # Eval script template
│   └── static/           # Web UI (index.html, app.js)
├── tests/
│   ├── conftest.py       # Pytest fixtures
│   └── test_core_flows.py
└── pyproject.toml
```

## Architecture

### Core Class: Boba

```python
from simboba import Boba

boba = Boba()

# Single eval - judges output against expected
result = boba.eval(
    input="Hello",
    output="Hi there!",
    expected="Should greet the user",
)
# Returns: {"passed": bool, "reasoning": str, "run_id": int}

# Dataset eval - runs agent against all cases in a dataset
result = boba.run(
    agent=my_agent_fn,  # Callable[[str], str]
    dataset="my-dataset",
)
# Returns: {"passed": int, "failed": int, "total": int, "score": float, "run_id": int}
```

### Data Model

- **Dataset**: Named collection of eval cases
- **EvalCase**: Test case with inputs, expected outcome, optional source reference
- **EvalRun**: One execution (from `boba.eval()` or `boba.run()`)
- **EvalResult**: Per-case result with actual output, judgment, reasoning

### EvalCase Structure

```python
{
    "inputs": [
        {"role": "user", "message": "...", "attachments": [{"file": "doc.pdf"}]}
    ],
    "expected_outcome": "Agent should...",
    "expected_source": {        # optional - for verification
        "file": "doc.pdf",
        "page": 12,
        "excerpt": "..."
    }
}
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `boba init` | Create `evals/` folder with templates |
| `boba init --docker` | Quick setup for Docker Compose |
| `boba init --local` | Quick setup for local Python |
| `boba setup` | Print AI prompt for configuring eval scripts |
| `boba serve` | Start web UI at localhost:8787 |
| `boba datasets` | List all datasets |
| `boba generate "desc"` | Generate dataset from description |
| `boba reset` | Delete database |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/datasets` | GET, POST | List/create datasets |
| `/api/datasets/{id}` | GET, DELETE | Get/delete dataset |
| `/api/datasets/{id}/export` | GET | Export dataset as JSON |
| `/api/datasets/import` | POST | Import dataset from JSON |
| `/api/cases` | GET, POST | List/create cases |
| `/api/cases/{id}` | GET, PUT, DELETE | CRUD for single case |
| `/api/cases/bulk` | POST | Bulk create cases |
| `/api/generate` | POST | Generate synthetic cases |
| `/api/generate/with-files` | POST | Generate from PDF files |
| `/api/generate/accept` | POST | Accept generated cases |
| `/api/runs` | GET | List runs |
| `/api/runs/{id}` | GET, DELETE | Get/delete run |
| `/api/settings` | GET, PUT | Get/update settings |

Note: Runs are created by the Boba class in Python scripts, not via API.

### Docker Integration

When configured for Docker (`boba init --docker`), commands auto-exec into the container:

```yaml
# evals/.boba.yaml
runtime: docker-compose
service: api
```

Set `BOBA_NO_DOCKER=1` to bypass.

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Test Coverage

- `TestBoba`: `eval()`, `run()` methods
- `TestDatasetManagement`: Dataset CRUD
- `TestCaseManagement`: Case CRUD
- `TestRunsAPI`: List/delete runs
- `TestJudge`: Simple keyword judge
- `TestUIServing`: Health, index endpoints

### Key Files for Common Changes

| Task | Files |
|------|-------|
| Change Boba API | `boba.py`, `__init__.py` |
| Add CLI command | `cli.py` |
| Add API endpoint | `server.py` |
| Change data model | `models.py`, `database.py` |
| Update templates | `samples/setup.py`, `samples/test_chat.py` |
| Change judging | `judge.py`, `prompts/judge.py` |
| Update UI | `static/index.html`, `static/app.js` |

### Adding New Features

1. Update model in `models.py` if needed
2. Add API endpoint in `server.py`
3. Update `boba.py` if it affects the Python API
4. Update UI in `static/app.js`
5. Add tests in `test_core_flows.py`
6. Update README.md and CLAUDE.md

## Design System

### Colors (Tailwind Zinc + Taro accent)

| Token | Usage |
|-------|-------|
| `--zinc-50` | Page background |
| `--zinc-200` | Borders |
| `--zinc-900` | Primary text |
| `--taro` (#8B7BA5) | Primary accent |
| `--green-500` | Pass states |
| `--red-500` | Fail states |

### Principles

- Sharp corners (max 4px border-radius)
- Minimal - zinc grays, single taro accent
- Monospace for data (scores, counts, timestamps)

## Judge Configuration

The LLM judge uses Claude by default. Set `ANTHROPIC_API_KEY` environment variable.

```python
# Judge function signature
def judge(inputs, expected, actual) -> tuple[bool, str]:
    """Returns (passed, reasoning)"""
```

A simple keyword-matching fallback (`create_simple_judge()`) is used when no API key is available.
