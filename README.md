# simboba

[![PyPI](https://img.shields.io/pypi/v/simboba)](https://pypi.org/project/simboba/)

```
     ( )
   .-~~~-.
  /       \
  |  ===  |
  | ::::: |
  |_:::::_|
    '---'
```

Lightweight eval tracking with LLM-as-judge. Run evals as Python scripts, track results as git-friendly JSON files, view in a web UI. Designed for 1-click setup with your favourite AI coding tool.

## Installation

```bash
pip install simboba
```

## Quick Start

```bash
boba init          # Create boba-evals/ folder with templates
boba magic         # Prompt for your AI tool to set up and run your first eval
boba run           # Run your evals (handles Docker automatically)
boba baseline      # Save run as baseline for regression detection
boba serve         # View results at http://localhost:8787
```

## Commands

| Command                       | Description                                                             |
| ----------------------------- | ----------------------------------------------------------------------- |
| `boba init`                   | Create `boba-evals/` folder with starter templates                      |
| `boba magic`                  | Print detailed prompt for AI coding assistant                           |
| `boba run [script]`           | Run eval script (default: `test_chat.py`). Handles Docker automatically |
| `boba baseline`               | Save a run as baseline for regression detection                         |
| `boba serve`                  | Start web UI to view results                                            |
| `boba datasets`               | List all datasets                                                       |
| `boba generate "description"` | Generate a dataset from a description                                   |
| `boba reset`                  | Clear run history (keeps datasets and baselines)                        |

## Writing Evals

Evals are Python scripts. Edit `boba-evals/test_chat.py`:

```python
from simboba import Boba
from setup import get_context, cleanup

boba = Boba()

def agent(message: str) -> str:
    """Call your agent and return its response."""
    ctx = get_context()
    response = requests.post(
        "http://localhost:8000/api/chat",
        json={"user_id": ctx["user_id"], "message": message},
    )
    return response.json()["response"]

if __name__ == "__main__":
    try:
        # Option 1: Single eval
        boba.eval(
            input="Hello",
            output=agent("Hello"),
            expected="Should greet the user",
        )

        # Option 2: Run against a dataset
        # boba.run(agent, dataset="my-dataset")

        print("Done! Run 'boba serve' to view results.")
    finally:
        cleanup()
```

## Metadata Checking

Metadata (citations, tool_calls, etc.) is always passed to the LLM judge when provided. For strict deterministic checks, add a `metadata_checker` function:

```python
# Mode 1: No metadata - LLM judges output only
boba.eval(input="Hello", output="Hi!", expected="Should greet")

# Mode 2: LLM evaluates output + metadata together
boba.eval(
    input="What's my order status?",
    output="Your order #123 is shipped.",
    expected="Should look up order status",
    expected_metadata={"tool_calls": ["get_orders"]},
    actual_metadata={"tool_calls": ["get_orders"]},
)

# Mode 3: LLM evaluates + deterministic check (both must pass)
def check_tool_calls(expected, actual):
    expected_tools = set(expected.get("tool_calls", []))
    actual_tools = set(actual.get("tool_calls", []))
    return expected_tools == actual_tools

boba.eval(
    input="What's my order status?",
    output="Your order #123 is shipped.",
    expected="Should look up order status",
    expected_metadata={"tool_calls": ["get_orders"]},
    actual_metadata={"tool_calls": ["get_orders"]},
    metadata_checker=check_tool_calls,  # Additional deterministic gate
)
```

When using `metadata_checker`:
- LLM still sees metadata for context/reasoning
- Your function runs as an additional gate
- Case passes only if **both** LLM judgment and metadata check pass
- Results include `metadata_passed` field for visibility

## Regression Detection

Track regressions across code changes:

```bash
# Run evals and compare to baseline
boba run
# Output shows regressions: "REGRESSIONS: 2 cases now failing"

# Save current results as new baseline
boba baseline
# Commit to git for tracking
git add boba-evals/baselines/
git commit -m "Update eval baseline"
```

## Creating Datasets

### Via CLI

```bash
boba generate "A customer support chatbot for an e-commerce site"
```

### Via Web UI

1. `boba serve`
2. Click "New Dataset" -> "Generate with AI"
3. Enter a description of your agent and we'll create test cases for you.

### Via API

```python
from simboba import Boba
boba = Boba()
boba.run(agent, dataset="my-dataset")  # Uses dataset created above
```

## Test Fixtures (setup.py)

Edit `boba-evals/setup.py` to create test data your agent needs:

```python
def get_context():
    """Create test fixtures, return context dict."""
    user = create_test_user(email="eval@test.com")
    return {
        "user_id": user.id,
        "api_token": user.generate_token(),
    }

def cleanup():
    """Clean up test data after evals."""
    delete_test_users()
```

## Environment Variables

Boba loads `.env` automatically. Set your LLM API key for judging (Claude Haiku 4.5 is the default):

```bash
ANTHROPIC_API_KEY=sk-ant-...   # Required for default model (Claude)
OPENAI_API_KEY=sk-...          # For OpenAI models
GEMINI_API_KEY=...             # For Gemini models
```

> **Note:** Without an API key, boba falls back to a simple keyword-matching judge which is less accurate.

## Project Structure

```
your-project/
├── boba-evals/
│   ├── datasets/           # Dataset JSON files (git tracked)
│   ├── baselines/          # Baseline results (git tracked)
│   ├── runs/               # Run history (gitignored)
│   ├── files/              # Uploaded attachments
│   ├── setup.py            # Test fixtures
│   ├── test_chat.py        # Your eval script
│   ├── settings.json       # Configuration
│   └── .boba.yaml          # Runtime config (docker vs local)
└── ...
```

## Future Updates

- **File Uploads** - Allow uploads via UI to help create datasets
- **Eval methods** - Built-in evaluation strategies beyond LLM-as-judge
- **Cloud storage** - Sync datasets and runs to the cloud for team collaboration

## Development

To work on the web UI:

```bash
cd frontend
npm install
npm run dev      # Dev server with hot reload (proxies to localhost:8787)
npm run build    # Build to simboba/static/
```

Run `boba serve` in another terminal to start the backend.

## License

MIT
