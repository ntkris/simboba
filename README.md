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

Lightweight eval tracking with LLM-as-judge. Run evals as Python scripts, track results in a web UI.

## Installation

```bash
pip install simboba
```

## Quick Start

```bash
boba init          # Create boba-evals/ folder with templates
boba magic         # Print AI prompt to help configure your evals
boba run           # Run your evals (handles Docker automatically)
boba serve         # View results at http://localhost:8787
```

## Commands

| Command | Description |
|---------|-------------|
| `boba init` | Create `boba-evals/` folder with starter templates |
| `boba magic` | Print detailed AI prompt to configure your eval scripts |
| `boba setup` | Print basic setup instructions |
| `boba run [script]` | Run eval script (default: `test_chat.py`). Handles Docker automatically |
| `boba serve` | Start web UI to view results |
| `boba datasets` | List all datasets |
| `boba generate "description"` | Generate a dataset from a description |
| `boba reset` | Delete database |

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

## Creating Datasets

### Via CLI
```bash
boba generate "A customer support chatbot for an e-commerce site"
```

### Via Web UI
1. `boba serve`
2. Click "New Dataset" → "Generate with AI"
3. Enter a description of your agent

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

Boba loads `.env` automatically. Set your LLM API key for judging (Claude 3.5 Haiku is the default):

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
│   ├── setup.py        # Test fixtures
│   ├── test_chat.py    # Your eval script
│   ├── .boba.yaml      # Config (docker vs local)
│   └── simboba.db      # Results database
└── ...
```

## License

MIT
