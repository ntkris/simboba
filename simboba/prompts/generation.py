"""Prompts for test case generation."""

DATASET_GENERATION_PROMPT = """You are an expert at creating eval datasets for AI agents.

Given a product description, generate a complete eval dataset with a name, description, and test cases.

Product Description:
{product_description}

Generate a JSON object with:
1. A short, kebab-case dataset name (e.g., "customer-support-bot", "doc-qa-agent")
2. A brief description of what the dataset tests
3. 5-10 diverse test cases covering different scenarios

Output format:
```json
{{
  "name": "dataset-name",
  "description": "Brief description of what this dataset evaluates",
  "cases": [
    {{
      "name": "Single-turn example",
      "inputs": [
        {{"role": "user", "message": "User's request or question", "attachments": []}}
      ],
      "expected_outcome": "What the agent should do or respond with"
    }},
    {{
      "name": "Multi-turn conversation example",
      "inputs": [
        {{"role": "user", "message": "User's initial request", "attachments": []}},
        {{"role": "assistant", "message": "Agent's first response"}},
        {{"role": "user", "message": "User's follow-up question or clarification", "attachments": []}},
        {{"role": "assistant", "message": "Agent's second response"}},
        {{"role": "user", "message": "User's final input", "attachments": []}}
      ],
      "expected_outcome": "What the agent should do in its final response"
    }}
  ]
}}
```

## Guidelines

### Conversation Structure
- Include a mix of single-turn AND multi-turn conversations
- Multi-turn cases should have realistic back-and-forth dialogue
- The last message in inputs should always be from "user" (what the agent needs to respond to)
- Cover happy paths, edge cases, and error handling

### Writing Expected Outcomes

Expected outcomes are evaluated by an LLM judge. Write them as specific, testable criteria.

**For behavioral evals** (testing tone, format, reasoning):
- Describe the behavior, not exact wording
- Good: "Should acknowledge the frustration empathetically and offer to help resolve the issue"
- Bad: "Responds nicely" (too vague)

**For factual evals** (testing knowledge retrieval):
- Include the ground truth directly in the expected outcome
- The conversation should contain the information needed to answer
- Good: "Should state that the return window is 30 days and requires original receipt"
- Bad: "Should explain the return policy" (judge can't verify correctness)

**Format:**
- Be specific about what MUST be included
- Mention constraints if relevant (e.g., "should NOT reveal internal pricing")
- Use action verbs: "Should explain...", "Must include...", "Should ask for..."

Only output the JSON object, no other text."""


def build_dataset_generation_prompt(product_description: str) -> str:
    """Build a prompt to generate a complete dataset from a product description.

    Args:
        product_description: Description of the product/agent to test

    Returns:
        Formatted prompt string
    """
    return DATASET_GENERATION_PROMPT.format(product_description=product_description)
