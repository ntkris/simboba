"""
Eval: Test your agent's chat functionality.

PURPOSE:
    Run evaluations against your agent and track results.
    Modify this script to test different features of your agent.

USAGE:
    python evals/test_chat.py
    boba serve  # View results in UI

--------------------------------------------------------------------------------
INSTRUCTIONS FOR AI ASSISTANTS (Claude, Cursor, Copilot, etc.)
--------------------------------------------------------------------------------
When modifying this file:

1. UNDERSTAND THE USER'S AGENT:
   - How is it called? (HTTP API, Python function, webhook, etc.)
   - What input format does it expect?
   - What output format does it return?
   - What auth/context does it need?

2. UPDATE THE agent() FUNCTION:
   - Get context from setup.py using get_context()
   - Translate the message string to your app's expected input format
   - Call your agent (HTTP request, function call, webhook, etc.)
   - Translate the response back to a string
   - Handle errors gracefully

3. COMMON PATTERNS:

   HTTP API:
       def agent(message):
           ctx = get_context()
           response = requests.post(
               "http://localhost:8000/api/chat",
               headers={"Authorization": f"Bearer {ctx['api_token']}"},
               json={"user_id": ctx["user_id"], "message": message},
           )
           response.raise_for_status()
           return response.json()["response"]

   Python function:
       def agent(message):
           ctx = get_context()
           from myapp.agent import run
           return run(user_id=ctx["user_id"], message=message)

   Webhook (async):
       def agent(message):
           ctx = get_context()
           job = requests.post(url, json={"message": message})
           job_id = job.json()["job_id"]
           # Poll for result
           for _ in range(30):
               time.sleep(1)
               result = requests.get(f"{url}/jobs/{job_id}")
               if result.json()["status"] == "complete":
                   return result.json()["response"]
           raise TimeoutError("Agent did not respond")

4. UPDATE THE EVALS:
   - Add test cases that cover your agent's functionality
   - Use boba.eval() for individual cases
   - Use boba.run(agent, dataset="name") to run against a dataset

5. INPUT TRANSLATION:
   If your agent expects more than just a message string (e.g., attachments,
   conversation history), handle that in the agent() function:

       def agent(message):
           ctx = get_context()
           # Build the full input your agent expects
           full_input = {
               "user_id": ctx["user_id"],
               "conversation": [{"role": "user", "content": message}],
               "attachments": [],
           }
           return call_agent(full_input)
--------------------------------------------------------------------------------
"""

import requests
from simboba import Boba
from setup import get_context, cleanup


boba = Boba()


# =============================================================================
# AGENT FUNCTION
# =============================================================================

def agent(message: str) -> str:
    """
    Call your agent with the given message and return its response.

    Args:
        message: The user's message (string)

    Returns:
        The agent's response (string)

    This function handles:
        1. Getting test context (user_id, api_token, etc.) from setup.py
        2. Translating message → your app's input format
        3. Calling your agent
        4. Translating response → string
    """
    ctx = get_context()

    # -------------------------------------------------------------------------
    # OPTION 1: HTTP API
    # -------------------------------------------------------------------------
    # response = requests.post(
    #     "http://localhost:8000/api/chat",
    #     headers={"Authorization": f"Bearer {ctx['api_token']}"},
    #     json={
    #         "user_id": ctx["user_id"],
    #         "project_id": ctx["project_id"],
    #         "message": message,
    #     },
    # )
    # response.raise_for_status()
    # return response.json()["response"]

    # -------------------------------------------------------------------------
    # OPTION 2: Direct Python call
    # -------------------------------------------------------------------------
    # from myapp.agent import run
    # return run(
    #     user_id=ctx["user_id"],
    #     project_id=ctx["project_id"],
    #     message=message,
    # )

    # -------------------------------------------------------------------------
    # PLACEHOLDER - Replace with your agent call
    # -------------------------------------------------------------------------
    raise NotImplementedError(
        "Connect your agent here. See instructions at top of file."
    )


# =============================================================================
# RUN EVALS
# =============================================================================

if __name__ == "__main__":
    try:
        # ---------------------------------------------------------------------
        # OPTION 1: Single eval (quick test)
        # ---------------------------------------------------------------------
        boba.eval(
            input="Hello",
            output=agent("Hello"),
            expected="Should greet the user politely",
        )

        # ---------------------------------------------------------------------
        # OPTION 2: Multiple evals (inline test cases)
        # ---------------------------------------------------------------------
        # test_cases = [
        #     ("Hello", "Should greet the user"),
        #     ("What can you do?", "Should explain capabilities"),
        #     ("Goodbye", "Should say farewell"),
        # ]
        #
        # for input_msg, expected in test_cases:
        #     boba.eval(
        #         input=input_msg,
        #         output=agent(input_msg),
        #         expected=expected,
        #     )

        # ---------------------------------------------------------------------
        # OPTION 3: Run against a dataset
        # ---------------------------------------------------------------------
        # boba.run(agent, dataset="my-dataset")

        print("Eval complete! Run 'boba serve' to view results.")

    finally:
        cleanup()
