"""
Setup: Shared test fixtures for all evals.

PURPOSE:
    Create test data that your agent needs to run (users, projects, API tokens, etc.)
    This is imported by your eval scripts so you don't duplicate setup code.

USAGE:
    from setup import get_context, cleanup

    ctx = get_context()  # Get test fixtures
    # ... run evals ...
    cleanup()  # Optional: clean up after

--------------------------------------------------------------------------------
INSTRUCTIONS FOR AI ASSISTANTS (Claude, Cursor, Copilot, etc.)
--------------------------------------------------------------------------------
When modifying this file:

1. Look at the user's app to understand what test data is needed:
   - What models exist? (User, Project, Team, etc.)
   - What relationships between them?
   - What auth is needed? (API tokens, session, etc.)

2. Update get_context() to create the minimum test data needed:
   - Create a test user
   - Create any related objects (projects, teams, etc.)
   - Generate auth tokens if needed
   - Return a dict with IDs and tokens

3. Update cleanup() to delete test data:
   - Delete in reverse order of creation (handle foreign keys)
   - This is optional but recommended to avoid test data accumulation

4. The returned context dict keys should match what the agent function needs.
   Example: if agent() needs user_id and api_token, return those keys.
--------------------------------------------------------------------------------
"""

# TODO: Import your app's models or API client
# from myapp.models import User, Project
# from myapp.api import create_user, delete_user


# Cache context so we don't recreate on every call
_ctx = None


def get_context() -> dict:
    """
    Create test fixtures and return context dict.

    Called once per eval run. Returns a dict that gets passed to your agent.

    Returns:
        dict with keys your agent needs, e.g.:
            {
                "user_id": 123,
                "project_id": 456,
                "api_token": "xxx",
            }
    """
    global _ctx
    if _ctx is not None:
        return _ctx

    # -------------------------------------------------------------------------
    # CREATE TEST USER
    # -------------------------------------------------------------------------
    # user = create_user(
    #     email="eval-test@example.com",
    #     name="Eval Test User",
    # )

    # -------------------------------------------------------------------------
    # CREATE RELATED DATA (projects, teams, etc.)
    # -------------------------------------------------------------------------
    # project = create_project(
    #     user_id=user.id,
    #     name="Eval Test Project",
    # )

    # -------------------------------------------------------------------------
    # GENERATE AUTH TOKEN
    # -------------------------------------------------------------------------
    # api_token = user.generate_token()

    # -------------------------------------------------------------------------
    # RETURN CONTEXT
    # -------------------------------------------------------------------------
    _ctx = {
        # "user_id": user.id,
        # "project_id": project.id,
        # "api_token": api_token,
        "_placeholder": "Edit setup.py to create your test fixtures",
    }
    return _ctx


def cleanup():
    """
    Clean up test fixtures after eval run.

    Optional but recommended to avoid accumulating test data.
    Delete in reverse order of creation to handle foreign key constraints.
    """
    global _ctx
    if _ctx is None:
        return

    # -------------------------------------------------------------------------
    # DELETE TEST DATA
    # -------------------------------------------------------------------------
    # delete_project(_ctx["project_id"])
    # delete_user(_ctx["user_id"])

    _ctx = None
