"""FastAPI server for simboba."""

import json
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Query, File, UploadFile, Form

logger = logging.getLogger(__name__)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from simboba.database import init_db, get_session_factory
from simboba.models import Dataset, EvalCase, EvalRun, EvalResult, Settings
from simboba.utils.models import LLMClient
from simboba.prompts import (
    build_dataset_generation_prompt,
    build_generation_prompt,
    build_generation_prompt_with_files,
)

STATIC_DIR = Path(__file__).parent / "static"


# --- Database Dependency ---

def get_db():
    """Get a database session."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Request/Response Models ---

class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class DatasetImport(BaseModel):
    name: str
    description: Optional[str] = None
    cases: list[dict]


class MessageInput(BaseModel):
    role: str
    message: str
    attachments: list = []
    created_at: Optional[str] = None


class ExpectedSource(BaseModel):
    file: str
    page: int
    excerpt: Optional[str] = None  # optional snippet for quick reference


class CaseCreate(BaseModel):
    dataset_id: int
    name: Optional[str] = None
    inputs: list[MessageInput]
    expected_outcome: str
    expected_source: Optional[ExpectedSource] = None


class CaseUpdate(BaseModel):
    name: Optional[str] = None
    inputs: Optional[list[MessageInput]] = None
    expected_outcome: Optional[str] = None
    expected_source: Optional[ExpectedSource] = None


class BulkCreateCases(BaseModel):
    dataset_id: int
    cases: list[dict]


class GenerateDatasetRequest(BaseModel):
    product_description: str


class GenerateRequest(BaseModel):
    dataset_id: int
    agent_description: str
    num_cases: int = 5
    complexity: str = "mixed"


class AcceptCasesRequest(BaseModel):
    dataset_id: int
    cases: list[dict]


# --- App Factory ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    init_db()
    yield


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Simboba",
        description="Eval dataset generation and LLM-as-judge evaluations",
        version="0.1.0",
        lifespan=lifespan,
    )

    # --- Health & UI Routes ---

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def index():
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return {"message": "Simboba API is running. Static files not found."}

    # --- Dataset Routes ---

    @app.get("/api/datasets")
    def list_datasets(db: Session = Depends(get_db)):
        datasets = db.query(Dataset).order_by(Dataset.updated_at.desc()).all()
        return [d.to_dict() for d in datasets]

    @app.post("/api/datasets")
    def create_dataset(data: DatasetCreate, db: Session = Depends(get_db)):
        existing = db.query(Dataset).filter(Dataset.name == data.name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Dataset with this name already exists")
        dataset = Dataset(name=data.name, description=data.description)
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        return dataset.to_dict()

    @app.get("/api/datasets/{dataset_id}")
    def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return dataset.to_dict()

    @app.put("/api/datasets/{dataset_id}")
    def update_dataset(dataset_id: int, data: DatasetUpdate, db: Session = Depends(get_db)):
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if data.name is not None:
            existing = db.query(Dataset).filter(Dataset.name == data.name, Dataset.id != dataset_id).first()
            if existing:
                raise HTTPException(status_code=400, detail="Dataset with this name already exists")
            dataset.name = data.name
        if data.description is not None:
            dataset.description = data.description
        db.commit()
        db.refresh(dataset)
        return dataset.to_dict()

    @app.delete("/api/datasets/{dataset_id}")
    def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        db.delete(dataset)
        db.commit()
        return {"message": "Dataset deleted"}

    @app.get("/api/datasets/{dataset_id}/export")
    def export_dataset(dataset_id: int, db: Session = Depends(get_db)):
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {
            "name": dataset.name,
            "description": dataset.description,
            "cases": [case.to_dict() for case in dataset.cases],
        }

    @app.post("/api/datasets/import")
    def import_dataset(data: DatasetImport, db: Session = Depends(get_db)):
        existing = db.query(Dataset).filter(Dataset.name == data.name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Dataset with this name already exists")
        dataset = Dataset(name=data.name, description=data.description)
        db.add(dataset)
        db.flush()
        for case_data in data.cases:
            case = EvalCase(
                dataset_id=dataset.id,
                name=case_data.get("name"),
                inputs=case_data.get("inputs", []),
                expected_outcome=case_data.get("expected_outcome", ""),
            )
            db.add(case)
        db.commit()
        db.refresh(dataset)
        return dataset.to_dict()

    @app.post("/api/datasets/generate")
    def generate_dataset(data: GenerateDatasetRequest, db: Session = Depends(get_db)):
        """Generate a complete dataset from a product description."""
        import traceback
        try:
            prompt = build_dataset_generation_prompt(data.product_description)
            model = Settings.get(db, "model")
            print(f"[generate_dataset] Using model: {model}")

            client = LLMClient(model=model)
            response = client.generate(prompt)
            result = client.parse_json_response(response)
        except Exception as e:
            print(f"[generate_dataset] ERROR: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

        # Validate required fields
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Invalid response format")
        if not result.get("name"):
            raise HTTPException(status_code=500, detail="Generated dataset missing name")
        if not result.get("cases"):
            raise HTTPException(status_code=500, detail="Generated dataset has no cases")

        # Check for duplicate name
        name = result["name"]
        existing = db.query(Dataset).filter(Dataset.name == name).first()
        if existing:
            # Append a number to make it unique
            i = 1
            while db.query(Dataset).filter(Dataset.name == f"{name}-{i}").first():
                i += 1
            name = f"{name}-{i}"

        # Create the dataset
        dataset = Dataset(name=name, description=result.get("description", ""))
        db.add(dataset)
        db.flush()

        # Create the cases
        for case_data in result["cases"]:
            case = EvalCase(
                dataset_id=dataset.id,
                name=case_data.get("name"),
                inputs=case_data.get("inputs", []),
                expected_outcome=case_data.get("expected_outcome", ""),
                expected_source=case_data.get("expected_source"),
            )
            db.add(case)

        db.commit()
        db.refresh(dataset)
        return dataset.to_dict()

    # --- Case Routes ---

    @app.get("/api/cases")
    def list_cases(dataset_id: Optional[int] = Query(None), db: Session = Depends(get_db)):
        query = db.query(EvalCase)
        if dataset_id is not None:
            query = query.filter(EvalCase.dataset_id == dataset_id)
        cases = query.order_by(EvalCase.created_at.desc()).all()
        return [c.to_dict() for c in cases]

    @app.post("/api/cases")
    def create_case(data: CaseCreate, db: Session = Depends(get_db)):
        dataset = db.query(Dataset).filter(Dataset.id == data.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        inputs = [msg.model_dump() for msg in data.inputs]
        case = EvalCase(
            dataset_id=data.dataset_id,
            name=data.name,
            inputs=inputs,
            expected_outcome=data.expected_outcome,
            expected_source=data.expected_source.model_dump() if data.expected_source else None,
        )
        db.add(case)
        db.commit()
        db.refresh(case)
        return case.to_dict()

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: int, db: Session = Depends(get_db)):
        case = db.query(EvalCase).filter(EvalCase.id == case_id).first()
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        return case.to_dict()

    @app.put("/api/cases/{case_id}")
    def update_case(case_id: int, data: CaseUpdate, db: Session = Depends(get_db)):
        case = db.query(EvalCase).filter(EvalCase.id == case_id).first()
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        if data.name is not None:
            case.name = data.name
        if data.inputs is not None:
            case.inputs = [msg.model_dump() for msg in data.inputs]
        if data.expected_outcome is not None:
            case.expected_outcome = data.expected_outcome
        if data.expected_source is not None:
            case.expected_source = data.expected_source.model_dump()
        db.commit()
        db.refresh(case)
        return case.to_dict()

    @app.delete("/api/cases/{case_id}")
    def delete_case(case_id: int, db: Session = Depends(get_db)):
        case = db.query(EvalCase).filter(EvalCase.id == case_id).first()
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        db.delete(case)
        db.commit()
        return {"message": "Case deleted"}

    @app.post("/api/cases/bulk")
    def bulk_create_cases(data: BulkCreateCases, db: Session = Depends(get_db)):
        dataset = db.query(Dataset).filter(Dataset.id == data.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        created = []
        for case_data in data.cases:
            case = EvalCase(
                dataset_id=data.dataset_id,
                name=case_data.get("name"),
                inputs=case_data.get("inputs", []),
                expected_outcome=case_data.get("expected_outcome", ""),
            )
            db.add(case)
            created.append(case)
        db.commit()
        for case in created:
            db.refresh(case)
        return [c.to_dict() for c in created]

    # --- Generation Routes ---

    @app.post("/api/generate")
    def generate_cases(data: GenerateRequest, db: Session = Depends(get_db)):
        dataset = db.query(Dataset).filter(Dataset.id == data.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        prompt = build_generation_prompt(data.agent_description, data.num_cases, data.complexity)
        model = Settings.get(db, "model")
        try:
            client = LLMClient(model=model)
            response = client.generate(prompt)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        cases = _parse_generated_cases(response)

        return {
            "cases": cases,
            "message": f"Generated {len(cases)} cases. Review and accept the ones you want to add."
        }

    @app.post("/api/generate/with-files")
    async def generate_cases_with_files(
        dataset_id: int = Form(...),
        agent_description: str = Form(...),
        num_cases: int = Form(5),
        complexity: str = Form("mixed"),
        files: list[UploadFile] = File(...),
        db: Session = Depends(get_db)
    ):
        """Generate test cases using uploaded files as reference material."""
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        if not files:
            raise HTTPException(status_code=400, detail="At least one file is required")

        # Extract text from all uploaded files
        files_data = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Only PDF files are supported. Got: {file.filename}"
                )
            content = await file.read()
            pages = _extract_pdf_text(content, file.filename)
            files_data.append({
                "filename": file.filename,
                "pages": pages
            })

        # Build prompt with file contents
        file_contents = _format_file_contents(files_data)
        prompt = build_generation_prompt_with_files(
            agent_description, num_cases, complexity, file_contents
        )

        model = Settings.get(db, "model")
        try:
            client = LLMClient(model=model)
            response = client.generate(prompt)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        cases = _parse_generated_cases(response)

        return {
            "cases": cases,
            "files": [f["filename"] for f in files_data],
            "message": f"Generated {len(cases)} cases from {len(files_data)} file(s). Review and accept the ones you want to add."
        }

    @app.post("/api/generate/accept")
    def accept_cases(data: AcceptCasesRequest, db: Session = Depends(get_db)):
        dataset = db.query(Dataset).filter(Dataset.id == data.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        created = []
        for case_data in data.cases:
            # Handle expected_source if present
            expected_source = case_data.get("expected_source")
            case = EvalCase(
                dataset_id=data.dataset_id,
                name=case_data.get("name"),
                inputs=case_data.get("inputs", []),
                expected_outcome=case_data.get("expected_outcome", ""),
                expected_source=expected_source,
            )
            db.add(case)
            created.append(case)
        db.commit()
        for case in created:
            db.refresh(case)
        return {
            "cases": [c.to_dict() for c in created],
            "message": f"Added {len(created)} cases to dataset '{dataset.name}'"
        }

    # --- Eval Run Routes ---

    @app.get("/api/runs")
    def list_runs(dataset_id: Optional[int] = Query(None), db: Session = Depends(get_db)):
        """List eval runs, optionally filtered by dataset."""
        query = db.query(EvalRun)
        if dataset_id is not None:
            query = query.filter(EvalRun.dataset_id == dataset_id)
        runs = query.order_by(EvalRun.started_at.desc()).all()
        return [r.to_dict() for r in runs]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: int, db: Session = Depends(get_db)):
        """Get a specific eval run with results."""
        run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        run_dict = run.to_dict()
        run_dict["results"] = [r.to_dict() for r in run.results]
        return run_dict

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: int, db: Session = Depends(get_db)):
        """Delete an eval run."""
        run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        db.delete(run)
        db.commit()
        return {"message": "Run deleted"}

    # --- Settings Routes ---

    @app.get("/api/settings")
    def get_settings(db: Session = Depends(get_db)):
        """Get all settings."""
        return Settings.get_all(db)

    @app.put("/api/settings")
    def update_settings(updates: dict, db: Session = Depends(get_db)):
        """Update settings."""
        for key, value in updates.items():
            Settings.set(db, key, value)
        return Settings.get_all(db)

    # --- Playground Routes ---

    class PlaygroundQuery(BaseModel):
        query: str

    class PlaygroundSQL(BaseModel):
        sql: str

    @app.post("/api/playground/sql")
    def playground_sql(data: PlaygroundSQL, db: Session = Depends(get_db)):
        """Execute a SQL query directly (for hardcoded example queries)."""
        from sqlalchemy import text

        sql = data.sql.strip()

        # Safety check: only allow SELECT queries
        sql_upper = sql.upper().strip()
        if not sql_upper.startswith("SELECT"):
            return {
                "success": False,
                "error": "Only SELECT queries are allowed",
                "sql": sql,
                "results": []
            }

        try:
            result = db.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]

            return {
                "success": True,
                "sql": sql,
                "columns": columns,
                "results": rows
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "sql": sql,
                "results": []
            }

    @app.post("/api/playground/query")
    def playground_query(data: PlaygroundQuery, db: Session = Depends(get_db)):
        """Execute a natural language query against the database."""
        from sqlalchemy import text

        # Get the model from settings
        model = Settings.get(db, "model")

        # Build the prompt for SQL generation
        schema_description = """
Database Schema:

TABLE datasets (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    description TEXT,
    created_at DATETIME,
    updated_at DATETIME
)

TABLE eval_cases (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id),
    name VARCHAR(255),
    inputs JSON,  -- List of {role, message, attachments}
    expected_outcome TEXT,
    expected_source JSON,  -- {file, page, excerpt}
    created_at DATETIME,
    updated_at DATETIME
)

TABLE eval_runs (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id),
    eval_name VARCHAR(255),
    status VARCHAR(50),  -- pending, running, completed, failed
    passed INTEGER,
    failed INTEGER,
    total INTEGER,
    score FLOAT,  -- percentage 0-100
    error_message TEXT,
    started_at DATETIME,
    completed_at DATETIME
)

TABLE eval_results (
    id INTEGER PRIMARY KEY,
    run_id INTEGER REFERENCES eval_runs(id),
    case_id INTEGER REFERENCES eval_cases(id),
    passed BOOLEAN,
    actual_output TEXT,
    judgment TEXT,
    reasoning TEXT,
    error_message TEXT,
    execution_time_ms INTEGER,
    created_at DATETIME
)

TABLE settings (
    key VARCHAR(255) PRIMARY KEY,
    value TEXT
)
"""

        prompt = f"""You are a SQL query generator. Convert the user's natural language query into a SQLite SELECT query.

{schema_description}

Rules:
1. ONLY generate SELECT queries. Never generate INSERT, UPDATE, DELETE, DROP, or any other modifying queries.
2. Return ONLY the SQL query, no explanations.
3. Use proper SQLite syntax.
4. Limit results to 100 rows maximum unless the user specifies otherwise.
5. For recent/latest queries, order by appropriate datetime columns DESC.

User query: {data.query}

SQL query:"""

        try:
            client = LLMClient(model=model)
            response = client.generate(prompt)
            sql = response.strip()

            # Clean up the SQL (remove markdown code blocks if present)
            if sql.startswith("```sql"):
                sql = sql[6:]
            elif sql.startswith("```"):
                sql = sql[3:]
            if sql.endswith("```"):
                sql = sql[:-3]
            sql = sql.strip()

            # Safety check: only allow SELECT queries
            sql_upper = sql.upper().strip()
            if not sql_upper.startswith("SELECT"):
                return {
                    "success": False,
                    "error": "Only SELECT queries are allowed",
                    "sql": sql,
                    "results": []
                }

            # Execute the query
            result = db.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]

            return {
                "success": True,
                "sql": sql,
                "columns": columns,
                "results": rows
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "sql": sql if 'sql' in dir() else None,
                "results": []
            }

    # Serve static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


# --- Generation Helpers ---

def _extract_pdf_text(file_content: bytes, filename: str) -> list[dict]:
    """Extract text from PDF, returning list of {page, text} dicts."""
    try:
        from pypdf import PdfReader
        import io
    except ImportError:
        raise HTTPException(status_code=500, detail="pypdf package not installed")

    try:
        reader = PdfReader(io.BytesIO(file_content))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i, "text": text.strip()})
        return pages
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF '{filename}': {e}")


def _format_file_contents(files_data: list[dict]) -> str:
    """Format extracted file contents for the prompt."""
    parts = []
    for file_data in files_data:
        filename = file_data["filename"]
        pages = file_data["pages"]
        parts.append(f"=== {filename} ===")
        for page_data in pages:
            parts.append(f"--- Page {page_data['page']} ---")
            parts.append(page_data["text"])
            parts.append("")
    return "\n".join(parts)


def _parse_generated_cases(response: str) -> list[dict]:
    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        cases = json.loads(text)
        if not isinstance(cases, list):
            raise ValueError("Response is not a list")
        return cases
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse generated cases: {e}")


# Create the app instance
app = create_app()
