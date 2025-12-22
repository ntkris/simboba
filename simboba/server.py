"""FastAPI server for simboba."""

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from simboba import storage
from simboba.utils import LLMClient
from simboba.prompts import (
    build_dataset_generation_prompt,
    build_generation_prompt,
    build_generation_prompt_with_files,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


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
    excerpt: Optional[str] = None


class CaseCreate(BaseModel):
    dataset_name: str
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
    dataset_name: str
    cases: list[dict]


class GenerateDatasetRequest(BaseModel):
    product_description: str


class GenerateRequest(BaseModel):
    dataset_name: str
    agent_description: str
    num_cases: int = 5
    complexity: str = "mixed"


class AcceptCasesRequest(BaseModel):
    dataset_name: str
    cases: list[dict]


# --- App Factory ---

def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Simboba",
        description="Eval dataset generation and LLM-as-judge evaluations",
        version="0.2.0",
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
    # Datasets can be looked up by either name or UUID (id)

    def _get_dataset_by_name_or_id(identifier: str) -> Optional[dict]:
        """Look up a dataset by name first, then by UUID."""
        # Try by name first
        dataset = storage.get_dataset(identifier)
        if dataset:
            return dataset
        # Try by UUID
        return storage.get_dataset_by_id(identifier)

    @app.get("/api/datasets")
    def list_datasets():
        datasets = storage.list_datasets()
        return datasets

    @app.post("/api/datasets")
    def create_dataset(data: DatasetCreate):
        if storage.dataset_exists(data.name):
            raise HTTPException(status_code=400, detail="Dataset with this name already exists")
        dataset = {
            "name": data.name,
            "description": data.description,
            "cases": [],
        }
        result = storage.save_dataset(dataset)
        logger.info(f"Created dataset '{data.name}' with id={result.get('id')}")
        return result

    @app.get("/api/datasets/{identifier}")
    def get_dataset(identifier: str):
        """Get a dataset by name or UUID."""
        dataset = _get_dataset_by_name_or_id(identifier)
        if not dataset:
            logger.warning(f"Dataset not found: {identifier}")
            raise HTTPException(status_code=404, detail="Dataset not found")
        return dataset

    @app.put("/api/datasets/{identifier}")
    def update_dataset(identifier: str, data: DatasetUpdate):
        """Update a dataset by name or UUID."""
        dataset = _get_dataset_by_name_or_id(identifier)
        if not dataset:
            logger.warning(f"Dataset not found for update: {identifier}")
            raise HTTPException(status_code=404, detail="Dataset not found")

        current_name = dataset["name"]

        # Handle rename using the rename_dataset function (preserves UUID)
        if data.name is not None and data.name != current_name:
            try:
                dataset = storage.rename_dataset(current_name, data.name)
                if not dataset:
                    raise HTTPException(status_code=404, detail="Dataset not found")
                logger.info(f"Renamed dataset '{current_name}' to '{data.name}'")
            except ValueError as e:
                logger.error(f"Failed to rename dataset: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        if data.description is not None:
            dataset["description"] = data.description
            dataset = storage.save_dataset(dataset)

        return dataset

    @app.delete("/api/datasets/{identifier}")
    def delete_dataset(identifier: str):
        """Delete a dataset by name or UUID."""
        dataset = _get_dataset_by_name_or_id(identifier)
        if not dataset:
            logger.warning(f"Dataset not found for delete: {identifier}")
            raise HTTPException(status_code=404, detail="Dataset not found")

        if not storage.delete_dataset(dataset["name"]):
            raise HTTPException(status_code=404, detail="Dataset not found")
        logger.info(f"Deleted dataset '{dataset['name']}' (id={dataset.get('id')})")
        return {"message": "Dataset deleted"}

    @app.get("/api/datasets/{identifier}/export")
    def export_dataset(identifier: str):
        """Export a dataset by name or UUID."""
        dataset = _get_dataset_by_name_or_id(identifier)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {
            "name": dataset["name"],
            "description": dataset.get("description"),
            "cases": dataset.get("cases", []),
        }

    @app.post("/api/datasets/import")
    def import_dataset(data: DatasetImport):
        if storage.dataset_exists(data.name):
            raise HTTPException(status_code=400, detail="Dataset with this name already exists")
        dataset = {
            "name": data.name,
            "description": data.description,
            "cases": data.cases,
        }
        return storage.save_dataset(dataset)

    @app.post("/api/datasets/generate")
    def generate_dataset(data: GenerateDatasetRequest):
        """Generate a complete dataset from a product description."""
        import traceback
        try:
            prompt = build_dataset_generation_prompt(data.product_description)
            model = storage.get_setting("model")
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
        if storage.dataset_exists(name):
            i = 1
            while storage.dataset_exists(f"{name}-{i}"):
                i += 1
            name = f"{name}-{i}"

        # Create the dataset
        dataset = {
            "name": name,
            "description": result.get("description", ""),
            "cases": result.get("cases", []),
        }
        return storage.save_dataset(dataset)

    # --- Case Routes ---
    # Cases use dataset_id (UUID) for lookup, with fallback to name

    @app.get("/api/cases")
    def list_cases(
        dataset_name: Optional[str] = Query(None),
        dataset_id: Optional[str] = Query(None)
    ):
        """List cases, optionally filtered by dataset name or ID."""
        identifier = dataset_id or dataset_name
        if identifier:
            dataset = _get_dataset_by_name_or_id(identifier)
            if not dataset:
                raise HTTPException(status_code=404, detail="Dataset not found")
            cases = dataset.get("cases", [])
            for case in cases:
                case["dataset_name"] = dataset["name"]
                case["dataset_id"] = dataset["id"]
            return cases
        else:
            # Return all cases from all datasets
            all_cases = []
            for dataset in storage.list_datasets():
                for case in dataset.get("cases", []):
                    case["dataset_name"] = dataset["name"]
                    case["dataset_id"] = dataset["id"]
                    all_cases.append(case)
            return all_cases

    @app.post("/api/cases")
    def create_case(data: CaseCreate):
        dataset = _get_dataset_by_name_or_id(data.dataset_name)
        if not dataset:
            logger.warning(f"Dataset not found for case creation: {data.dataset_name}")
            raise HTTPException(status_code=404, detail="Dataset not found")

        case = {
            "name": data.name,
            "inputs": [msg.model_dump() for msg in data.inputs],
            "expected_outcome": data.expected_outcome,
            "expected_source": data.expected_source.model_dump() if data.expected_source else None,
        }
        result = storage.add_case(dataset["name"], case)
        logger.info(f"Created case '{data.name}' in dataset '{dataset['name']}'")
        return result

    @app.get("/api/cases/{dataset_identifier}/{case_id}")
    def get_case(dataset_identifier: str, case_id: str):
        """Get a case by dataset name/ID and case ID."""
        dataset = _get_dataset_by_name_or_id(dataset_identifier)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        case = storage.get_case(dataset["name"], case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        return case

    @app.put("/api/cases/{dataset_identifier}/{case_id}")
    def update_case(dataset_identifier: str, case_id: str, data: CaseUpdate):
        """Update a case by dataset name/ID and case ID."""
        dataset = _get_dataset_by_name_or_id(dataset_identifier)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        updates = {}
        if data.name is not None:
            updates["name"] = data.name
        if data.inputs is not None:
            updates["inputs"] = [msg.model_dump() for msg in data.inputs]
        if data.expected_outcome is not None:
            updates["expected_outcome"] = data.expected_outcome
        if data.expected_source is not None:
            updates["expected_source"] = data.expected_source.model_dump()

        case = storage.update_case(dataset["name"], case_id, updates)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        logger.info(f"Updated case '{case_id}' in dataset '{dataset['name']}'")
        return case

    @app.delete("/api/cases/{dataset_identifier}/{case_id}")
    def delete_case(dataset_identifier: str, case_id: str):
        """Delete a case by dataset name/ID and case ID."""
        dataset = _get_dataset_by_name_or_id(dataset_identifier)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if not storage.delete_case(dataset["name"], case_id):
            raise HTTPException(status_code=404, detail="Case not found")
        logger.info(f"Deleted case '{case_id}' from dataset '{dataset['name']}'")
        return {"message": "Case deleted"}

    @app.post("/api/cases/bulk")
    def bulk_create_cases(data: BulkCreateCases):
        dataset = _get_dataset_by_name_or_id(data.dataset_name)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        created = []
        for case_data in data.cases:
            case = storage.add_case(dataset["name"], case_data)
            created.append(case)
        logger.info(f"Bulk created {len(created)} cases in dataset '{dataset['name']}'")
        return created

    # --- Generation Routes ---

    @app.post("/api/generate")
    def generate_cases(data: GenerateRequest):
        dataset = _get_dataset_by_name_or_id(data.dataset_name)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        prompt = build_generation_prompt(data.agent_description, data.num_cases, data.complexity)
        model = storage.get_setting("model")
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
        dataset_name: str = Form(...),
        agent_description: str = Form(...),
        num_cases: int = Form(5),
        complexity: str = Form("mixed"),
        files: list[UploadFile] = File(...),
    ):
        """Generate test cases using uploaded files as reference material."""
        dataset = _get_dataset_by_name_or_id(dataset_name)
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

        model = storage.get_setting("model")
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
    def accept_cases(data: AcceptCasesRequest):
        dataset = _get_dataset_by_name_or_id(data.dataset_name)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        created = []
        for case_data in data.cases:
            case = storage.add_case(dataset["name"], case_data)
            created.append(case)

        return {
            "cases": created,
            "message": f"Added {len(created)} cases to dataset '{data.dataset_name}'"
        }

    # --- Eval Run Routes ---
    # Runs are stored by dataset_id (UUID), not name

    @app.get("/api/runs")
    def list_runs(dataset_id: Optional[str] = Query(None)):
        """List eval runs, optionally filtered by dataset ID."""
        runs = storage.list_runs(dataset_id)

        # Enrich with dataset name for display
        for run in runs:
            ds_id = run.get("dataset_id")
            if ds_id and ds_id != "_adhoc":
                ds = storage.get_dataset_by_id(ds_id)
                run["dataset_name"] = ds["name"] if ds else None

        # Return summary without full results
        return [
            {
                "dataset_id": r.get("dataset_id"),
                "dataset_name": r.get("dataset_name"),
                "filename": r.get("filename"),
                "eval_name": r.get("eval_name"),
                "status": r.get("status"),
                "passed": r.get("passed"),
                "failed": r.get("failed"),
                "total": r.get("total"),
                "score": r.get("score"),
                "started_at": r.get("started_at"),
                "completed_at": r.get("completed_at"),
            }
            for r in runs
        ]

    @app.get("/api/runs/{dataset_id}/{filename}")
    def get_run(dataset_id: str, filename: str):
        """Get a specific eval run with results."""
        run = storage.get_run(dataset_id, filename)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        # Enrich with dataset name
        if dataset_id != "_adhoc":
            ds = storage.get_dataset_by_id(dataset_id)
            run["dataset_name"] = ds["name"] if ds else None

        return run

    @app.delete("/api/runs/{dataset_id}/{filename}")
    def delete_run(dataset_id: str, filename: str):
        """Delete an eval run."""
        if not storage.delete_run(dataset_id, filename):
            raise HTTPException(status_code=404, detail="Run not found")
        return {"message": "Run deleted"}

    # --- Baseline Routes ---
    # Baselines are stored by dataset_id (UUID), not name

    @app.get("/api/baselines")
    def list_baselines():
        """List all baselines."""
        baselines = storage.list_baselines()

        # Enrich with dataset name for display
        for baseline in baselines:
            ds_id = baseline.get("dataset_id")
            if ds_id:
                ds = storage.get_dataset_by_id(ds_id)
                baseline["dataset_name"] = ds["name"] if ds else None

        return baselines

    @app.get("/api/baselines/{dataset_id}")
    def get_baseline(dataset_id: str):
        """Get the baseline for a dataset by ID."""
        baseline = storage.get_baseline(dataset_id)
        if not baseline:
            raise HTTPException(status_code=404, detail="Baseline not found")

        # Enrich with dataset name
        ds = storage.get_dataset_by_id(dataset_id)
        baseline["dataset_name"] = ds["name"] if ds else None

        return baseline

    # --- Settings Routes ---

    @app.get("/api/settings")
    def get_settings():
        """Get all settings."""
        return storage.get_settings()

    @app.put("/api/settings")
    def update_settings(updates: dict):
        """Update settings."""
        current = storage.get_settings()
        current.update(updates)
        return storage.save_settings(current)

    # --- File Upload Routes ---

    @app.post("/api/files/upload")
    async def upload_file(file: UploadFile = File(...)):
        """Upload a file for use in eval cases."""
        content = await file.read()
        filename = storage.save_file(file.filename, content)
        return {"filename": filename, "message": f"Uploaded {filename}"}

    @app.get("/api/files/{filename}")
    def get_file(filename: str):
        """Get a file."""
        path = storage.get_file_path(filename)
        if not path:
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path)

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
