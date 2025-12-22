"""FastAPI server for simboba."""

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from simboba import storage
from simboba.utils import LLMClient
from simboba.prompts import build_dataset_generation_prompt

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

    # Serve static assets (JS, CSS, images)
    if (STATIC_DIR / "assets").exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    # Serve favicon
    @app.get("/favicon.svg")
    def favicon():
        favicon_path = STATIC_DIR / "favicon.svg"
        if favicon_path.exists():
            return FileResponse(favicon_path, media_type="image/svg+xml")
        raise HTTPException(status_code=404, detail="Not found")

    # SPA fallback - serve index.html for non-API routes (React Router support)
    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        """Serve index.html for SPA client-side routing."""
        # Skip API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Not found")

    return app


# Create the app instance
app = create_app()
