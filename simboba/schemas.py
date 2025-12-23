"""Pydantic schemas for simboba data models."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# --- Message/Input Models ---

class MessageInput(BaseModel):
    """A single message in a conversation."""
    role: str
    message: str
    attachments: list[dict] = Field(default_factory=list)
    metadata: Optional[dict] = None  # For tool_calls, citations, etc.
    created_at: Optional[str] = None


# --- Case Models ---

class CaseCreate(BaseModel):
    """Request model for creating a case."""
    name: Optional[str] = None
    inputs: list[MessageInput]
    expected_outcome: str
    expected_metadata: Optional[dict] = None  # Expected citations, tool_calls, etc.


class CaseUpdate(BaseModel):
    """Request model for updating a case."""
    name: Optional[str] = None
    inputs: Optional[list[MessageInput]] = None
    expected_outcome: Optional[str] = None
    expected_metadata: Optional[dict] = None


class Case(BaseModel):
    """A single eval case."""
    id: str
    name: Optional[str] = None
    inputs: list[dict] = Field(default_factory=list)
    expected_outcome: str = ""
    expected_metadata: Optional[dict] = None
    created_at: str
    updated_at: str
    dataset_name: Optional[str] = None


# --- Dataset Models ---

class DatasetCreate(BaseModel):
    """Request model for creating a dataset."""
    name: str
    description: Optional[str] = None


class DatasetUpdate(BaseModel):
    """Request model for updating a dataset."""
    name: Optional[str] = None
    description: Optional[str] = None


class DatasetImport(BaseModel):
    """Request model for importing a dataset."""
    name: str
    description: Optional[str] = None
    cases: list[dict]


class Dataset(BaseModel):
    """A dataset containing multiple eval cases."""
    id: str  # Stable UUID for referencing runs/baselines
    name: str
    description: Optional[str] = None
    cases: list[dict] = Field(default_factory=list)
    created_at: str
    updated_at: str
    case_count: int = 0


# --- Run/Result Models ---

class ResultCreate(BaseModel):
    """A single result from evaluating a case."""
    case_id: str
    passed: bool
    actual_output: Optional[str] = None
    judgment: Optional[str] = None
    reasoning: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    expected_metadata: Optional[dict] = None
    actual_metadata: Optional[dict] = None
    metadata_passed: Optional[bool] = None


class Result(BaseModel):
    """A single result with case data."""
    case_id: str
    passed: bool
    actual_output: Optional[str] = None
    judgment: Optional[str] = None
    reasoning: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    expected_metadata: Optional[dict] = None
    actual_metadata: Optional[dict] = None
    metadata_passed: Optional[bool] = None  # None if no checker, True/False if checked
    created_at: str
    case: Optional[dict] = None


class Run(BaseModel):
    """An evaluation run against a dataset."""
    dataset_id: str  # UUID of the dataset
    dataset_name: Optional[str] = None  # For display purposes
    filename: str
    eval_name: str
    status: str = "pending"  # pending, running, completed, failed
    passed: int = 0
    failed: int = 0
    total: int = 0
    score: Optional[float] = None
    error_message: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None
    results: dict[str, dict] = Field(default_factory=dict)  # case_id -> result


class RunSummary(BaseModel):
    """Summary of a run for listing."""
    dataset_id: str  # UUID of the dataset
    dataset_name: Optional[str] = None  # For display purposes
    filename: str
    eval_name: str
    status: str
    passed: int
    failed: int
    total: int
    score: Optional[float] = None
    started_at: str
    completed_at: Optional[str] = None


# --- Baseline Models ---

class Baseline(BaseModel):
    """A baseline snapshot of run results."""
    dataset_id: str  # UUID of the dataset
    dataset_name: Optional[str] = None  # For display purposes
    saved_at: str
    source_run: str  # filename of the run this baseline was created from
    passed: int
    failed: int
    total: int
    score: Optional[float] = None
    results: dict[str, dict] = Field(default_factory=dict)  # case_id -> result


# --- Generation Models ---

class GenerateDatasetRequest(BaseModel):
    """Request model for generating a complete dataset."""
    product_description: str


class GenerateRequest(BaseModel):
    """Request model for generating cases."""
    dataset_name: str
    agent_description: str
    num_cases: int = 5
    complexity: str = "mixed"


class AcceptCasesRequest(BaseModel):
    """Request model for accepting generated cases."""
    dataset_name: str
    cases: list[dict]


# --- Bulk Operations ---

class BulkCreateCases(BaseModel):
    """Request model for bulk creating cases."""
    dataset_name: str
    cases: list[dict]


# --- Comparison/Regression Models ---

class RegressionReport(BaseModel):
    """Report comparing a run to its baseline."""
    regressions: list[str] = Field(default_factory=list)  # case IDs that regressed
    fixes: list[str] = Field(default_factory=list)  # case IDs that were fixed
    new_cases: list[str] = Field(default_factory=list)  # new case IDs
    removed_cases: list[str] = Field(default_factory=list)  # removed case IDs
    has_baseline: bool = False
