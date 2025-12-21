"""SQLAlchemy models for datasets and eval cases."""

from datetime import datetime
from typing import Optional
from sqlalchemy import String, Text, ForeignKey, JSON, Float, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from simboba.database import Base


class Dataset(Base):
    """A dataset containing multiple eval cases."""

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, onupdate=datetime.utcnow
    )

    cases: Mapped[list["EvalCase"]] = relationship(
        "EvalCase", back_populates="dataset", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "case_count": len(self.cases) if self.cases else 0,
        }


class EvalCase(Base):
    """A single eval case with inputs and expected outcome."""

    __tablename__ = "eval_cases"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    inputs: Mapped[list] = mapped_column(JSON, default=list)
    expected_outcome: Mapped[str] = mapped_column(Text, default="")
    expected_source: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, onupdate=datetime.utcnow
    )

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="cases")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "inputs": self.inputs,
            "expected_outcome": self.expected_outcome,
            "expected_source": self.expected_source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class EvalRun(Base):
    """A single evaluation run against a dataset."""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[Optional[int]] = mapped_column(ForeignKey("datasets.id"), index=True, nullable=True)
    eval_name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), default="pending")  # pending, running, completed, failed
    passed: Mapped[int] = mapped_column(default=0)
    failed: Mapped[int] = mapped_column(default=0)
    total: Mapped[int] = mapped_column(default=0)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    dataset: Mapped[Optional["Dataset"]] = relationship("Dataset")
    results: Mapped[list["EvalResult"]] = relationship(
        "EvalResult", back_populates="run", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset.name if self.dataset else None,
            "eval_name": self.eval_name,
            "status": self.status,
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
            "score": self.score,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class Settings(Base):
    """Application settings stored as key-value pairs."""

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")

    # Default settings
    DEFAULTS = {
        "model": "gemini/gemini-3-flash-preview",  # Default model for generation and judging
    }

    @classmethod
    def get(cls, db, key: str) -> str:
        """Get a setting value, returning default if not set."""
        setting = db.query(cls).filter(cls.key == key).first()
        if setting:
            return setting.value
        return cls.DEFAULTS.get(key, "")

    @classmethod
    def set(cls, db, key: str, value: str):
        """Set a setting value."""
        setting = db.query(cls).filter(cls.key == key).first()
        if setting:
            setting.value = value
        else:
            setting = cls(key=key, value=value)
            db.add(setting)
        db.commit()

    @classmethod
    def get_all(cls, db) -> dict:
        """Get all settings as a dict, with defaults for unset keys."""
        result = dict(cls.DEFAULTS)
        for setting in db.query(cls).all():
            result[setting.key] = setting.value
        return result


class EvalResult(Base):
    """Result for a single eval case within a run."""

    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("eval_runs.id"), index=True)
    case_id: Mapped[Optional[int]] = mapped_column(ForeignKey("eval_cases.id"), index=True, nullable=True)
    # For ad-hoc runs (when case_id is null), store inputs/expected inline
    inputs: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    expected_outcome: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    actual_output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    judgment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    execution_time_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    run: Mapped["EvalRun"] = relationship("EvalRun", back_populates="results")
    case: Mapped[Optional["EvalCase"]] = relationship("EvalCase")

    def to_dict(self) -> dict:
        # For ad-hoc results, build case data from inline fields
        if self.case:
            case_data = self.case.to_dict()
        elif self.inputs is not None:
            case_data = {
                "id": None,
                "inputs": self.inputs,
                "expected_outcome": self.expected_outcome or "",
            }
        else:
            case_data = None

        return {
            "id": self.id,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "passed": self.passed,
            "actual_output": self.actual_output,
            "judgment": self.judgment,
            "reasoning": self.reasoning,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "created_at": self.created_at.isoformat(),
            "case": case_data,
        }
