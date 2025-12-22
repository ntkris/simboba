"""Tests for core simboba flows."""


class TestUIServing:
    """Test that the UI is served correctly."""

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_index_serves_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "simboba" in response.text.lower()


class TestDatasetManagement:
    """Test dataset CRUD operations."""

    def test_create_dataset(self, client):
        response = client.post(
            "/api/datasets",
            json={"name": "my-dataset", "description": "Test dataset"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "my-dataset"
        assert data["description"] == "Test dataset"
        assert data["case_count"] == 0
        assert "id" in data  # Should have a generated UUID

    def test_list_datasets(self, client):
        # Create two datasets
        client.post("/api/datasets", json={"name": "dataset-1"})
        client.post("/api/datasets", json={"name": "dataset-2"})

        response = client.get("/api/datasets")
        assert response.status_code == 200
        datasets = response.json()
        assert len(datasets) == 2

    def test_get_dataset(self, client):
        client.post("/api/datasets", json={"name": "test"})

        response = client.get("/api/datasets/test")
        assert response.status_code == 200
        assert response.json()["name"] == "test"

    def test_delete_dataset(self, client):
        client.post("/api/datasets", json={"name": "to-delete"})

        response = client.delete("/api/datasets/to-delete")
        assert response.status_code == 200

        # Verify it's gone
        get_resp = client.get("/api/datasets/to-delete")
        assert get_resp.status_code == 404

    def test_duplicate_name_rejected(self, client):
        client.post("/api/datasets", json={"name": "unique-name"})
        response = client.post("/api/datasets", json={"name": "unique-name"})
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]


class TestCaseManagement:
    """Test eval case CRUD operations."""

    def test_create_case(self, client):
        # First create a dataset
        client.post("/api/datasets", json={"name": "test-ds"})

        # Create a case
        response = client.post(
            "/api/cases",
            json={
                "dataset_name": "test-ds",
                "name": "Basic test",
                "inputs": [
                    {"role": "user", "message": "Hello", "attachments": []},
                    {"role": "assistant", "message": "Hi there", "attachments": []},
                ],
                "expected_outcome": "Agent greets the user politely",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Basic test"
        assert len(data["inputs"]) == 2
        assert data["expected_outcome"] == "Agent greets the user politely"
        assert "id" in data  # Should have a generated ID

    def test_list_cases_by_dataset(self, client):
        # Create dataset and cases
        client.post("/api/datasets", json={"name": "ds"})

        for i in range(3):
            client.post(
                "/api/cases",
                json={
                    "dataset_name": "ds",
                    "name": f"Case {i}",
                    "inputs": [{"role": "user", "message": "test", "attachments": []}],
                    "expected_outcome": "test",
                },
            )

        response = client.get("/api/cases?dataset_name=ds")
        assert response.status_code == 200
        assert len(response.json()) == 3

    def test_update_case(self, client):
        client.post("/api/datasets", json={"name": "ds"})

        case_resp = client.post(
            "/api/cases",
            json={
                "dataset_name": "ds",
                "name": "Original",
                "inputs": [{"role": "user", "message": "Hi", "attachments": []}],
                "expected_outcome": "Original outcome",
            },
        )
        case_id = case_resp.json()["id"]

        response = client.put(
            f"/api/cases/ds/{case_id}",
            json={"name": "Updated", "expected_outcome": "Updated outcome"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "Updated"
        assert response.json()["expected_outcome"] == "Updated outcome"

    def test_delete_case(self, client):
        client.post("/api/datasets", json={"name": "ds"})

        case_resp = client.post(
            "/api/cases",
            json={
                "dataset_name": "ds",
                "inputs": [{"role": "user", "message": "Hi", "attachments": []}],
                "expected_outcome": "test",
            },
        )
        case_id = case_resp.json()["id"]

        response = client.delete(f"/api/cases/ds/{case_id}")
        assert response.status_code == 200

        get_resp = client.get(f"/api/cases/ds/{case_id}")
        assert get_resp.status_code == 404


class TestExportImport:
    """Test dataset export and import."""

    def test_export_dataset(self, client):
        client.post("/api/datasets", json={"name": "export-test"})

        client.post(
            "/api/cases",
            json={
                "dataset_name": "export-test",
                "name": "Case 1",
                "inputs": [{"role": "user", "message": "Hello", "attachments": []}],
                "expected_outcome": "Greet back",
            },
        )

        response = client.get("/api/datasets/export-test/export")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "export-test"
        assert len(data["cases"]) == 1

    def test_import_dataset(self, client):
        response = client.post(
            "/api/datasets/import",
            json={
                "name": "imported-dataset",
                "description": "Imported from JSON",
                "cases": [
                    {
                        "name": "Imported case",
                        "inputs": [
                            {"role": "user", "message": "Test", "attachments": []}
                        ],
                        "expected_outcome": "Test outcome",
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "imported-dataset"
        assert data["case_count"] == 1


class TestBoba:
    """Test the Boba class for running evaluations."""

    def test_eval_single(self, client, evals_dir, monkeypatch):
        """Test single eval with Boba class."""
        from simboba import Boba, storage

        monkeypatch.setattr(storage, "get_evals_dir", lambda: evals_dir)

        boba = Boba()
        result = boba.eval(
            input="Hello",
            output="Hi there! How can I help you?",
            expected="Should greet the user politely",
        )

        assert "passed" in result
        assert "reasoning" in result
        assert "run_id" in result
        assert result["run_id"] is not None

    def test_eval_with_name(self, client, evals_dir, monkeypatch):
        """Test single eval with custom name."""
        from simboba import Boba, storage

        monkeypatch.setattr(storage, "get_evals_dir", lambda: evals_dir)

        boba = Boba()
        result = boba.eval(
            input="What's 2+2?",
            output="4",
            expected="Should return 4",
            name="math-test",
        )

        assert "passed" in result
        assert "run_id" in result

    def test_run_against_dataset(self, client, evals_dir, monkeypatch):
        """Test running an agent against a dataset."""
        from simboba import Boba, storage

        monkeypatch.setattr(storage, "get_evals_dir", lambda: evals_dir)

        # Create a dataset with cases via API
        client.post(
            "/api/datasets",
            json={"name": "test-run-dataset", "description": "For testing boba.run()"},
        )

        # Add test cases
        test_cases = [
            {"message": "Hello", "expected": "Should greet back"},
            {"message": "How are you?", "expected": "Should respond politely"},
            {"message": "Goodbye", "expected": "Should say farewell"},
        ]

        for tc in test_cases:
            client.post(
                "/api/cases",
                json={
                    "dataset_name": "test-run-dataset",
                    "inputs": [{"role": "user", "message": tc["message"], "attachments": []}],
                    "expected_outcome": tc["expected"],
                },
            )

        # Define a simple agent function
        def echo_agent(message: str) -> str:
            return f"You said: {message}. Hello! I'm doing well, goodbye!"

        # Run the agent against the dataset
        boba = Boba()
        result = boba.run(agent=echo_agent, dataset="test-run-dataset")

        # Verify results
        assert result["total"] == 3
        assert result["passed"] + result["failed"] == 3
        assert "score" in result
        assert "run_id" in result
        assert result["run_id"] is not None


class TestJudge:
    """Test the judge module."""

    def test_simple_judge_pass(self):
        from simboba.judge import create_simple_judge

        judge = create_simple_judge()
        inputs = [{"role": "user", "message": "Book appointment"}]
        expected = "Agent should book appointment"
        actual = "I have booked your appointment for tomorrow"

        passed, reasoning = judge(inputs, expected, actual)
        assert passed is True
        assert "expected terms" in reasoning.lower()

    def test_simple_judge_fail(self):
        from simboba.judge import create_simple_judge

        judge = create_simple_judge()
        inputs = [{"role": "user", "message": "Book appointment"}]
        expected = "Agent should book appointment and confirm time"
        actual = "Hello there!"

        passed, reasoning = judge(inputs, expected, actual)
        assert passed is False


class TestRunsAPI:
    """Test the eval run API endpoints."""

    def test_list_runs_empty(self, client):
        # Create dataset and get its ID
        ds_resp = client.post("/api/datasets", json={"name": "ds"})
        dataset_id = ds_resp.json()["id"]

        # Use dataset_id for filtering runs
        response = client.get(f"/api/runs?dataset_id={dataset_id}")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_and_delete_run(self, client, evals_dir, monkeypatch):
        """Test that runs created by Boba can be viewed and deleted via API."""
        from simboba import Boba, storage

        monkeypatch.setattr(storage, "get_evals_dir", lambda: evals_dir)

        # Create a run using the Boba class
        boba = Boba()
        result = boba.eval(
            input="Test input",
            output="Test output",
            expected="Should work",
        )
        run_id = result["run_id"]

        # Get run details via API (ad-hoc evals use "_adhoc" as dataset_id)
        detail_resp = client.get(f"/api/runs/_adhoc/{run_id}")
        assert detail_resp.status_code == 200
        details = detail_resp.json()
        assert details["status"] == "completed"
        assert details["dataset_id"] == "_adhoc"

        # List all runs
        list_resp = client.get("/api/runs")
        assert list_resp.status_code == 200
        runs = list_resp.json()
        assert len(runs) >= 1
        assert runs[0]["dataset_id"] == "_adhoc"

        # Delete run
        del_resp = client.delete(f"/api/runs/_adhoc/{run_id}")
        assert del_resp.status_code == 200

        # Verify it's gone
        get_resp = client.get(f"/api/runs/_adhoc/{run_id}")
        assert get_resp.status_code == 404


class TestBaselines:
    """Test baseline API endpoints."""

    def test_list_baselines_empty(self, client):
        response = client.get("/api/baselines")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_baseline_not_found(self, client):
        response = client.get("/api/baselines/nonexistent")
        assert response.status_code == 404


class TestSettings:
    """Test settings API endpoints."""

    def test_get_settings(self, client):
        response = client.get("/api/settings")
        assert response.status_code == 200
        settings = response.json()
        assert "model" in settings

    def test_update_settings(self, client):
        response = client.put(
            "/api/settings",
            json={"model": "gpt-4"},
        )
        assert response.status_code == 200
        assert response.json()["model"] == "gpt-4"

        # Verify it persisted
        get_resp = client.get("/api/settings")
        assert get_resp.json()["model"] == "gpt-4"
