import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from docker.errors import BuildError, DockerException, ImageNotFound

from scald.common.types import TaskType
from scald.environment.docker_runner import DockerRunner, run_actor_in_docker


class TestDockerRunner:
    @patch("scald.environment.docker_runner.docker.from_env")
    def test_init_success(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.return_value = mock_client

        runner = DockerRunner(image_name="test-image:v1")

        assert runner.image_name == "test-image:v1"
        assert runner.client == mock_client
        mock_docker.assert_called_once()

    @patch("scald.environment.docker_runner.docker.from_env")
    def test_init_docker_exception(self, mock_docker):
        mock_docker.side_effect = DockerException("Docker not available")

        with pytest.raises(DockerException):
            DockerRunner()

    @patch("scald.environment.docker_runner.docker.from_env")
    def test_build_image(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.return_value = mock_client

        runner = DockerRunner()
        dockerfile = Path("/tmp/Dockerfile")
        context = Path("/tmp")

        runner.build_image(dockerfile, context)

        mock_client.images.build.assert_called_once_with(
            path=str(context),
            dockerfile=str(dockerfile),
            tag="scald-actor:latest",
            rm=True,
            pull=True,
        )

    @patch("scald.environment.docker_runner.docker.from_env")
    def test_build_image_failure(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.images.build.side_effect = BuildError("Build failed", "")

        runner = DockerRunner()
        with pytest.raises(BuildError):
            runner.build_image(Path("/tmp/Dockerfile"), Path("/tmp"))

    @patch("scald.environment.docker_runner.docker.from_env")
    def test_ensure_image_exists_when_present(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.images.get.return_value = MagicMock()

        runner = DockerRunner()
        runner.ensure_image_exists()

        mock_client.images.get.assert_called_once_with("scald-actor:latest")

    @patch("scald.environment.docker_runner.docker.from_env")
    def test_ensure_image_exists_when_missing(self, mock_docker):
        mock_client = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.images.get.side_effect = ImageNotFound("Image not found")

        runner = DockerRunner()
        with patch.object(runner, "build_image") as mock_build:
            runner.ensure_image_exists()
            mock_build.assert_called_once()

    @patch("scald.environment.docker_runner.docker.from_env")
    @patch("scald.environment.docker_runner.get_session_dir")
    def test_run_actor_volumes_setup(self, mock_session_dir, mock_docker):
        mock_session_dir.return_value = Path("/tmp/session")
        mock_client = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.images.get.return_value = MagicMock()

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b"Success"
        mock_client.containers.run.return_value = mock_container

        runner = DockerRunner()
        csv_path = Path("/tmp/data/test.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("a,b\n1,2")

        solution_file = Path("/tmp/session/actor_output/solution.json")
        solution_file.parent.mkdir(parents=True, exist_ok=True)
        solution_file.write_text(
            json.dumps({"predictions_path": "/output/pred.csv", "metrics": {"acc": 0.9}})
        )

        try:
            runner.run_actor(
                csv_path=csv_path, target="b", task_type=TaskType.CLASSIFICATION, feedback=None
            )

            call_kwargs = mock_client.containers.run.call_args[1]
            assert "volumes" in call_kwargs
            volumes = call_kwargs["volumes"]
            assert str(csv_path.parent) in volumes
            assert "/tmp/session/actor_output" in volumes
        finally:
            csv_path.unlink(missing_ok=True)
            solution_file.unlink(missing_ok=True)

    @patch("scald.environment.docker_runner.docker.from_env")
    @patch("scald.environment.docker_runner.get_session_dir")
    def test_run_actor_environment_variables(self, mock_session_dir, mock_docker):
        mock_session_dir.return_value = Path("/tmp/session")
        mock_client = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.images.get.return_value = MagicMock()

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b"Success"
        mock_client.containers.run.return_value = mock_container

        runner = DockerRunner()
        csv_path = Path("/tmp/data/test.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("a,b\n1,2")

        solution_file = Path("/tmp/session/actor_output/solution.json")
        solution_file.parent.mkdir(parents=True, exist_ok=True)
        solution_file.write_text(
            json.dumps({"predictions_path": "/output/pred.csv", "metrics": {"acc": 0.9}})
        )

        try:
            runner.run_actor(
                csv_path=csv_path,
                target="species",
                task_type=TaskType.CLASSIFICATION,
                feedback="improve accuracy",
            )

            call_kwargs = mock_client.containers.run.call_args[1]
            env = call_kwargs["environment"]

            assert env["CSV_PATH"] == f"/data/{csv_path.name}"
            assert env["TARGET"] == "species"
            assert env["TASK_TYPE"] == "classification"
            assert env["FEEDBACK"] == "improve accuracy"
            assert "OPENROUTER_API_KEY" in env
        finally:
            csv_path.unlink(missing_ok=True)
            solution_file.unlink(missing_ok=True)

    @patch("scald.environment.docker_runner.docker.from_env")
    @patch("scald.environment.docker_runner.get_session_dir")
    def test_run_actor_container_failure(self, mock_session_dir, mock_docker):
        mock_session_dir.return_value = Path("/tmp/session")
        mock_client = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.images.get.return_value = MagicMock()

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 1}
        mock_container.logs.return_value = b"Error occurred"
        mock_client.containers.run.return_value = mock_container

        runner = DockerRunner()
        csv_path = Path("/tmp/data/test.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("a,b\n1,2")

        try:
            with pytest.raises(RuntimeError, match="Actor failed with exit code 1"):
                runner.run_actor(
                    csv_path=csv_path, target="b", task_type=TaskType.CLASSIFICATION, feedback=None
                )
        finally:
            csv_path.unlink(missing_ok=True)

    @patch("scald.environment.docker_runner.docker.from_env")
    @patch("scald.environment.docker_runner.get_session_dir")
    def test_run_actor_missing_solution_file(self, mock_session_dir, mock_docker):
        session_path = Path("/tmp/session_missing")
        session_path.mkdir(parents=True, exist_ok=True)
        mock_session_dir.return_value = session_path

        mock_client = MagicMock()
        mock_docker.return_value = mock_client
        mock_client.images.get.return_value = MagicMock()

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b"Success"
        mock_client.containers.run.return_value = mock_container

        runner = DockerRunner()
        csv_path = Path("/tmp/data/test.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("a,b\n1,2")

        try:
            with pytest.raises(RuntimeError, match="did not produce solution.json"):
                runner.run_actor(
                    csv_path=csv_path, target="b", task_type=TaskType.CLASSIFICATION, feedback=None
                )
        finally:
            csv_path.unlink(missing_ok=True)
            import shutil

            shutil.rmtree(session_path, ignore_errors=True)


class TestRunActorInDocker:
    @patch("scald.environment.docker_runner.DockerRunner")
    def test_convenience_function(self, mock_runner_class):
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner
        mock_solution = Mock()
        mock_runner.run_actor.return_value = mock_solution

        result = run_actor_in_docker(
            csv_path=Path("/tmp/data.csv"),
            target="target",
            task_type=TaskType.CLASSIFICATION,
            feedback="test feedback",
        )

        mock_runner_class.assert_called_once()
        mock_runner.run_actor.assert_called_once()
        call_args = mock_runner.run_actor.call_args
        assert call_args[0][0] == Path("/tmp/data.csv")
        assert call_args[0][1] == "target"
        assert call_args[0][2] == TaskType.CLASSIFICATION
        assert call_args[0][3] == "test feedback"
        assert result == mock_solution
