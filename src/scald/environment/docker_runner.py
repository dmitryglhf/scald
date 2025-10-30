import json
import os
from pathlib import Path

import docker
from docker.errors import BuildError, ContainerError, DockerException, ImageNotFound

from scald.common.logger import get_logger, get_session_dir
from scald.common.types import ActorSolution, TaskType

logger = get_logger()


class DockerRunner:
    def __init__(self, image_name: str = "scald-actor:latest", rebuild: bool = False):
        self.image_name = image_name
        self.rebuild = rebuild
        try:
            self.client = docker.from_env()
            logger.debug("Docker client initialized")
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise

    def build_image(self, dockerfile_path: Path, context_path: Path) -> None:
        logger.info(f"Building Docker image: {self.image_name}")
        try:
            self.client.images.build(
                path=str(context_path),
                dockerfile=str(dockerfile_path),
                tag=self.image_name,
                rm=True,
                pull=True,
            )
            logger.info(f"Built image: {self.image_name}")
        except BuildError as e:
            logger.error(f"Failed to build Docker image: {e}")
            raise

    def ensure_image_exists(self) -> None:
        project_root = Path(__file__).parent.parent.parent.parent

        if self.rebuild:
            logger.info("Rebuild flag set, rebuilding image...")
            try:
                self.client.images.remove(self.image_name, force=True)
            except Exception:
                pass
            self.build_image(
                dockerfile_path=project_root / "Dockerfile.actor",
                context_path=project_root,
            )
            return

        try:
            self.client.images.get(self.image_name)
            logger.debug(f"Image {self.image_name} exists")
        except ImageNotFound:
            logger.warning("Image not found, building...")
            self.build_image(
                dockerfile_path=project_root / "Dockerfile.actor",
                context_path=project_root,
            )

    def _prepare_directories(
        self, train_path: Path, test_path: Path
    ) -> tuple[Path, Path, Path, Path]:
        train_path = train_path.resolve()
        test_path = test_path.resolve()
        session_dir = get_session_dir().resolve()

        output_dir = (session_dir / "actor_output").resolve()
        output_dir.mkdir(exist_ok=True)

        logs_dir = (session_dir / "actor_logs").resolve()
        logs_dir.mkdir(exist_ok=True)

        return train_path, test_path, output_dir, logs_dir

    def _prepare_volumes(
        self, train_path: Path, test_path: Path, output_dir: Path, logs_dir: Path
    ) -> tuple[dict, str, str]:
        if train_path.parent == test_path.parent:
            volumes = {
                str(train_path.parent): {"bind": "/data", "mode": "ro"},
                str(output_dir): {"bind": "/output", "mode": "rw"},
                str(logs_dir): {"bind": "/app/scald_logs", "mode": "rw"},
            }
            train_docker_path = f"/data/{train_path.name}"
            test_docker_path = f"/data/{test_path.name}"
        else:
            volumes = {
                str(train_path.parent): {"bind": "/data/train", "mode": "ro"},
                str(test_path.parent): {"bind": "/data/test", "mode": "ro"},
                str(output_dir): {"bind": "/output", "mode": "rw"},
                str(logs_dir): {"bind": "/app/scald_logs", "mode": "rw"},
            }
            train_docker_path = f"/data/train/{train_path.name}"
            test_docker_path = f"/data/test/{test_path.name}"

        return volumes, train_docker_path, test_docker_path

    def _prepare_environment(
        self,
        train_docker_path: str,
        test_docker_path: str,
        target: str,
        task_type: TaskType,
        feedback: str | None,
        memory_context: Optional[list[ActorMemoryContext]],
    ) -> dict:
        environment = {
            "TRAIN_PATH": train_docker_path,
            "TEST_PATH": test_docker_path,
            "TARGET": target,
            "TASK_TYPE": task_type.value,
            "OUTPUT_DIR": "/output",
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
        }

        if feedback:
            environment["FEEDBACK"] = feedback

        if memory_context:
            try:
                # Serialize Pydantic models to dict
                memory_dicts = [mem.model_dump() for mem in memory_context]
                environment["MEMORY_CONTEXT"] = json.dumps(memory_dicts)
                logger.debug(f"Serialized {len(memory_context)} memory entries to ENV")
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to serialize memory context to JSON: {e}")

        return environment

    def _stream_container_logs(self, container) -> None:
        import sys

        for line in container.logs(stream=True, follow=True):
            log_line = line.decode("utf-8").strip()
            if log_line:
                print(f"[Actor] {log_line}", file=sys.stderr)
                logger.opt(depth=1).debug(f"[Actor] {log_line}")

    def _read_solution(self, output_dir: Path) -> ActorSolution:
        solution_file = output_dir / "solution.json"
        if not solution_file.exists():
            raise RuntimeError("Actor did not produce solution.json")

        with open(solution_file) as f:
            solution_data = json.load(f)

        return ActorSolution(**solution_data)

    def run_actor(
        self,
        train_path: Path,
        test_path: Path,
        target: str,
        task_type: TaskType,
        feedback: str | None = None,
        memory_context: list[dict[str, Any]] | None = None,
    ) -> ActorSolution:
        self.ensure_image_exists()

        train_path, test_path, output_dir, logs_dir = self._prepare_directories(
            train_path, test_path
        )

        volumes, train_docker_path, test_docker_path = self._prepare_volumes(
            train_path, test_path, output_dir, logs_dir
        )

        environment = self._prepare_environment(
            train_docker_path, test_docker_path, target, task_type, feedback, memory_context
        )

        logger.info(f"Running Actor in Docker container for {task_type.value} task")
        logger.debug(f"Train: {train_path}, Test: {test_path}, Target: {target}")

        try:
            container = self.client.containers.run(
                image=self.image_name,
                volumes=volumes,
                environment=environment,
                detach=True,
                remove=False,
                network_mode="bridge",
            )

            self._stream_container_logs(container)

            result = container.wait()
            exit_code = result["StatusCode"]
            container.remove()

            if exit_code != 0:
                logger.error(f"Actor container exited with code {exit_code}")
                raise RuntimeError(f"Actor failed with exit code {exit_code}")

            solution = self._read_solution(output_dir)
            logger.info("Actor completed successfully in Docker container")
            return solution

        except ContainerError as e:
            logger.error(f"Container execution failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to run Actor in Docker: {e}")
            raise


def run_actor_in_docker(
    train_path: Path,
    test_path: Path,
    target: str,
    task_type: TaskType,
    feedback: str | None = None,
    memory_context: list[dict[str, Any]] | None = None,
) -> ActorSolution:
    runner = DockerRunner()
    return runner.run_actor(train_path, test_path, target, task_type, feedback, memory_context)
