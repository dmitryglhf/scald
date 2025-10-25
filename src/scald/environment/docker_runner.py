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

    def run_actor(
        self,
        csv_path: Path,
        target: str,
        task_type: TaskType,
        feedback: str | None = None,
    ) -> ActorSolution:
        self.ensure_image_exists()

        # Prepare paths (must be absolute for Docker)
        csv_path = csv_path.resolve()
        session_dir = get_session_dir().resolve()
        output_dir = (session_dir / "actor_output").resolve()
        output_dir.mkdir(exist_ok=True)

        logs_dir = (session_dir / "actor_logs").resolve()
        logs_dir.mkdir(exist_ok=True)

        # Prepare volumes (Docker requires absolute paths)
        volumes = {
            str(csv_path.parent.resolve()): {"bind": "/data", "mode": "ro"},
            str(output_dir): {"bind": "/output", "mode": "rw"},
            str(logs_dir): {"bind": "/app/scald_logs", "mode": "rw"},
        }

        # Prepare environment
        environment = {
            "CSV_PATH": f"/data/{csv_path.name}",
            "TARGET": target,
            "TASK_TYPE": task_type.value,
            "OUTPUT_DIR": "/output",
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
        }

        if feedback:
            environment["FEEDBACK"] = feedback

        logger.info(f"Running Actor in Docker container for {task_type.value} task")
        logger.debug(f"CSV: {csv_path}, Target: {target}")

        try:
            # Run container
            container = self.client.containers.run(
                image=self.image_name,
                volumes=volumes,
                environment=environment,
                detach=True,
                remove=False,
                network_mode="bridge",
            )

            # Stream logs in real-time
            import sys

            for line in container.logs(stream=True, follow=True):
                log_line = line.decode("utf-8").strip()
                if log_line:
                    # Print to console without timestamp
                    print(f"[Actor] {log_line}", file=sys.stderr)
                    # Log to file with full context
                    logger.opt(depth=1).debug(f"[Actor] {log_line}")

            # Wait for completion
            result = container.wait()
            exit_code = result["StatusCode"]

            # Remove container
            container.remove()

            if exit_code != 0:
                logger.error(f"Actor container exited with code {exit_code}")
                raise RuntimeError(f"Actor failed with exit code {exit_code}")

            # Read results from output directory
            solution_file = output_dir / "solution.json"
            if not solution_file.exists():
                raise RuntimeError("Actor did not produce solution.json")

            with open(solution_file) as f:
                solution_data = json.load(f)

            logger.info("Actor completed successfully in Docker container")
            return ActorSolution(**solution_data)

        except ContainerError as e:
            logger.error(f"Container execution failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to run Actor in Docker: {e}")
            raise


def run_actor_in_docker(
    csv_path: Path,
    target: str,
    task_type: TaskType,
    feedback: str | None = None,
) -> ActorSolution:
    runner = DockerRunner()
    return runner.run_actor(csv_path, target, task_type, feedback)
