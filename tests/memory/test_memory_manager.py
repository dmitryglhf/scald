import tempfile
from pathlib import Path

import pytest

from scald.common.types import ActorSolution, CriticEvaluation, TaskType
from scald.memory import MemoryManager


@pytest.fixture
def temp_db_file():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmpfile:
        yield Path(tmpfile.name)
        if tmpfile.name and Path(tmpfile.name).exists():
            Path(tmpfile.name).unlink()


@pytest.fixture
def memory_manager(temp_db_file):
    memory = MemoryManager(persist_path=str(temp_db_file))
    yield memory
    memory.clear_all()


@pytest.fixture
def sample_actor_solution():
    return ActorSolution(
        predictions_path=None,
        predictions=[0, 1, 2, 1, 0],
        metrics={"accuracy": 0.92, "f1": 0.90},
        report="Used CatBoost with tuned hyperparameters: lr=0.1, depth=6.",
    )


@pytest.fixture
def sample_critic_evaluation():
    return CriticEvaluation(score=1, feedback="Model is well-tuned with good generalization.")


class TestMemoryManagerInitialization:
    def test_initialization(self, temp_db_file):
        memory = MemoryManager(persist_path=str(temp_db_file))

        assert memory.db is not None
        assert memory.actors is not None
        assert memory.critics is not None


class TestActorMemory:
    def test_save_actor_solution(self, memory_manager, sample_actor_solution):
        memory_id = memory_manager.save_actor_solution(
            solution=sample_actor_solution,
            task_type=TaskType.CLASSIFICATION,
            target="Species",
            iteration=1,
            accepted=True,
        )

        assert memory_id != ""
        assert len(memory_manager.actors.all()) == 1

        saved = memory_manager.actors.all()[0]
        assert saved["task_type"] == "classification"
        assert saved["target"] == "Species"
        assert saved["accepted"] is True

    def test_update_actor_solution_status(self, memory_manager, sample_actor_solution):
        # Save solution as not accepted
        memory_manager.save_actor_solution(
            solution=sample_actor_solution,
            task_type=TaskType.CLASSIFICATION,
            target="Species",
            iteration=1,
            accepted=False,
        )

        # Verify initial status
        saved = memory_manager.actors.all()[0]
        assert saved["accepted"] is False

        # Update to accepted
        success = memory_manager.update_actor_solution_status(
            task_type=TaskType.CLASSIFICATION,
            target="Species",
            iteration=1,
            accepted=True,
        )

        assert success is True

        # Verify updated status
        saved = memory_manager.actors.all()[0]
        assert saved["accepted"] is True

    def test_update_actor_solution_status_not_found(self, memory_manager):
        success = memory_manager.update_actor_solution_status(
            task_type=TaskType.CLASSIFICATION,
            target="NonExistent",
            iteration=999,
            accepted=True,
        )

        assert success is False

    def test_get_actor_context_empty(self, memory_manager):
        context = memory_manager.get_actor_context(
            task_type=TaskType.CLASSIFICATION, target="Species", limit=3
        )

        assert context == []

    def test_get_actor_context_with_memories(self, memory_manager, sample_actor_solution):
        memory_manager.save_actor_solution(
            solution=sample_actor_solution,
            task_type=TaskType.CLASSIFICATION,
            target="Species",
            iteration=1,
            accepted=True,
        )

        context = memory_manager.get_actor_context(
            task_type=TaskType.CLASSIFICATION, target="Species", limit=3
        )

        assert len(context) == 1
        assert context[0]["task_type"] == "classification"
        assert context[0]["target"] == "Species"
        assert context[0]["accepted"] is True

    def test_actor_context_filters_by_task_type(self, memory_manager, sample_actor_solution):
        memory_manager.save_actor_solution(
            solution=sample_actor_solution,
            task_type=TaskType.CLASSIFICATION,
            target="Species",
            iteration=1,
            accepted=True,
        )

        context = memory_manager.get_actor_context(
            task_type=TaskType.REGRESSION, target="Price", limit=3
        )

        assert len(context) == 0

    def test_actor_context_sandwich_pattern(self, memory_manager, sample_actor_solution):
        # Add 5 solutions with different iterations and acceptance
        for i in range(5):
            memory_manager.save_actor_solution(
                solution=sample_actor_solution,
                task_type=TaskType.CLASSIFICATION,
                target="Species",
                iteration=i + 1,
                accepted=(i % 2 == 0),  # 1st, 3rd, 5th accepted
            )

        context = memory_manager.get_actor_context(
            task_type=TaskType.CLASSIFICATION, target="Species", limit=3
        )

        assert len(context) == 3
        # Best (accepted + highest iteration) should be first
        assert context[0]["iteration"] == 5
        assert context[0]["accepted"] is True
        # Second best should be last (sandwich pattern)
        assert context[-1]["iteration"] == 3
        assert context[-1]["accepted"] is True


class TestCriticMemory:
    def test_save_critic_evaluation(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        memory_id = memory_manager.save_critic_evaluation(
            evaluation=sample_critic_evaluation,
            task_type=TaskType.CLASSIFICATION,
            iteration=1,
        )

        assert memory_id != ""
        assert len(memory_manager.critics.all()) == 1

        saved = memory_manager.critics.all()[0]
        assert saved["task_type"] == "classification"
        assert saved["score"] == 1

    def test_get_critic_context_empty(self, memory_manager):
        context = memory_manager.get_critic_context(task_type=TaskType.CLASSIFICATION, limit=3)

        assert context == []

    def test_get_critic_context_with_memories(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        memory_manager.save_critic_evaluation(
            evaluation=sample_critic_evaluation,
            task_type=TaskType.CLASSIFICATION,
            iteration=1,
        )

        context = memory_manager.get_critic_context(task_type=TaskType.CLASSIFICATION, limit=3)

        assert len(context) == 1
        assert context[0]["task_type"] == "classification"
        assert context[0]["score"] == 1


class TestUtilityMethods:
    def test_clear_all(self, memory_manager, sample_actor_solution, sample_critic_evaluation):
        # Add some data
        memory_manager.save_actor_solution(
            solution=sample_actor_solution,
            task_type=TaskType.CLASSIFICATION,
            target="Species",
            iteration=1,
            accepted=True,
        )
        memory_manager.save_critic_evaluation(
            evaluation=sample_critic_evaluation,
            task_type=TaskType.CLASSIFICATION,
            iteration=1,
        )

        assert len(memory_manager.actors.all()) == 1
        assert len(memory_manager.critics.all()) == 1

        # Clear everything
        memory_manager.clear_all()

        assert len(memory_manager.actors.all()) == 0
        assert len(memory_manager.critics.all()) == 0


class TestPersistence:
    def test_memory_persists_across_instances(self, temp_db_file, sample_actor_solution):
        # Create first instance and save data
        memory1 = MemoryManager(persist_path=str(temp_db_file))
        memory1.save_actor_solution(
            solution=sample_actor_solution,
            task_type=TaskType.CLASSIFICATION,
            target="Species",
            iteration=1,
            accepted=True,
        )

        del memory1

        # Create second instance and verify data persists
        memory2 = MemoryManager(persist_path=str(temp_db_file))
        context = memory2.get_actor_context(
            task_type=TaskType.CLASSIFICATION, target="Species", limit=3
        )

        assert len(context) == 1
        assert context[0]["target"] == "Species"
