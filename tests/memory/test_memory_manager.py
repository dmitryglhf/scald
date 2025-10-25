import tempfile
from pathlib import Path

import pytest

from scald.common.types import ActorSolution, CriticEvaluation, TaskType
from scald.memory import MemoryManager


@pytest.fixture
def temp_db_file():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmpfile:
        yield Path(tmpfile.name)
        # Cleanup
        if tmpfile.name and Path(tmpfile.name).exists():
            Path(tmpfile.name).unlink()


@pytest.fixture
def memory_manager(temp_db_file):
    memory = MemoryManager(persist_path=str(temp_db_file), enabled=True)
    yield memory
    memory.clear()


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
    def test_initialization_enabled(self, temp_db_file):
        memory = MemoryManager(persist_path=str(temp_db_file), enabled=True)

        assert memory.enabled is True
        assert memory.db is not None
        assert memory.actors is not None
        assert memory.critics is not None
        assert memory.generic is not None

    def test_initialization_disabled(self, temp_db_file):
        memory = MemoryManager(persist_path=str(temp_db_file), enabled=False)

        assert memory.enabled is False
        assert memory.db is None
        assert memory.actors is None


class TestActorMemory:
    def test_save_actor_solution(self, memory_manager, sample_actor_solution):
        memory_manager.save_actor_solution(
            solution=sample_actor_solution,
            task_type=TaskType.CLASSIFICATION,
            target="Species",
            iteration=1,
            accepted=True,
        )

        all_memories = memory_manager.get_all(filter_metadata={"agent": "actor"})
        assert len(all_memories) == 1
        assert all_memories[0]["metadata"]["task_type"] == "classification"
        assert all_memories[0]["metadata"]["target"] == "Species"
        assert all_memories[0]["metadata"]["accepted"] is True

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
        all_memories = memory_manager.get_all(filter_metadata={"agent": "actor"})
        assert all_memories[0]["metadata"]["accepted"] is False

        # Update to accepted
        success = memory_manager.update_actor_solution_status(
            task_type=TaskType.CLASSIFICATION,
            target="Species",
            iteration=1,
            accepted=True,
        )

        assert success is True

        # Verify updated status
        all_memories = memory_manager.get_all(filter_metadata={"agent": "actor"})
        assert all_memories[0]["metadata"]["accepted"] is True

    def test_update_actor_solution_status_not_found(self, memory_manager):
        # Try to update non-existent solution
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
        assert "text" in context[0]
        assert "metadata" in context[0]
        assert "id" in context[0]
        assert context[0]["metadata"]["accepted"] is True

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
        assert context[0]["metadata"]["iteration"] == 5
        assert context[0]["metadata"]["accepted"] is True


class TestCriticMemory:
    def test_save_critic_evaluation(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        memory_manager.save_critic_evaluation(
            evaluation=sample_critic_evaluation,
            solution=sample_actor_solution,
            task_type=TaskType.CLASSIFICATION,
            iteration=1,
        )

        all_memories = memory_manager.get_all(filter_metadata={"agent": "critic"})
        assert len(all_memories) == 1
        assert all_memories[0]["metadata"]["task_type"] == "classification"
        assert all_memories[0]["metadata"]["score"] == 1

    def test_get_critic_context_empty(self, memory_manager):
        context = memory_manager.get_critic_context(task_type=TaskType.CLASSIFICATION, limit=3)

        assert context == []

    def test_get_critic_context_with_memories(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        memory_manager.save_critic_evaluation(
            evaluation=sample_critic_evaluation,
            solution=sample_actor_solution,
            task_type=TaskType.CLASSIFICATION,
            iteration=1,
        )

        context = memory_manager.get_critic_context(task_type=TaskType.CLASSIFICATION, limit=3)

        assert len(context) == 1
        assert "text" in context[0]
        assert "metadata" in context[0]
        assert context[0]["metadata"]["score"] == 1


class TestGenericOperations:
    def test_add_memory(self, memory_manager):
        memory_id = memory_manager.add(text="Test memory text", metadata={"custom": "data"})

        assert memory_id != ""
        all_memories = memory_manager.get_all()
        assert len(all_memories) == 1

    def test_search(self, memory_manager):
        memory_manager.add(
            text="CatBoost classifier with high accuracy", metadata={"type": "solution"}
        )

        results = memory_manager.search(filter_metadata={"type": "solution"}, limit=5)
        assert len(results) == 1

    def test_delete(self, memory_manager):
        memory_id = memory_manager.add(text="Test memory", metadata={"test": True})

        success = memory_manager.delete(memory_id)
        assert success is True

        all_memories = memory_manager.get_all()
        assert len(all_memories) == 0

    def test_clear_all(self, memory_manager):
        memory_manager.add("Memory 1", {"id": 1})
        memory_manager.add("Memory 2", {"id": 2})

        memory_manager.clear()

        all_memories = memory_manager.get_all()
        assert len(all_memories) == 0

    def test_clear_with_filter(self, memory_manager):
        memory_manager.add("Memory 1", {"category": "A"})
        memory_manager.add("Memory 2", {"category": "B"})

        memory_manager.clear(filter_metadata={"category": "A"})

        all_memories = memory_manager.get_all()
        assert len(all_memories) == 1
        assert all_memories[0]["metadata"]["category"] == "B"


class TestDisabledMemory:
    def test_operations_when_disabled(self, temp_db_file, sample_actor_solution):
        memory = MemoryManager(persist_path=str(temp_db_file), enabled=False)

        memory.save_actor_solution(
            solution=sample_actor_solution,
            task_type=TaskType.CLASSIFICATION,
            target="Species",
            iteration=1,
            accepted=True,
        )

        context = memory.get_actor_context(TaskType.CLASSIFICATION, "Species")
        assert context == []

        memory_id = memory.add("test", {})
        assert memory_id == ""


class TestPersistence:
    def test_memory_persists_across_instances(self, temp_db_file, sample_actor_solution):
        memory1 = MemoryManager(persist_path=str(temp_db_file), enabled=True)

        memory1.save_actor_solution(
            solution=sample_actor_solution,
            task_type=TaskType.CLASSIFICATION,
            target="Species",
            iteration=1,
            accepted=True,
        )

        del memory1

        memory2 = MemoryManager(persist_path=str(temp_db_file), enabled=True)

        all_memories = memory2.get_all()
        assert len(all_memories) == 1
        assert all_memories[0]["metadata"]["target"] == "Species"
