import shutil
import tempfile

import pytest

from scald.agents.actor import ActorSolution
from scald.agents.critic import CriticEvaluation
from scald.memory import MemoryManager


@pytest.fixture
def temp_db_path():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def memory_manager(temp_db_path):
    memory = MemoryManager(persist_path=temp_db_path, use_jina=False)
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
    def test_initialization(self, temp_db_path):
        memory = MemoryManager(persist_path=temp_db_path, use_jina=False)

        assert memory.client is not None
        assert memory.actors is not None
        assert memory.critics is not None


class TestActorMemory:
    def test_save_actor_solution(self, memory_manager, sample_actor_solution):
        memory_id = memory_manager.save_actor_solution(
            solution=sample_actor_solution,
            task_type="classification",
            target="Species",
            iteration=1,
            accepted=True,
        )

        assert memory_id != ""
        assert memory_manager.actors.count() == 1

        contexts = memory_manager.get_actor_context(task_type="classification", target="Species")
        assert len(contexts) == 1
        assert contexts[0].task_type == "classification"
        assert contexts[0].target == "Species"
        assert contexts[0].accepted is True

    def test_update_actor_solution_status(self, memory_manager, sample_actor_solution):
        memory_manager.save_actor_solution(
            solution=sample_actor_solution,
            task_type="classification",
            target="Species",
            iteration=1,
            accepted=False,
        )

        contexts = memory_manager.get_actor_context(task_type="classification", target="Species")
        assert contexts[0].accepted is False

        success = memory_manager.update_actor_solution_status(
            task_type="classification",
            target="Species",
            iteration=1,
            accepted=True,
        )

        assert success is True

        contexts = memory_manager.get_actor_context(task_type="classification", target="Species")
        assert contexts[0].accepted is True

    def test_update_actor_solution_status_not_found(self, memory_manager):
        success = memory_manager.update_actor_solution_status(
            task_type="classification",
            target="NonExistent",
            iteration=999,
            accepted=True,
        )

        assert success is False

    def test_get_actor_context_empty(self, memory_manager):
        context = memory_manager.get_actor_context(
            task_type="classification", target="Species", limit=3
        )

        assert context == []

    def test_get_actor_context_with_memories(self, memory_manager, sample_actor_solution):
        memory_manager.save_actor_solution(
            solution=sample_actor_solution,
            task_type="classification",
            target="Species",
            iteration=1,
            accepted=True,
        )

        context = memory_manager.get_actor_context(
            task_type="classification", target="Species", limit=3
        )

        assert len(context) == 1
        assert context[0].task_type == "classification"
        assert context[0].target == "Species"
        assert context[0].accepted is True

    def test_actor_context_filters_by_task_type(self, memory_manager, sample_actor_solution):
        memory_manager.save_actor_solution(
            solution=sample_actor_solution,
            task_type="classification",
            target="Species",
            iteration=1,
            accepted=True,
        )

        context = memory_manager.get_actor_context(task_type="regression", target="Price", limit=3)

        assert len(context) == 0

    def test_actor_context_sorting(self, memory_manager, sample_actor_solution):
        for i in range(5):
            memory_manager.save_actor_solution(
                solution=sample_actor_solution,
                task_type="classification",
                target="Species",
                iteration=i + 1,
                accepted=(i % 2 == 0),
            )

        context = memory_manager.get_actor_context(
            task_type="classification", target="Species", limit=3
        )

        assert len(context) == 3
        assert context[0].iteration == 5
        assert context[0].accepted is True


class TestCriticMemory:
    def test_save_critic_evaluation(self, memory_manager, sample_critic_evaluation):
        memory_id = memory_manager.save_critic_evaluation(
            evaluation=sample_critic_evaluation,
            task_type="classification",
            iteration=1,
        )

        assert memory_id != ""
        assert memory_manager.critics.count() == 1

        contexts = memory_manager.get_critic_context(task_type="classification")
        assert len(contexts) == 1
        assert contexts[0].task_type == "classification"
        assert contexts[0].score == 1

    def test_get_critic_context_empty(self, memory_manager):
        context = memory_manager.get_critic_context(task_type="classification", limit=3)

        assert context == []

    def test_get_critic_context_with_memories(self, memory_manager, sample_critic_evaluation):
        memory_manager.save_critic_evaluation(
            evaluation=sample_critic_evaluation,
            task_type="classification",
            iteration=1,
        )

        context = memory_manager.get_critic_context(task_type="classification", limit=3)

        assert len(context) == 1
        assert context[0].task_type == "classification"
        assert context[0].score == 1


class TestUtilityMethods:
    def test_clear_all(self, memory_manager, sample_actor_solution, sample_critic_evaluation):
        memory_manager.save_actor_solution(
            solution=sample_actor_solution,
            task_type="classification",
            target="Species",
            iteration=1,
            accepted=True,
        )
        memory_manager.save_critic_evaluation(
            evaluation=sample_critic_evaluation,
            task_type="classification",
            iteration=1,
        )

        assert memory_manager.actors.count() == 1
        assert memory_manager.critics.count() == 1

        memory_manager.clear_all()

        assert memory_manager.actors.count() == 0
        assert memory_manager.critics.count() == 0


class TestPersistence:
    def test_memory_persists_across_instances(self, temp_db_path, sample_actor_solution):
        memory1 = MemoryManager(persist_path=temp_db_path, use_jina=False)
        memory1.save_actor_solution(
            solution=sample_actor_solution,
            task_type="classification",
            target="Species",
            iteration=1,
            accepted=True,
        )

        del memory1

        memory2 = MemoryManager(persist_path=temp_db_path, use_jina=False)
        context = memory2.get_actor_context(task_type="classification", target="Species", limit=3)

        assert len(context) == 1
        assert context[0].target == "Species"
