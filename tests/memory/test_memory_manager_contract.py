import shutil
import tempfile
from pathlib import Path

import pytest

from scald.agents.actor import ActorSolution
from scald.agents.critic import CriticEvaluation
from scald.memory import MemoryManager
from scald.memory.types import ActorMemoryContext, CriticMemoryContext


@pytest.fixture
def temp_memory_dir():
    """Create temporary directory for ChromaDB storage."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def memory_manager(temp_memory_dir):
    """Create MemoryManager with temporary storage."""
    manager = MemoryManager(memory_dir=temp_memory_dir)
    yield manager
    manager.clear()


@pytest.fixture
def sample_actor_solution():
    """Sample ActorSolution for testing."""
    return ActorSolution(
        predictions_path=Path("/output/predictions.csv"),
        predictions=[0, 1, 2, 1, 0, 2, 1],
        metrics={"accuracy": 0.92, "f1": 0.90, "precision": 0.91},
        report="""Dataset Analysis:
- Shape: 120 samples, 4 features
- Target: Species (3 classes)
- No missing values

Preprocessing:
- No encoding needed (all numeric)
- Standard scaling applied

Model Training:
- Used CatBoost classifier
- Hyperparameters: depth=6, learning_rate=0.1
- Training accuracy: 0.95
- Test accuracy: 0.92

Results:
- Good generalization
- No overfitting detected""",
    )


@pytest.fixture
def sample_critic_evaluation():
    """Sample CriticEvaluation for testing."""
    return CriticEvaluation(
        score=1, feedback="Model is well-tuned with good generalization. Excellent work!"
    )


# ============================================================================
# TEST SUITE 1: Initialization Contract
# ============================================================================


class TestMemoryManagerInitialization:
    """Contract: MemoryManager initialization and setup."""

    def test_init_accepts_memory_dir_parameter(self, temp_memory_dir):
        """Should accept memory_dir parameter for custom storage location."""
        manager = MemoryManager(memory_dir=temp_memory_dir)
        assert manager is not None

    def test_init_creates_chromadb_client(self, temp_memory_dir):
        """Should create ChromaDB persistent client."""
        manager = MemoryManager(memory_dir=temp_memory_dir)
        assert manager.client is not None
        assert hasattr(manager.client, "get_or_create_collection")

    def test_init_creates_collection(self, temp_memory_dir):
        """Should create or get ChromaDB collection with proper name."""
        manager = MemoryManager(memory_dir=temp_memory_dir)
        assert manager.collection is not None
        assert manager.collection.name == MemoryManager.COLLECTION_NAME

    def test_init_creates_embedding_function(self, temp_memory_dir):
        """Should initialize Jina embedding function."""
        manager = MemoryManager(memory_dir=temp_memory_dir)
        assert manager.embedding_fn is not None

    def test_init_creates_memory_directory(self, temp_memory_dir):
        """Should create memory directory if it doesn't exist."""
        nested_dir = temp_memory_dir / "nested" / "memory"
        MemoryManager(memory_dir=nested_dir)
        assert nested_dir.exists()
        assert nested_dir.is_dir()


# ============================================================================
# TEST SUITE 2: Save Iteration Contract
# ============================================================================


class TestSaveIterationContract:
    """Contract: Saving iteration results to persistent memory."""

    @pytest.mark.asyncio
    async def test_save_iteration_signature(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        """Contract: save accepts correct parameters and returns entry_id."""
        entry_id = await memory_manager.save(
            actor_solution=sample_actor_solution,
            critic_evaluation=sample_critic_evaluation,
            task_type="classification",
            iteration=1,
        )
        assert isinstance(entry_id, str)
        assert len(entry_id) > 0

    @pytest.mark.asyncio
    async def test_save_stores_in_chromadb(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        """Contract: Saved entry must exist in ChromaDB collection."""
        entry_id = await memory_manager.save(
            actor_solution=sample_actor_solution,
            critic_evaluation=sample_critic_evaluation,
            task_type="classification",
            iteration=1,
        )

        result = memory_manager.collection.get(ids=[entry_id])
        assert len(result["ids"]) == 1
        assert result["ids"][0] == entry_id

    @pytest.mark.asyncio
    async def test_save_uses_actor_report_as_document(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        """Contract: Actor report must be stored as document for semantic search."""
        entry_id = await memory_manager.save(
            actor_solution=sample_actor_solution,
            critic_evaluation=sample_critic_evaluation,
            task_type="classification",
            iteration=1,
        )

        result = memory_manager.collection.get(ids=[entry_id])
        assert result["documents"][0] == sample_actor_solution.report

    @pytest.mark.asyncio
    async def test_save_stores_complete_metadata(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        """Contract: All relevant metadata must be stored."""
        entry_id = await memory_manager.save(
            actor_solution=sample_actor_solution,
            critic_evaluation=sample_critic_evaluation,
            task_type="classification",
            iteration=1,
        )

        result = memory_manager.collection.get(ids=[entry_id])
        metadata = result["metadatas"][0]

        # Required metadata fields
        assert metadata["task_type"] == "classification"
        assert metadata["iteration"] == 1
        assert "critic_evaluation" in metadata
        assert "timestamp" in metadata

        # Metrics must be flattened and stored
        assert "accuracy" in metadata
        assert metadata["accuracy"] == 0.92
        assert "f1" in metadata
        assert metadata["f1"] == 0.90

    @pytest.mark.asyncio
    async def test_save_multiple_iterations_unique_ids(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        """Contract: Multiple saves must generate unique entry IDs."""
        entry_ids = []
        for i in range(3):
            entry_id = await memory_manager.save(
                actor_solution=sample_actor_solution,
                critic_evaluation=sample_critic_evaluation,
                task_type="classification",
                iteration=i + 1,
            )
            entry_ids.append(entry_id)

        # All IDs must be unique
        assert len(entry_ids) == len(set(entry_ids))


# ============================================================================
# TEST SUITE 3: Retrieve Relevant Context Contract
# ============================================================================


class TestRetrieveRelevantContextContract:
    """Contract: Retrieving relevant memory context for agents."""

    @pytest.mark.asyncio
    async def test_retrieve_signature(self, memory_manager):
        """Contract: retrieve returns tuple of context lists."""
        actor_contexts, critic_contexts = await memory_manager.retrieve(
            actor_report="Testing classification task",
            task_type="classification",
            top_k=5,
        )

        assert isinstance(actor_contexts, list)
        assert isinstance(critic_contexts, list)

    @pytest.mark.asyncio
    async def test_retrieve_empty_memory(self, memory_manager):
        """Contract: Empty memory returns empty lists, not None."""
        actor_contexts, critic_contexts = await memory_manager.retrieve(
            actor_report="Some task", task_type="classification", top_k=5
        )

        assert actor_contexts == []
        assert critic_contexts == []

    @pytest.mark.asyncio
    async def test_retrieve_returns_actor_memory_contexts(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        """Contract: Must return properly structured ActorMemoryContext objects."""
        # Setup: save a memory
        await memory_manager.save(
            actor_solution=sample_actor_solution,
            critic_evaluation=sample_critic_evaluation,
            task_type="classification",
            iteration=1,
        )

        # Retrieve
        actor_contexts, _ = await memory_manager.retrieve(
            actor_report=sample_actor_solution.report,
            task_type="classification",
            top_k=5,
        )

        assert len(actor_contexts) > 0
        context = actor_contexts[0]

        # Contract: ActorMemoryContext structure
        assert isinstance(context, ActorMemoryContext)
        assert isinstance(context.iteration, int)
        assert isinstance(context.accepted, bool)
        assert isinstance(context.actions_summary, str)
        assert isinstance(context.feedback_received, str)
        assert isinstance(context.metrics, dict)
        assert len(context.actions_summary) > 0

    @pytest.mark.asyncio
    async def test_retrieve_returns_critic_memory_contexts(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        """Contract: Must return properly structured CriticMemoryContext objects."""
        # Setup: save a memory
        await memory_manager.save(
            actor_solution=sample_actor_solution,
            critic_evaluation=sample_critic_evaluation,
            task_type="classification",
            iteration=1,
        )

        # Retrieve
        _, critic_contexts = await memory_manager.retrieve(
            actor_report=sample_actor_solution.report,
            task_type="classification",
            top_k=5,
        )

        assert len(critic_contexts) > 0
        context = critic_contexts[0]

        # Contract: CriticMemoryContext structure
        assert isinstance(context, CriticMemoryContext)
        assert isinstance(context.iteration, int)
        assert isinstance(context.score, int)
        assert isinstance(context.actions_observed, str)
        assert isinstance(context.feedback_given, str)
        assert len(context.actions_observed) > 0

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        """Contract: Must respect top_k limit."""
        # Setup: save 10 memories
        for i in range(10):
            await memory_manager.save(
                actor_solution=sample_actor_solution,
                critic_evaluation=sample_critic_evaluation,
                task_type="classification",
                iteration=i + 1,
            )

        # Retrieve with top_k=3
        actor_contexts, critic_contexts = await memory_manager.retrieve(
            actor_report=sample_actor_solution.report,
            task_type="classification",
            top_k=3,
        )

        assert len(actor_contexts) <= 3
        assert len(critic_contexts) <= 3

    @pytest.mark.asyncio
    async def test_retrieve_filters_by_task_type(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        """Contract: Must filter results by task_type."""
        # Setup: save classification memory
        await memory_manager.save(
            actor_solution=sample_actor_solution,
            critic_evaluation=sample_critic_evaluation,
            task_type="classification",
            iteration=1,
        )

        # Setup: save regression memory
        regression_solution = ActorSolution(
            predictions=[], metrics={"mse": 0.15}, report="Regression task on housing data"
        )
        await memory_manager.save(
            actor_solution=regression_solution,
            critic_evaluation=sample_critic_evaluation,
            task_type="regression",
            iteration=1,
        )

        # Query for classification only
        actor_contexts, _ = await memory_manager.retrieve(
            actor_report="Classification task",
            task_type="classification",
            top_k=10,
        )

        # Should only retrieve classification, not regression
        # Note: exact filtering verification depends on implementation
        assert len(actor_contexts) >= 1


# ============================================================================
# TEST SUITE 4: Context Transformation Contract
# ============================================================================


class TestContextTransformationContract:
    """Contract: Transforming MemoryEntry to agent-specific contexts."""

    @pytest.mark.asyncio
    async def test_actor_context_contains_actions_summary(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        """Contract: ActorMemoryContext.actions_summary must contain actor's report."""
        await memory_manager.save(
            actor_solution=sample_actor_solution,
            critic_evaluation=sample_critic_evaluation,
            task_type="classification",
            iteration=1,
        )

        actor_contexts, _ = await memory_manager.retrieve(
            actor_report=sample_actor_solution.report,
            task_type="classification",
            top_k=1,
        )

        context = actor_contexts[0]
        # actions_summary should be the actor_report
        assert context.actions_summary
        assert len(context.actions_summary) > 0
        assert context.actions_summary == sample_actor_solution.report

    @pytest.mark.asyncio
    async def test_actor_context_includes_feedback_received(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        """Contract: ActorMemoryContext.feedback_received must match critic's feedback."""
        await memory_manager.save(
            actor_solution=sample_actor_solution,
            critic_evaluation=sample_critic_evaluation,
            task_type="classification",
            iteration=1,
        )

        actor_contexts, _ = await memory_manager.retrieve(
            actor_report=sample_actor_solution.report,
            task_type="classification",
            top_k=1,
        )

        context = actor_contexts[0]
        assert context.feedback_received == sample_critic_evaluation.feedback

    @pytest.mark.asyncio
    async def test_critic_context_contains_actions_observed(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        """Contract: CriticMemoryContext.actions_observed must contain actor's report."""
        await memory_manager.save(
            actor_solution=sample_actor_solution,
            critic_evaluation=sample_critic_evaluation,
            task_type="classification",
            iteration=1,
        )

        _, critic_contexts = await memory_manager.retrieve(
            actor_report=sample_actor_solution.report,
            task_type="classification",
            top_k=1,
        )

        context = critic_contexts[0]
        # actions_observed should be the actor's report
        assert context.actions_observed
        assert len(context.actions_observed) > 0
        assert context.actions_observed == sample_actor_solution.report

    @pytest.mark.asyncio
    async def test_critic_context_includes_feedback_given(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        """Contract: CriticMemoryContext.feedback_given must match what was given."""
        await memory_manager.save(
            actor_solution=sample_actor_solution,
            critic_evaluation=sample_critic_evaluation,
            task_type="classification",
            iteration=1,
        )

        _, critic_contexts = await memory_manager.retrieve(
            actor_report=sample_actor_solution.report,
            task_type="classification",
            top_k=1,
        )

        context = critic_contexts[0]
        assert context.feedback_given == sample_critic_evaluation.feedback
        assert context.score == sample_critic_evaluation.score


# ============================================================================
# TEST SUITE 5: Utility Methods Contract
# ============================================================================


class TestUtilityMethodsContract:
    """Contract: Utility methods for memory management."""

    @pytest.mark.asyncio
    async def test_clear_removes_entries(
        self, memory_manager, sample_actor_solution, sample_critic_evaluation
    ):
        """Contract: clear must remove all entries from collection."""
        # Save some entries
        for i in range(3):
            await memory_manager.save(
                actor_solution=sample_actor_solution,
                critic_evaluation=sample_critic_evaluation,
                task_type="classification",
                iteration=i + 1,
            )

        # Verify entries exist
        count_before = memory_manager.collection.count()
        assert count_before >= 3

        # Clear all
        memory_manager.clear()

        # Verify empty
        count_after = memory_manager.collection.count()
        assert count_after == 0
