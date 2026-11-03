"""
Unit tests for Decision and Action Item extraction modules.
Strictly verifies the explicit owner attribution rules:
- Owner is assigned ONLY when explicitly stated in the transcript
- Ambiguous or collective tasks are strictly 'Unassigned'
- Never guess or infer owners.
"""

import pytest
from src.schemas import TranscriptSegment
from src.extraction.decisions import DecisionExtractor
from src.extraction.action_items import ActionItemExtractor


@pytest.fixture(scope="module")
def decision_extractor():
    return DecisionExtractor()


@pytest.fixture(scope="module")
def action_extractor():
    return ActionItemExtractor()


def test_direct_named_assignment(action_extractor):
    """Test assigning task to explicitly named person."""
    seg = TranscriptSegment(
        segment_id=0,
        speaker="Alice",
        start=10.0,
        end=14.0,
        text="Priya, please send the quarterly performance report by Friday.",
    )
    items = action_extractor.extract([seg])
    assert len(items) == 1
    assert items[0].owner == "Priya"
    assert "quarterly performance report" in items[0].description.lower()
    assert items[0].deadline == "Friday"
    assert items[0].source_speaker == "Alice"
    assert items[0].timestamp == 10.0


def test_third_person_assignment(action_extractor):
    """Test explicit third-person delegation."""
    seg = TranscriptSegment(
        segment_id=1,
        speaker="Anas",
        start=20.0,
        end=24.0,
        text="Bob will deploy the staging database by next Monday.",
    )
    items = action_extractor.extract([seg])
    assert len(items) == 1
    assert items[0].owner == "Bob"
    assert "deploy the staging database" in items[0].description.lower()
    assert items[0].deadline == "next Monday"


def test_first_person_commitment_known_speaker(action_extractor):
    """Test first-person commitment ('I will...') when speaker identity is known."""
    seg = TranscriptSegment(
        segment_id=2,
        speaker="Priya",
        start=30.0,
        end=34.0,
        text="I will review the pull requests before end of day.",
    )
    items = action_extractor.extract([seg])
    assert len(items) == 1
    assert items[0].owner == "Priya"
    assert "review the pull requests" in items[0].description.lower()
    assert items[0].deadline == "end of day"


def test_first_person_commitment_unknown_speaker(action_extractor):
    """Test first-person commitment when speaker is unknown falls back to Unassigned."""
    seg = TranscriptSegment(
        segment_id=3,
        speaker="Speaker unknown",
        start=40.0,
        end=43.0,
        text="I'll update the unit test suite tomorrow.",
    )
    items = action_extractor.extract([seg])
    assert len(items) == 1
    # Because speaker identity is unknown, we must NOT guess
    assert items[0].owner == "Unassigned"
    assert items[0].deadline == "tomorrow"


def test_collective_directive_strictly_unassigned(action_extractor):
    """Test collective suggestions ('we need to') strictly set owner to 'Unassigned'."""
    seg = TranscriptSegment(
        segment_id=4,
        speaker="Alice",
        start=50.0,
        end=54.0,
        text="We need to update the API documentation before next week.",
    )
    items = action_extractor.extract([seg])
    assert len(items) == 1
    assert items[0].owner == "Unassigned"
    assert items[0].deadline == "next week"


def test_passive_directive_strictly_unassigned(action_extractor):
    """Test passive/ambiguous suggestions ('someone needs to') strictly set owner to 'Unassigned'."""
    seg = TranscriptSegment(
        segment_id=5,
        speaker="Bob",
        start=60.0,
        end=63.0,
        text="Someone needs to check the production error logs.",
    )
    items = action_extractor.extract([seg])
    assert len(items) == 1
    assert items[0].owner == "Unassigned"
    assert items[0].deadline is None


def test_non_person_entity_not_owner(action_extractor):
    """Ensure days, tools, and technical terms are never parsed as person owners."""
    seg = TranscriptSegment(
        segment_id=6,
        speaker="Alice",
        start=70.0,
        end=74.0,
        text="PostgreSQL will store our analytical event stream.",
    )
    items = action_extractor.extract([seg])
    # PostgreSQL is not a person
    for item in items:
        assert item.owner != "Postgresql"


def test_decisions_extraction(decision_extractor):
    """Test formal decision detection and proposer extraction."""
    seg1 = TranscriptSegment(
        segment_id=0,
        speaker="Anas",
        start=15.0,
        end=20.0,
        text="We decided to proceed with the microservices architecture.",
    )
    seg2 = TranscriptSegment(
        segment_id=1,
        speaker="Priya",
        start=25.0,
        end=30.0,
        text="Let's go with Python for the backend services.",
    )
    # Question should be ignored
    seg3 = TranscriptSegment(
        segment_id=2,
        speaker="Bob",
        start=35.0,
        end=38.0,
        text="Did we decide to use Kubernetes or ECS?",
    )

    decisions = decision_extractor.extract([seg1, seg2, seg3])
    assert len(decisions) == 2
    assert "microservices architecture" in decisions[0].statement.lower()
    assert decisions[0].proposed_by == "Anas"
    assert decisions[0].timestamp == 15.0

    assert "python for the backend" in decisions[1].statement.lower()
    assert decisions[1].proposed_by == "Priya"
