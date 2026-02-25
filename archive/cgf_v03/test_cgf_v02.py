"""
test_cgf_v02.py - Capacity Governance Framework v0.2 Integration Tests

Tests:
1. Tool call governance (existing v0.1 tests must still pass)
2. Memory write governance (new v0.2):
   - memory_write allowed
   - memory_write constrained (quarantine)
   - memory_write blocked on CGF down
3. Schema validation
4. Replay reconstruction
"""

import asyncio
import sys
from pathlib import Path

# Add current dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cgf_server_v02 import app
from fastapi.testclient import TestClient
from cgf_schemas_v02 import (
    SCHEMA_VERSION,
    ActionType,
    DecisionType,
    HostEventType,
    MemoryWriteParams,
    get_event_required_fields
)

# Create test client
client = TestClient(app)

TEST_RESULTS = []

def test(name):
    """Decorator to track test results."""
    def decorator(func):
        async def wrapper():
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print('='*60)
            try:
                await func()
                TEST_RESULTS.append((name, True, None))
                print(f"✅ PASS: {name}")
                return True
            except AssertionError as e:
                TEST_RESULTS.append((name, False, str(e)))
                print(f"❌ FAIL: {name}")
                print(f"   {e}")
                return False
            except Exception as e:
                TEST_RESULTS.append((name, False, f"Exception: {e}"))
                print(f"💥 ERROR: {name}")
                print(f"   {type(e).__name__}: {e}")
                return False
        return wrapper
    return decorator

# ============== TEST SUITE ==============

@test("Schema version constant defined")
async def test_schema_version():
    """Verify schema version is 0.2.0."""
    assert SCHEMA_VERSION == "0.2.0", f"Expected 0.2.0, got {SCHEMA_VERSION}"
    print(f"  Schema version: {SCHEMA_VERSION}")

@test("Server root endpoint returns correct version")
async def test_server_version():
    """Verify server reports correct schema version."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["schema_version"] == "0.2.0"
    assert data["policy_version"] == "0.2.0"
    print(f"  Service: {data['service']}")
    print(f"  Schema: {data['schema_version']}")
    print(f"  Policy: {data['policy_version']}")

@test("Registration includes schema version")
async def test_registration_schema():
    """Verify registration payload includes schema version."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "adapter_type": "openclaw",
        "host_config": {
            "host_type": "openclaw",
            "namespace": "default",
            "capabilities": ["tool_call", "memory_write"],
            "version": "0.2.0"
        }
    }
    response = client.post("/v1/register", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["schema_version"] == "0.2.0"
    print(f"  Adapter registered: {data['adapter_id'][:20]}...")

@test("Tool call - denylisted tool blocked")
async def test_tool_call_blocked():
    """Verify file_write is blocked (existing v0.1 behavior)."""
    # Register first
    reg = client.post("/v1/register", json={
        "schema_version": SCHEMA_VERSION,
        "adapter_type": "test",
        "host_config": {"host_type": "openclaw", "namespace": "test"}
    }).json()
    adapter_id = reg["adapter_id"]
    
    payload = {
        "schema_version": SCHEMA_VERSION,
        "adapter_id": adapter_id,
        "host_config": {"host_type": "openclaw", "namespace": "test"},
        "proposal": {
            "proposal_id": "prop-test-1",
            "timestamp": 1234567890,
            "action_type": "tool_call",
            "action_params": {
                "tool_name": "file_write",
                "tool_args_hash": "abc123",
                "side_effects_hint": ["write"],
                "idempotent_hint": False
            },
            "context_refs": ["session-1", "agent-1"],
            "risk_tier": "high",
            "priority": 0
        },
        "context": {"turn_number": 0},
        "capacity_signals": {}
    }
    
    response = client.post("/v1/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    decision = data["decision"]
    
    assert decision["decision"] == "BLOCK"
    assert "file_write" in decision["justification"]
    print(f"  Decision: {decision['decision']}")
    print(f"  Justification: {decision['justification']}")

@test("Tool call - non-denylisted tool allowed")
async def test_tool_call_allowed():
    """Verify ls is allowed (existing v0.1 behavior)."""
    reg = client.post("/v1/register", json={
        "schema_version": SCHEMA_VERSION,
        "adapter_type": "test",
        "host_config": {"host_type": "openclaw", "namespace": "test"}
    }).json()
    adapter_id = reg["adapter_id"]
    
    payload = {
        "schema_version": SCHEMA_VERSION,
        "adapter_id": adapter_id,
        "host_config": {"host_type": "openclaw", "namespace": "test"},
        "proposal": {
            "proposal_id": "prop-test-2",
            "timestamp": 1234567890,
            "action_type": "tool_call",
            "action_params": {
                "tool_name": "ls",
                "tool_args_hash": "def456",
                "side_effects_hint": [],
                "idempotent_hint": True
            },
            "context_refs": ["session-1"],
            "risk_tier": "medium",
            "priority": 0
        },
        "context": {"turn_number": 0},
        "capacity_signals": {}
    }
    
    response = client.post("/v1/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    decision = data["decision"]
    
    assert decision["decision"] == "ALLOW"
    print(f"  Decision: {decision['decision']}")

@test("Memory write - low sensitivity allowed")
async def test_memory_write_allowed():
    """Verify low sensitivity memory write is allowed."""
    reg = client.post("/v1/register", json={
        "schema_version": SCHEMA_VERSION,
        "adapter_type": "test",
        "host_config": {"host_type": "openclaw", "namespace": "test"}
    }).json()
    adapter_id = reg["adapter_id"]
    
    payload = {
        "schema_version": SCHEMA_VERSION,
        "adapter_id": adapter_id,
        "host_config": {"host_type": "openclaw", "namespace": "test"},
        "proposal": {
            "proposal_id": "prop-mem-1",
            "timestamp": 1234567890,
            "action_type": "memory_write",
            "action_params": {
                "namespace": "sessions/default",
                "size_bytes": 1024,
                "ttl": None,
                "sensitivity_hint": "low",
                "content_hash": "a" * 64,
                "context_refs": ["session-abc"],
                "operation": "update"
            },
            "context_refs": ["session-abc"],
            "risk_tier": "low",
            "priority": 0
        },
        "context": {"turn_number": 0},
        "capacity_signals": {}
    }
    
    response = client.post("/v1/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    decision = data["decision"]
    
    assert decision["decision"] == "ALLOW"
    assert decision["confidence"] > 0.8
    print(f"  Decision: {decision['decision']}")
    print(f"  Confidence: {decision['confidence']}")
    print(f"  Justification: {decision['justification']}")

@test("Memory write - large size constrained (quarantine)")
async def test_memory_write_constrained():
    """Verify large memory write is constrained to quarantine."""
    reg = client.post("/v1/register", json={
        "schema_version": SCHEMA_VERSION,
        "adapter_type": "test",
        "host_config": {"host_type": "openclaw", "namespace": "test"}
    }).json()
    adapter_id = reg["adapter_id"]
    
    payload = {
        "schema_version": SCHEMA_VERSION,
        "adapter_id": adapter_id,
        "host_config": {"host_type": "openclaw", "namespace": "test"},
        "proposal": {
            "proposal_id": "prop-mem-large",
            "timestamp": 1234567890,
            "action_type": "memory_write",
            "action_params": {
                "namespace": "sessions/large",
                "size_bytes": 15_000_000,  # 15MB - exceeds threshold
                "ttl": None,
                "sensitivity_hint": "medium",
                "content_hash": "b" * 64,
                "context_refs": ["session-large"],
                "operation": "update"
            },
            "context_refs": ["session-large"],
            "risk_tier": "medium",
            "priority": 0
        },
        "context": {"turn_number": 0},
        "capacity_signals": {}
    }
    
    response = client.post("/v1/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    decision = data["decision"]
    
    assert decision["decision"] == "CONSTRAIN"
    assert decision["constraint"] is not None
    assert decision["constraint"]["type"] == "quarantine_namespace"
    assert "_quarantine_" in decision["constraint"]["params"]["target_namespace"]
    print(f"  Decision: {decision['decision']}")
    print(f"  Constraint: {decision['constraint']['type']}")
    print(f"  Justification: {decision['justification']}")

@test("Memory write - high sensitivity allowed (high confidence)")
async def test_memory_high_sensitivity_allowed():
    """Verify high sensitivity with high confidence (>=0.8) is allowed."""
    reg = client.post("/v1/register", json={
        "schema_version": SCHEMA_VERSION,
        "adapter_type": "test",
        "host_config": {"host_type": "openclaw", "namespace": "test"}
    }).json()
    adapter_id = reg["adapter_id"]
    
    payload = {
        "schema_version": SCHEMA_VERSION,
        "adapter_id": adapter_id,
        "host_config": {"host_type": "openclaw", "namespace": "test"},
        "proposal": {
            "proposal_id": "prop-mem-high",
            "timestamp": 1234567890,
            "action_type": "memory_write",
            "action_params": {
                "namespace": "sensitive/vault",
                "size_bytes": 100,
                "ttl": None,
                "sensitivity_hint": "high",
                "content_hash": "c" * 64,
                "context_refs": ["session-vault"],
                "operation": "update"
            },
            "context_refs": ["session-vault"],
            "risk_tier": "high",
            "priority": 0
        },
        "context": {"turn_number": 0},
        "capacity_signals": {}
    }
    
    response = client.post("/v1/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    decision = data["decision"]
    
    # High sensitivity with default confidence (0.85 >= 0.8 threshold) → ALLOW
    # Note: Would be CONSTRAIN if confidence < 0.8
    assert decision["decision"] == "ALLOW"
    print(f"  Decision: {decision['decision']}")
    print(f"  Confidence: {decision['confidence']:.2f} (>= 0.8 threshold)")

@test("Event type enum validation")
async def test_event_type_enum():
    """Verify event type enum has all 19 required types."""
    required_events = [
        "adapter_registered",
        "adapter_disconnected",
        "proposal_received",
        "proposal_enacted",
        "proposal_expired",
        "proposal_revoked",
        "decision_made",
        "decision_rejected",
        "action_allowed",
        "action_blocked",
        "action_constrained",
        "action_deferred",
        "action_audited",
        "errors",
        "constraint_failed",
        "cgf_unreachable",
        "evaluate_timeout",
        "outcome_logged",
        "side_effect_reported"
    ]
    
    for event_name in required_events:
        exists = hasattr(HostEventType, event_name.upper())
        assert exists, f"Missing event type: {event_name}"
    
    print(f"  All {len(required_events)} event types present")

@test("Event required fields validation")
async def test_event_validation():
    """Verify event validation enforces required fields."""
    # ACTION_BLOCKED requires: decision_id, proposal_id, justification, reason_code
    required = get_event_required_fields(HostEventType.ACTION_BLOCKED)
    assert "decision_id" in required
    assert "proposal_id" in required
    assert "justification" in required
    assert "reason_code" in required
    print(f"  Required fields: {list(required.keys())}")

@test("Outcome reporting includes committed/quarantined")
async def test_outcome_report():
    """Verify outcome report includes v0.2 fields (committed, quarantined)."""
    reg = client.post("/v1/register", json={
        "schema_version": SCHEMA_VERSION,
        "adapter_type": "test",
        "host_config": {"host_type": "openclaw", "namespace": "test"}
    }).json()
    adapter_id = reg["adapter_id"]
    
    outcome = {
        "schema_version": SCHEMA_VERSION,
        "adapter_id": adapter_id,
        "proposal_id": "prop-xyz",
        "decision_id": "dec-abc",
        "executed": True,
        "executed_at": 1234567890,
        "duration_ms": 45.5,
        "success": True,
        "committed": True,
        "quarantined": False,
        "result_summary": "Write completed"
    }
    
    response = client.post("/v1/outcomes/report", json=outcome)
    assert response.status_code == 200
    data = response.json()
    assert data["received"] is True
    print(f"  Outcome reported: committed={outcome['committed']}, quarantined={outcome['quarantined']}")

@test("Replay pack generation")
async def test_replay_pack():
    """Verify replay endpoint returns complete pack."""
    # First create a proposal
    reg = client.post("/v1/register", json={
        "schema_version": SCHEMA_VERSION,
        "adapter_type": "test",
        "host_config": {"host_type": "openclaw", "namespace": "test"}
    }).json()
    adapter_id = reg["adapter_id"]
    
    proposal_id = "prop-replay-test"
    
    # Submit proposal
    client.post("/v1/evaluate", json={
        "schema_version": SCHEMA_VERSION,
        "adapter_id": adapter_id,
        "host_config": {"host_type": "openclaw", "namespace": "test"},
        "proposal": {
            "proposal_id": proposal_id,
            "timestamp": 1234567890,
            "action_type": "memory_write",
            "action_params": {
                "namespace": "test",
                "size_bytes": 1000,
                "sensitivity_hint": "medium",
                "content_hash": "d" * 64,
                "context_refs": [],
                "operation": "update"
            },
            "context_refs": ["session-test"],
            "risk_tier": "medium",
            "priority": 0
        },
        "context": {},
        "capacity_signals": {}
    })
    
    # Get replay pack
    response = client.get(f"/v1/proposals/{proposal_id}/replay")
    assert response.status_code == 200
    data = response.json()
    
    assert data["schema_version"] == "0.2.0"
    assert "replay" in data
    replay = data["replay"]
    assert replay["proposal"]["proposal_id"] == proposal_id
    assert replay["decision"] is not None
    print(f"  Replay ID: {replay['replay_id']}")
    print(f"  Completeness: {replay['completeness']}")

# ============== RUNNER ==============

async def run_all_tests():
    print("\n" + "=" * 70)
    print(f"CGF v0.2 INTEGRATION TEST SUITE")
    print(f"Schema Version: {SCHEMA_VERSION}")
    print("=" * 70)
    
    tests = [
        test_schema_version(),
        test_server_version(),
        test_registration_schema(),
        test_tool_call_blocked(),
        test_tool_call_allowed(),
        test_memory_write_allowed(),
        test_memory_write_constrained(),
        test_memory_high_sensitivity_allowed(),
        test_event_type_enum(),
        test_event_validation(),
        test_outcome_report(),
        test_replay_pack(),
    ]
    
    for t in tests:
        await t
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, p, _ in TEST_RESULTS if p)
    failed = sum(1 for _, p, _ in TEST_RESULTS if not p)
    
    for name, passed_status, error in TEST_RESULTS:
        status = "✅ PASS" if passed_status else "❌ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"      {error}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    return failed == 0

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
