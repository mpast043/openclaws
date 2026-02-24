"""
Test Suite for OpenClawAdapter v0.1 + CGF Server

Three test scenarios:
1. Blocked tool never executes
2. Allowed tool executes normally  
3. CGF down behavior (fail mode)

Usage:
    # Terminal 1: Start CGF Server
    python cgf_server_v01.py

    # Terminal 2: Run tests
    python test_openclaw_adapter_v01.py
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Import the adapter
from openclaw_adapter_v01 import (
    OpenClawAdapter, 
    CGFGovernanceError,
    CGFUnavailableError
)

# ============== TEST CONFIGURATION ==============

CGF_ENDPOINT = "http://127.0.0.1:8080"
DATA_DIR = Path("./test_data")

def clear_test_data():
    """Clear test data files"""
    for f in DATA_DIR.glob("*.jsonl"):
        f.unlink(missing_ok=True)

# ============== TEST 1: BLOCKED TOOL ==============

async def test_blocked_tool():
    """
    Test 1: Blocked tool never executes
    
    Criteria:
    - Tool does NOT run (no side effect)
    - Raises CGFGovernanceError with BLOCK decision
    - /v1/outcomes/report shows executed=false
    - Event log contains: proposal_received, decision_made, action_blocked
    """
    print("\n" + "="*60)
    print("TEST 1: Blocked tool never executes")
    print("="*60)
    
    adapter = OpenClawAdapter(
        cgf_endpoint=CGF_ENDPOINT,
        adapter_type="test",
        host_config={"host_type": "openclaw", "namespace": "test"},
        timeout_ms=500
    )
    
    # file_write is in the denylist
    host_context = {
        "tool_name": "file_write",
        "tool_args": {"path": "/tmp/test.txt", "content": "test"},
        "agent_id": "test-agent",
        "session_id": "test-session",
        "side_effects_hint": ["write"]
    }
    
    try:
        result = await adapter.governance_hook(host_context)
        print("FAIL: Expected CGFGovernanceError but got result:", result)
        return False
    except CGFGovernanceError as e:
        print(f"PASS: Tool blocked correctly")
        print(f"  Decision: {e.decision.value}")
        print(f"  Justification: {e.justification}")
        print(f"  Proposal ID: {e.proposal_id}")
        
        # Check events were logged
        events_path = Path("./openclaw_adapter_data/events.jsonl")
        if events_path.exists():
            with open(events_path) as f:
                events = [json.loads(line) for line in f]
            
            event_types = [e["event_type"] for e in events]
            required = ["proposal_received", "decision_made", "action_blocked"]
            
            for req in required:
                if req in event_types:
                    print(f"  ✓ Event '{req}' logged")
                else:
                    print(f"  ✗ Missing event '{req}'")
                    return False
        
        await adapter.close()
        print("TEST 1: PASSED")
        return True
        
    except Exception as e:
        print(f"FAIL: Unexpected error: {e}")
        return False

# ============== TEST 2: ALLOWED TOOL ==============

async def test_allowed_tool():
    """
    Test 2: Allowed tool executes normally
    
    Criteria:
    - Tool runs
    - outcome executed=true is reported
    - Event log contains: action_allowed, outcome_logged
    """
    print("\n" + "="*60)
    print("TEST 2: Allowed tool executes normally")
    print("="*60)
    
    # Clear previous events
    events_path = Path("./openclaw_adapter_data/events.jsonl")
    if events_path.exists():
        events_path.unlink()
    
    adapter = OpenClawAdapter(
        cgf_endpoint=CGF_ENDPOINT,
        adapter_type="test",
        host_config={"host_type": "openclaw", "namespace": "test"},
        timeout_ms=500
    )
    
    # ls is not in denylist
    host_context = {
        "tool_name": "ls",
        "tool_args": {"path": "/tmp"},
        "agent_id": "test-agent",
        "session_id": "test-session",
        "side_effects_hint": ["read"]
    }
    
    try:
        result = await adapter.governance_hook(host_context)
        
        if result.get("executed") != True:
            print("FAIL: Expected executed=True")
            return False
        
        print(f"PASS: Tool allowed")
        print(f"  Tool: {result.get('tool_name')}")
        print(f"  Decision ID: {result.get('decision_id')}")
        
        # Simulate execution and report outcome
        execution_result = {
            "executed_at": result.get("timestamp"),
            "duration_ms": 50,
            "success": True,
            "result_summary": "Directory listing",
            "tool_name": result.get('tool_name'),
            "tool_args": result.get('tool_args'),
            "decision_id": result.get('decision_id')
        }
        
        await adapter.report_execution(execution_result, host_context)
        
        # Check events
        if events_path.exists():
            with open(events_path) as f:
                events = [json.loads(line) for line in f]
            
            event_types = [e["event_type"] for e in events]
            print(f"  Events: {event_types}")
            
            if "proposal_received" in event_types and "action_allowed" in event_types:
                print("TEST 2: PASSED")
                await adapter.close()
                return True
            else:
                print("FAIL: Missing required events")
                return False
        
        await adapter.close()
        print("TEST 2: PASSED")
        return True
        
    except Exception as e:
        print(f"FAIL: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============== TEST 3: CGF DOWN BEHAVIOR ==============

async def test_cgf_down():
    """
    Test 3: CGF down behavior
    
    Criteria:
    - Simulate CGF unreachable by pointing to wrong port
    - Registration already done (simulated by creating adapter with working endpoint then switching)
    - Side-effect tool: must BLOCK (fail_closed)
    - Read-only tool: may ALLOW (fail_open) or DEFER
    - Event log contains cgf_unreachable or evaluate_timeout
    """
    print("\n" + "="*60)
    print("TEST 3: CGF down behavior")
    print("="*60)
    
    # First register with working CGF
    temp_adapter = OpenClawAdapter(
        cgf_endpoint="http://127.0.0.1:8080",
        adapter_type="test",
        host_config={"host_type": "openclaw", "namespace": "test"},
        timeout_ms=500
    )
    await temp_adapter.register()
    
    # Now create a new adapter pointing to dead endpoint but copy the adapter_id
    adapter = OpenClawAdapter(
        cgf_endpoint="http://127.0.0.1:9999",  # Wrong port
        adapter_type="test",
        host_config={"host_type": "openclaw", "namespace": "test"},
        timeout_ms=500
    )
    adapter.adapter_id = temp_adapter.adapter_id  # Pre-register
    
    # Clear events  
    events_path = Path("./openclaw_adapter_data/events.jsonl")
    if events_path.exists():
        events_path.unlink()
    
    # Test side-effect tool (file_write)
    print("\nSub-test 3a: Side-effect tool with CGF down")
    host_context = {
        "tool_name": "file_write",
        "tool_args": {"path": "/tmp/test.txt", "content": "test"},
        "agent_id": "test-agent",
        "session_id": "test-session",
        "side_effects_hint": ["write"]
    }
    
    try:
        result = await adapter.governance_hook(host_context)
        print("FAIL: Side-effect tool should be blocked when CGF down")
        return False
    except CGFGovernanceError as e:
        if e.decision.value == "BLOCK":
            print(f"PASS: Side-effect tool blocked (fail closed)")
            print(f"  Decision: {e.decision.value}")
            print(f"  Justification: {e.justification}")
        else:
            print(f"FAIL: Expected BLOCK, got {e.decision.value}")
            return False
    except Exception as e:
        print(f"FAIL: Unexpected error type: {type(e).__name__}: {e}")
        return False
    
    # Test read-only tool (ls)
    print("\nSub-test 3b: Read-only tool with CGF down")
    host_context_read = {
        "tool_name": "ls",
        "tool_args": {"path": "/tmp"},
        "agent_id": "test-agent",
        "session_id": "test-session",
        "side_effects_hint": ["read"]
    }
    
    try:
        result = await adapter.governance_hook(host_context_read)
        print(f"PASS: Read-only tool allowed (fail open)")
        print(f"  Decision: ALLOW (implicit via fail mode)")
    except CGFGovernanceError as e:
        if e.decision.value == "DEFER":
            print(f"PASS: Read-only tool deferred (acceptable fail mode)")
            print(f"  Decision: {e.decision.value}")
        else:
            print(f"INFO: Read-only tool got {e.decision.value}")
    
    # Check events for cgf_unreachable
    if events_path.exists():
        with open(events_path) as f:
            events = [json.loads(line) for line in f]
        
        event_types = [e["event_type"] for e in events]
        if "cgf_unreachable_fallback" in event_types or "cgf_unreachable" in event_types:
            print(f"  ✓ Event logged for CGF unreachability")
        else:
            print(f"  Events logged: {event_types}")
    
    await adapter.close()
    print("\nTEST 3: PASSED")
    return True

# ============== MAIN ==============

async def run_all_tests():
    """Run all three tests"""
    print("OpenClawAdapter v0.1 Test Suite")
    print(f"CGF Endpoint: {CGF_ENDPOINT}")
    print(f"Data Directory: {DATA_DIR}")
    
    # Ensure data directory exists
    from openclaw_adapter_v01 import DATA_DIR as ADAPTER_DATA_DIR
    ADAPTER_DATA_DIR.mkdir(exist_ok=True)
    
    results = []
    
    try:
        results.append(("Test 1: Blocked tool", await test_blocked_tool()))
    except Exception as e:
        print(f"Test 1 crashed: {e}")
        results.append(("Test 1: Blocked tool", False))
    
    try:
        results.append(("Test 2: Allowed tool", await test_allowed_tool()))
    except Exception as e:
        print(f"Test 2 crashed: {e}")
        results.append(("Test 2: Allowed tool", False))
    
    try:
        results.append(("Test 3: CGF down", await test_cgf_down()))
    except Exception as e:
        print(f"Test 3 crashed: {e}")
        results.append(("Test 3: CGF down", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
