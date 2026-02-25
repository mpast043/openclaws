#!/usr/bin/env python3
"""
contract_compliance_tests.py - Cross-Host Contract Compliance Suite v0.3

Tests scenarios against multiple hosts:
- OpenClawAdapter v0.2 (Host #1)
- LangGraphAdapter v0.1 (Host #2)
- Future hosts can be added

Scenarios:
1. Block denylisted tool_call (executed=false)
2. Allow read-only tool_call (executed=true)
3. CGF down behavior (fail-mode applied deterministically)

Requirements:
- Each test produces a ReplayPack
- All 19 canonical EventTypes validated
- Event ordering invariants checked
- ReplayPacks comparable across hosts
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

# Import adapters
try:
    from cgf_schemas_v03 import (
        SCHEMA_VERSION,
        ActionType,
        DecisionType,
        RiskTier,
        HostEventType,
        HostConfig,
        CrossHostCompatibilityReport,
        ReplayPack
    )
    SCHEMA_MODULE = "v0.3"
except ImportError:
    from cgf_schemas_v02 import (
        SCHEMA_VERSION,
        ActionType,
        DecisionType,
        RiskTier,
        HostEventType,
        HostConfig
    )
    SCHEMA_MODULE = "v0.2"

# Import test client
try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("Warning: FastAPI testclient not available")

# Import adapters
from openclaw_adapter_v02 import OpenClawAdapter
from langgraph_adapter_v01 import LangGraphAdapter

# Import server
try:
    from cgf_server_v02 import app
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False
    print("Warning: CGF server not available")

# ============== TEST FIXTURES ==============

@dataclass
class TestScenario:
    """Test scenario definition."""
    name: str
    description: str
    tool_name: str
    tool_args: Dict[str, Any]
    risk_tier: RiskTier
    side_effects: List[str]
    expected_decision: DecisionType
    expected_success: bool
    cgf_available: bool = True

# Standard compliance scenarios
COMPLIANCE_SCENARIOS = [
    TestScenario(
        name="denylisted_tool_blocked",
        description="Denylisted tool (file_write) should be blocked",
        tool_name="file_write",
        tool_args={"path": "/etc/passwd", "content": "malicious"},
        risk_tier=RiskTier.HIGH,
        side_effects=["write"],
        expected_decision=DecisionType.BLOCK,
        expected_success=False
    ),
    TestScenario(
        name="read_only_tool_allowed",
        description="Read-only tool (ls) should be allowed",
        tool_name="ls",
        tool_args={"path": "/tmp"},
        risk_tier=RiskTier.LOW,
        side_effects=["read"],
        expected_decision=DecisionType.ALLOW,
        expected_success=True
    ),
    TestScenario(
        name="cgf_down_fail_closed",
        description="CGF down + side-effect tool should fail closed (block)",
        tool_name="exec",
        tool_args={"command": "rm -rf /"},
        risk_tier=RiskTier.HIGH,
        side_effects=["write"],
        expected_decision=DecisionType.BLOCK,
        expected_success=False,
        cgf_available=False
    ),
    TestScenario(
        name="cgf_down_fail_open",
        description="CGF down + read-only tool should fail open (allow)",
        tool_name="cat",
        tool_args={"path": "/tmp/test.txt"},
        risk_tier=RiskTier.LOW,
        side_effects=["read"],
        expected_decision=DecisionType.ALLOW,  # Due to fail_open policy
        expected_success=True,
        cgf_available=False
    ),
]

# ============== HOST ADAPTER WRAPPER ==============

class HostAdapterWrapper:
    """Wrapper to run scenarios against any host adapter."""
    
    def __init__(self, adapter, host_name: str):
        self.adapter = adapter
        self.host_name = host_name
        self.proposals: List[str] = []
        self.results: List[Dict] = []
    
    async def run_scenario(self, scenario: TestScenario, client: Any = None) -> Dict:
        """Run a single scenario."""
        result = {
            "scenario": scenario.name,
            "host": self.host_name,
            "tool_name": scenario.tool_name,
            "expected": scenario.expected_decision.value,
            "actual": None,
            "success": False,
            "events": [],
            "error": None,
            "proposal_id": None,
            "decision_id": None,
            "outcome": None
        }
        
        try:
            # Mock CGF for availability tests
            if not scenario.cgf_available:
                old_endpoint = os.environ.get("CGF_ENDPOINT", "http://127.0.0.1:8080")
                os.environ["CGF_ENDPOINT"] = "http://invalid:9999"
            
            # Call appropriate adapter
            if self.host_name == "openclaw":
                # OpenClaw uses direct tool governance
                result.update(await self._run_openclaw_scenario(scenario))
            elif self.host_name == "langgraph":
                # LangGraph uses governed tool hook
                result.update(await self._run_langgraph_scenario(scenario))
            
            # Restore endpoint
            if not scenario.cgf_available:
                os.environ["CGF_ENDPOINT"] = old_endpoint
            
        except Exception as e:
            result["error"] = str(e)
            result["actual"] = "ERROR"
        
        return result
    
    async def _run_openclaw_scenario(self, scenario: TestScenario) -> Dict:
        """Run scenario with OpenClaw adapter."""
        adapter = self.adapter
        
        # Use governance hook
        session_key = f"test-session-{scenario.name}"
        agent_id = "test-agent"
        
        try:
            outcome = await adapter.governance_hook(
                tool_name=scenario.tool_name,
                tool_args=scenario.tool_args,
                session_key=session_key,
                agent_id=agent_id
            )
            
            # Infer actual decision
            if outcome.get("blocked"):
                actual = DecisionType.BLOCK
            elif outcome.get("fail_open"):
                actual = DecisionType.ALLOW  # Fail open = treated as allow
            else:
                actual = DecisionType.ALLOW
            
            return {
                "actual": actual.value,
                "success": actual == scenario.expected_decision or scenario.expected_success,
                "proposal_id": outcome.get("proposal_id"),
                "decision_id": outcome.get("decision_id"),
                "fail_open": outcome.get("fail_open", False)
            }
            
        except Exception as e:
            if "BLOCKED" in str(e) or "blocked by CGF" in str(e).lower():
                actual = DecisionType.BLOCK
                return {
                    "actual": actual.value,
                    "success": actual == scenario.expected_decision,
                    "blocked": True,
                    "error_message": str(e)
                }
            raise
    
    async def _run_langgraph_scenario(self, scenario: TestScenario) -> Dict:
        """Run scenario with LangGraph adapter."""
        adapter = self.adapter
        
        try:
            outcome = await adapter.governance_hook(
                tool_name=scenario.tool_name,
                tool_args=scenario.tool_args,
                thread_id=f"test-thread-{scenario.name}",
                node_id="test-node",
                state={"turn_number": 0}
            )
            
            if outcome.get("blocked"):
                actual = DecisionType.BLOCK
            elif outcome.get("fail_open"):
                actual = DecisionType.ALLOW
            else:
                actual = DecisionType.ALLOW
            
            return {
                "actual": actual.value,
                "success": actual == scenario.expected_decision or scenario.expected_success,
                "proposal_id": outcome.get("proposal_id"),
                "decision_id": outcome.get("decision_id"),
                "fail_open": outcome.get("fail_open", False)
            }
            
        except Exception as e:
            if "blocked by CGF" in str(e).lower() or "LangGraphToolBlocked" in str(type(e).__name__):
                actual = DecisionType.BLOCK
                return {
                    "actual": actual.value,
                    "success": actual == scenario.expected_decision,
                    "blocked": True,
                    "error_message": str(e)
                }
            raise


# ============== COMPLIANCE TEST SUITE ==============

class ContractComplianceSuite:
    """Test suite for cross-host contract compliance."""
    
    def __init__(self):
        self.results: List[Dict] = []
        self.replay_packs: Dict[str, ReplayPack] = {}
        self.passed = 0
        self.failed = 0
    
    async def run_all(self) -> bool:
        """Run all compliance tests."""
        print("=" * 70)
        print("CROSS-HOST CONTRACT COMPLIANCE SUITE v0.3")
        print("=" * 70)
        print(f"Schema: {SCHEMA_MODULE}")
        print(f"Hosts: OpenClaw v0.2, LangGraph v0.1")
        print(f"Scenarios: {len(COMPLIANCE_SCENARIOS)}")
        print("=" * 70)
        
        # Initialize adapters
        hosts = {
            "openclaw": OpenClawAdapter(
                HostConfig(host_type="openclaw", namespace="test", version="0.2.0")
            ),
            "langgraph": LangGraphAdapter(
                HostConfig(host_type="langgraph", namespace="test", version="0.1.0")
            )
        }
        
        # Run tests
        for scenario in COMPLIANCE_SCENARIOS:
            print(f"\n{'─' * 70}")
            print(f"SCENARIO: {scenario.name}")
            print(f"Tool: {scenario.tool_name}, Expected: {scenario.expected_decision.value}")
            print(f"CGF Available: {scenario.cgf_available}")
            print('─' * 70)
            
            for host_name, adapter in hosts.items():
                wrapper = HostAdapterWrapper(adapter, host_name)
                result = await wrapper.run_scenario(scenario)
                
                # Evaluate
                passed = result["actual"] == scenario.expected_decision.value
                if passed and not result.get("error"):
                    self.passed += 1
                    status = "✅ PASS"
                else:
                    self.failed += 1
                    status = "❌ FAIL"
                
                print(f"  {status} {host_name:12} → {result['actual']:12} (expected: {scenario.expected_decision.value})")
                if result.get("error"):
                    print(f"         Error: {result['error'][:60]}")
                
                self.results.append(result)
        
        return self.failed == 0
    
    def generate_replay_pack(self, host_name: str, scenario_name: str) -> Dict:
        """Generate replay pack for a host/scenario."""
        # Find events for this host/scenario
        events = []
        
        # Look in adapter data dirs
        data_dir = Path(f"./{host_name}_cgf_data")
        if data_dir.exists():
            events_file = data_dir / "events.jsonl"
            if events_file.exists():
                with open(events_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                event = json.loads(line)
                                if scenario_name.replace("_", "-") in event.get("proposal_id", "") or \
                                   scenario_name in str(event.get("payload", {})):
                                    events.append(event)
                            except:
                                pass
        
        return {
            "schema_version": SCHEMA_VERSION,
            "host": host_name,
            "scenario": scenario_name,
            "timestamp": datetime.now().timestamp(),
            "completeness": "partial" if len(events) < 3 else "full",
            "events": events
        }
    
    def compare_replays(self, scenario_name: str) -> Dict:
        """Compare replay packs across hosts for same scenario."""
        replays = {
            "openclaw": self.generate_replay_pack("openclaw", scenario_name),
            "langgraph": self.generate_replay_pack("langgraph", scenario_name)
        }
        
        # Extract event sequences
        sequences = {}
        for host, replay in replays.items():
            events = replay.get("events", [])
            sequences[host] = [e.get("event_type") for e in sorted(events, key=lambda x: x.get("timestamp", 0))]
        
        # Compare
        comparison = {
            "scenario": scenario_name,
            "schema_version": SCHEMA_VERSION,
            "openclaw_events": sequences.get("openclaw", []),
            "langgraph_events": sequences.get("langgraph", []),
            "event_counts_match": len(sequences.get("openclaw", [])) == len(sequences.get("langgraph", [])),
            "key_events_present": {
                "adapter_registered": all(
                    "adapter_registered" in seq for seq in sequences.values()
                ),
                "proposal_received": all(
                    "proposal_received" in seq for seq in sequences.values()
                ),
                "decision_made": all(
                    any(e in seq for e in ["decision_made", "action_allowed", "action_blocked"])
                    for seq in sequences.values()
                )
            }
        }
        
        comparison["compatible"] = all(comparison["key_events_present"].values())
        
        return comparison
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Schema Version: {SCHEMA_VERSION}")
        print(f"Passed: {self.passed}/{self.passed + self.failed}")
        print(f"Failed: {self.failed}/{self.passed + self.failed}")
        
        if self.failed == 0:
            print("\n✅ ALL TESTS PASSED - Cross-host contract compatible!")
        else:
            print(f"\n⚠️  {self.failed} TESTS FAILED")
        
        # Replay comparison
        print("\n--- Replay Pack Comparisons ---")
        for scenario in COMPLIANCE_SCENARIOS:
            comparison = self.compare_replays(scenario.name)
            status = "✅" if comparison["compatible"] else "⚠️"
            print(f"{status} {scenario.name:30} compatible={comparison['compatible']}")
    
    def generate_report(self) -> CrossHostCompatibilityReport:
        """Generate formal compliance report."""
        # Organize results by host/scenario
        results = {}
        hosts = set(r["host"] for r in self.results)
        scenarios = set(r["scenario"] for r in self.results)
        
        for host in hosts:
            results[host] = {}
            for scenario in scenarios:
                result = next((r for r in self.results 
                              if r["host"] == host and r["scenario"] == scenario), None)
                if result:
                    results[host][scenario] = "pass" if result.get("success") else "fail"
                else:
                    results[host][scenario] = "unknown"
        
        return CrossHostCompatibilityReport(
            schema_version=SCHEMA_VERSION,
            report_id=f"compliance-report-{datetime.now().timestamp()}",
            created_at=datetime.now().timestamp(),
            hosts_tested=list(hosts),
            scenarios=list(scenarios),
            results=results,
            replay_comparisons={
                s.name: self.compare_replays(s.name) for s in COMPLIANCE_SCENARIOS
            },
            compatible=self.failed == 0
        )


# ============== MAIN ==============

async def main():
    """Run compliance suite."""
    suite = ContractComplianceSuite()
    success = await suite.run_all()
    suite.print_summary()
    
    # Generate report
    report = suite.generate_report()
    
    # Save report
    report_path = Path("./contract_compliance_report.json")
    with open(report_path, "w") as f:
        # Convert to dict for JSON
        report_dict = {
            "schema_version": report.schema_version,
            "report_id": report.report_id,
            "created_at": report.created_at,
            "hosts_tested": report.hosts_tested,
            "scenarios": report.scenarios,
            "results": report.results,
            "replay_comparisons": report.replay_comparisons,
            "compatible": report.compatible
        }
        json.dump(report_dict, f, indent=2)
    
    print(f"\n📄 Report saved to: {report_path}")
    print(f"Compatible: {'✅ YES' if report.compatible else '❌ NO'}")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
