#!/usr/bin/env python3
"""
replay_governance_timeline.py - Deterministic replay of governance proposals.

Reads JSONL logs and reconstructs a single proposal's timeline.
Generates ReplayPack with proposal + decision + outcome + event timeline.

Usage:
    python replay_governance_timeline.py --proposal-id prop-xxx
    python replay_governance_timeline.py --proposal-id prop-xxx --output replay.json
"""

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from cgf_schemas_v02 import (
    SCHEMA_VERSION,
    HostProposal,
    CGFDecision,
    HostOutcomeReport,
    HostEvent,
    ReplayPack
)

# ============== CONFIGURATION ==============

DEFAULT_DATA_DIR = Path("./cgf_data")

# ============== REPLAY ==============

class TimelineReplay:
    """Reconstruct governance timeline from JSONL logs."""
    
    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        self.proposals: Dict[str, dict] = {}
        self.decisions: Dict[str, dict] = {}
        self.outcomes: Dict[str, dict] = {}
        self.events: List[dict] = []
        
        self._load_all()
    
    def _load_jsonl(self, filename: str) -> List[dict]:
        """Load records from JSONL file."""
        path = self.data_dir / filename
        if not path.exists():
            return []
        
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records
    
    def _load_all(self):
        """Load all log files."""
        # Load proposals
        for record in self._load_jsonl("proposals.jsonl"):
            if "proposal_id" in record:
                self.proposals[record["proposal_id"]] = record
        
        # Load decisions
        for record in self._load_jsonl("decisions.jsonl"):
            if "decision_id" in record and "proposal_id" in record:
                self.decisions[record["proposal_id"]] = record
        
        # Load outcomes
        for record in self._load_jsonl("outcomes.jsonl"):
            if "proposal_id" in record:
                self.outcomes[record["proposal_id"]] = record
        
        # Load events (all indexed for search)
        self.events = self._load_jsonl("events.jsonl")
    
    def find_events_for_proposal(self, proposal_id: str) -> List[dict]:
        """Find all events relevant to a proposal, sorted chronologically."""
        events = []
        for event in self.events:
            if event.get("proposal_id") == proposal_id:
                events.append(event)
            elif event.get("payload", {}).get("proposal_id") == proposal_id:
                events.append(event)
        return sorted(events, key=lambda e: e.get("timestamp", 0))
    
    def reconstruct_timeline(self, proposal_id: str) -> Optional[ReplayPack]:
        """
        Reconstruct full timeline for a proposal.
        
        Returns ReplayPack with:
        - proposal: Original proposal
        - decision: CGF decision (if available)
        - outcome: Execution outcome (if available)
        - events: Chronological event list
        - source_files: Pointers to source log files
        """
        # Get proposal
        proposal_data = self.proposals.get(proposal_id)
        if not proposal_data:
            print(f"ERROR: Proposal {proposal_id} not found in {self.data_dir}/proposals.jsonl")
            return None
        
        # Reconstruct proposal
        # Filter to only valid HostProposal fields
        proposal_fields = {k: v for k, v in proposal_data.items() 
                          if k in HostProposal.__annotations__}
        try:
            proposal = HostProposal(**proposal_fields)
        except Exception as e:
            print(f"ERROR: Failed to reconstruct proposal: {e}")
            proposal = None
        
        # Get decision
        decision_data = self.decisions.get(proposal_id)
        decision = None
        if decision_data:
            decision_fields = {k: v for k, v in decision_data.items()
                             if k in CGFDecision.__annotations__}
            try:
                decision = CGFDecision(**decision_fields)
            except Exception as e:
                print(f"WARNING: Failed to reconstruct decision: {e}")
        
        # Get outcome
        outcome_data = self.outcomes.get(proposal_id)
        outcome = None
        if outcome_data:
            outcome_fields = {k: v for k, v in outcome_data.items()
                            if k in HostOutcomeReport.__annotations__}
            try:
                outcome = HostOutcomeReport(**outcome_fields)
            except Exception as e:
                print(f"WARNING: Failed to reconstruct outcome: {e}")
        
        # Get events
        event_records = self.find_events_for_proposal(proposal_id)
        events = []
        for er in event_records:
            event_fields = {k: v for k, v in er.items()
                          if k in HostEvent.__annotations__}
            try:
                events.append(HostEvent(**event_fields))
            except Exception as e:
                continue  # Skip malformed events
        
        # Determine completeness
        if outcome:
            completeness = "full"
        elif decision:
            completeness = "decision-only"
        else:
            completeness = "partial"
        
        # Build replay pack
        replay = ReplayPack(
            schema_version=SCHEMA_VERSION,
            replay_id=f"replay-{uuid.uuid4().hex[:12]}",
            created_at=datetime.now().timestamp(),
            proposal=proposal,
            decision=decision,
            outcome=outcome,
            events=events,
            source_files={
                "proposals": str(self.data_dir / "proposals.jsonl"),
                "decisions": str(self.data_dir / "decisions.jsonl"),
                "outcomes": str(self.data_dir / "outcomes.jsonl"),
                "events": str(self.data_dir / "events.jsonl"),
            },
            replay_version="1.0",
            completeness=completeness
        )
        
        return replay
    
    def print_timeline(self, replay: ReplayPack):
        """Print human-readable timeline."""
        print("\n" + "=" * 70)
        print(f"GOVERNANCE TIMELINE: {replay.proposal.proposal_id}")
        print("=" * 70)
        
        print(f"\nReplay ID: {replay.replay_id}")
        print(f"Created: {datetime.fromtimestamp(replay.created_at).isoformat()}")
        print(f"Completeness: {replay.completeness}")
        print(f"Schema Version: {replay.schema_version}")
        
        # Proposal
        print(f"\n{'─' * 70}")
        print("PROPOSAL")
        print('─' * 70)
        p = replay.proposal
        print(f"  ID: {p.proposal_id}")
        print(f"  Timestamp: {datetime.fromtimestamp(p.timestamp).isoformat()}")
        print(f"  Action Type: {p.action_type}")
        print(f"  Risk Tier: {p.risk_tier}")
        print(f"  Action Params Keys: {list(p.action_params.keys())}")
        if "tool_name" in p.action_params:
            print(f"  Tool Name: {p.action_params['tool_name']}")
        if "namespace" in p.action_params:
            print(f"  Namespace: {p.action_params['namespace']}")
            print(f"  Size: {p.action_params.get('size_bytes', 'unknown')} bytes")
            print(f"  Sensitivity: {p.action_params.get('sensitivity_hint', 'unknown')}")
        
        # Decision
        if replay.decision:
            print(f"\n{'─' * 70}")
            print("DECISION")
            print('─' * 70)
            d = replay.decision
            print(f"  ID: {d.decision_id}")
            print(f"  Decision: {d.decision.value}")
            print(f"  Confidence: {d.confidence:.2f}")
            print(f"  Justification: {d.justification}")
            if d.reason_code:
                print(f"  Reason Code: {d.reason_code}")
            if d.constraint:
                print(f"  Constraint: {d.constraint.type}")
                print(f"    Params: {d.constraint.params}")
        else:
            print(f"\n{'─' * 70}")
            print("DECISION: Not recorded")
        
        # Outcome
        if replay.outcome:
            print(f"\n{'─' * 70}")
            print("OUTCOME")
            print('─' * 70)
            o = replay.outcome
            print(f"  Executed: {o.executed}")
            print(f"  Success: {o.success}")
            if o.committed is not None:
                print(f"  Committed: {o.committed}")
            if o.quarantined is not None:
                print(f"  Quarantined: {o.quarantined}")
            print(f"  Duration: {o.duration_ms:.2f} ms")
        else:
            print(f"\n{'─' * 70}")
            print("OUTCOME: Not recorded")
        
        # Events
        print(f"\n{'─' * 70}")
        print(f"EVENT TIMELINE ({len(replay.events)} events)")
        print('─' * 70)
        for i, event in enumerate(replay.events, 1):
            ts = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S.%f")[:-3]
            print(f"  {i}. [{ts}] {event.event_type.value}")
            if event.payload:
                # Show key fields
                keys = list(event.payload.keys())[:3]
                payload_str = ", ".join(f"{k}={event.payload[k]}" for k in keys)
                if keys:
                    print(f"       {payload_str}")
        
        print("\n" + "=" * 70)
    
    def export_replay(self, replay: ReplayPack, output_path: Path):
        """Export ReplayPack to JSON file."""
        with open(output_path, "w") as f:
            f.write(replay.model_dump_json(indent=2))
        print(f"\nReplay exported to: {output_path}")
    
    def list_proposals(self, limit: int = 20) -> List[str]:
        """List recent proposal IDs."""
        sorted_proposals = sorted(
            self.proposals.items(),
            key=lambda x: x[1].get("timestamp", 0),
            reverse=True
        )
        return [pid for pid, _ in sorted_proposals[:limit]]

# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser(
        description="Replay governance timeline for a proposal"
    )
    parser.add_argument(
        "--proposal-id",
        help="Proposal ID to replay (or 'latest' for most recent)"
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Data directory with JSONL logs (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent proposals"
    )
    parser.add_argument(
        "--output",
        help="Export replay to JSON file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output replay as JSON instead of human-readable"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    replay_engine = TimelineReplay(data_dir)
    
    if args.list:
        print(f"Recent proposals (from {data_dir}):")
        for pid in replay_engine.list_proposals():
            p = replay_engine.proposals[pid]
            action = p.get("action_type", "unknown")
            risk = p.get("risk_tier", "unknown")
            ts = datetime.fromtimestamp(p.get("timestamp", 0)).isoformat()
            print(f"  {pid} [{action}/{risk}] {ts}")
        return
    
    if not args.proposal_id:
        print("ERROR: --proposal-id required (or use --list)")
        return
    
    # Handle 'latest' special case
    proposal_id = args.proposal_id
    if proposal_id == "latest":
        recent = replay_engine.list_proposals(1)
        if recent:
            proposal_id = recent[0]
            print(f"Using latest proposal: {proposal_id}")
        else:
            print("ERROR: No proposals found")
            return
    
    # Reconstruct timeline
    replay = replay_engine.reconstruct_timeline(proposal_id)
    
    if not replay:
        print(f"ERROR: Failed to reconstruct timeline for {proposal_id}")
        return
    
    # Output
    if args.json:
        print(replay.model_dump_json(indent=2))
    else:
        replay_engine.print_timeline(replay)
    
    if args.output:
        replay_engine.export_replay(replay, Path(args.output))

if __name__ == "__main__":
    main()
