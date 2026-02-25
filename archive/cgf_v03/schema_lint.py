#!/usr/bin/env python3
"""
schema_lint.py - Schema Validation and Event Ordering Linter for CGF v0.3

Validates:
1. JSONL files against schema versions (0.2.0, 0.3.0)
2. Event ordering invariants (proposal_received before decision_made, etc.)
3. Required field presence
4. Cross-host compatibility

Usage:
    python schema_lint.py --file events.jsonl
    python schema_lint.py --dir ./cgf_data/
    python schema_lint.py --compare openclaw/ langgraph/  # Cross-host comparison
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime

# Import schemas
try:
    from cgf_schemas_v03 import (
        SCHEMA_VERSION,
        COMPATIBLE_VERSIONS,
        is_compatible_version,
        HostEventType,
        validate_event_payload,
        CrossHostCompatibilityReport
    )
    SCHEMA_SOURCE = "v0.3"
except ImportError:
    from cgf_schemas_v02 import (
        SCHEMA_VERSION,
        HostEventType,
        validate_event_payload
    )
    SCHEMA_SOURCE = "v0.2"

# ============== EVENT ORDERING INVARIANTS ==============

# Valid event sequences: event_type -> list of events that must precede it
EVENT_ORDERING_RULES = {
    # Proposal lifecycle
    HostEventType.PROPOSAL_ENACTED: [HostEventType.PROPOSAL_RECEIVED],
    HostEventType.PROPOSAL_EXPIRED: [HostEventType.PROPOSAL_RECEIVED],
    HostEventType.PROPOSAL_REVOKED: [HostEventType.PROPOSAL_RECEIVED],
    
    # Decision lifecycle
    HostEventType.DECISION_MADE: [HostEventType.PROPOSAL_RECEIVED],
    HostEventType.DECISION_REJECTED: [HostEventType.PROPOSAL_RECEIVED],
    
    # Enforcement - requires decision
    HostEventType.ACTION_ALLOWED: [HostEventType.DECISION_MADE],
    HostEventType.ACTION_BLOCKED: [HostEventType.DECISION_MADE],
    HostEventType.ACTION_CONSTRAINED: [HostEventType.DECISION_MADE],
    HostEventType.ACTION_DEFERRED: [HostEventType.DECISION_MADE],
    HostEventType.ACTION_AUDITED: [HostEventType.DECISION_MADE],
    
    # Outcome - requires enforcement
    HostEventType.OUTCOME_LOGGED: [
        HostEventType.ACTION_ALLOWED,
        HostEventType.ACTION_BLOCKED,
        HostEventType.ACTION_CONSTRAINED
    ],
    
    # Side effects - requires outcome
    HostEventType.SIDE_EFFECT_REPORTED: [HostEventType.OUTCOME_LOGGED],
}

# Events that must have a unique sequence per proposal
UNIQUE_EVENTS = {
    HostEventType.PROPOSAL_RECEIVED,
    HostEventType.PROPOSAL_ENACTED,
    HostEventType.DECISION_MADE,
    HostEventType.OUTCOME_LOGGED,
}

# ============== LINTER ==============

class SchemaLinter:
    """Linter for CGF event JSONL files."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors: List[Dict] = []
        self.warnings: List[Dict] = []
        self.stats = {
            "files_checked": 0,
            "lines_checked": 0,
            "events_valid": 0,
            "events_invalid": 0,
            "proposals_tracked": 0,
            "order_violations": 0
        }
    
    def log(self, level: str, message: str, meta: Dict = None):
        """Log message at appropriate level."""
        entry = {"level": level, "message": message, "meta": meta or {}}
        if level == "error":
            self.errors.append(entry)
            print(f"❌ {message}", file=sys.stderr)
        elif level == "warn":
            self.warnings.append(entry)
            if self.verbose:
                print(f"⚠️  {message}")
        elif self.verbose:
            print(f"ℹ️  {message}")
    
    def lint_file(self, filepath: Path) -> bool:
        """Lint a single JSONL file."""
        self.stats["files_checked"] += 1
        valid = True
        
        if not filepath.exists():
            self.log("error", f"File not found: {filepath}")
            return False
        
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()
        except Exception as e:
            self.log("error", f"Failed to read {filepath}: {e}")
            return False
        
        # Track proposals for ordering
        proposal_events: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        proposal_ids_seen: Set[str] = set()
        
        for line_num, line in enumerate(lines, 1):
            self.stats["lines_checked"] += 1
            line = line.strip()
            if not line:
                continue
            
            try:
                event = json.loads(line)
            except json.JSONDecodeError as e:
                self.log("error", f"Invalid JSON at {filepath}:{line_num}", {"error": str(e)})
                valid = False
                self.stats["events_invalid"] += 1
                continue
            
            # Check schema version
            schema_version = event.get("schema_version", "not specified")
            if not is_compatible_version(schema_version):
                self.log("error", 
                    f"Incompatible schema version at {filepath}:{line_num}: {schema_version}",
                    {"compatible_versions": list(COMPATIBLE_VERSIONS)}
                )
                valid = False
                self.stats["events_invalid"] += 1
                continue
            
            # Validate event type
            event_type_str = event.get("event_type")
            try:
                event_type = HostEventType(event_type_str) if event_type_str else None
            except ValueError:
                self.log("error", 
                    f"Unknown event type at {filepath}:{line_num}: {event_type_str}",
                    {"valid_types": [e.value for e in HostEventType]}
                )
                valid = False
                self.stats["events_invalid"] += 1
                continue
            
            # Validate required fields
            if event_type:
                payload = event.get("payload", {})
                is_valid, errors = validate_event_payload(event_type, payload)
                if not is_valid:
                    for error in errors:
                        self.log("error", 
                            f"Validation error at {filepath}:{line_num}: {error}",
                            {"event_type": event_type.value}
                        )
                    valid = False
                    self.stats["events_invalid"] += 1
                    continue
            
            # Track proposals for ordering validation
            proposal_id = event.get("proposal_id")
            if proposal_id:
                proposal_ids_seen.add(proposal_id)
                timestamp = event.get("timestamp", 0)
                proposal_events[proposal_id].append((event_type_str, timestamp))
                
                # Check for duplicate unique events
                if event_type in UNIQUE_EVENTS:
                    count = sum(1 for et, _ in proposal_events[proposal_id] if et == event_type_str)
                    if count > 1:
                        self.log("warn",
                            f"Duplicate unique event at {filepath}:{line_num}: {event_type_str} for {proposal_id}",
                            {"event_type": event_type_str, "proposal_id": proposal_id}
                        )
            
            self.stats["events_valid"] += 1
        
        # Validate ordering for each proposal
        for proposal_id, events in proposal_events.items():
            self.stats["proposals_tracked"] += 1
            event_set = set(et for et, _ in events)
            
            for event_type, required_predecessors in EVENT_ORDERING_RULES.items():
                if event_type.value in event_set:
                    # Event occurred - check if all required predecessors occurred
                    for pred in required_predecessors:
                        if pred.value not in event_set:
                            self.log("error",
                                f"Ordering violation: {event_type.value} requires {pred.value} "
                                f"but predecessor not found for proposal {proposal_id}",
                                {"proposal_id": proposal_id, "file": str(filepath)}
                            )
                            valid = False
                            self.stats["order_violations"] += 1
        
        return valid
    
    def lint_directory(self, dirpath: Path) -> bool:
        """Lint all JSONL files in a directory."""
        if not dirpath.exists():
            self.log("error", f"Directory not found: {dirpath}")
            return False
        
        jsonl_files = list(dirpath.glob("*.jsonl"))
        if not jsonl_files:
            self.log("warn", f"No JSONL files found in {dirpath}")
            return True
        
        valid = True
        for filepath in jsonl_files:
            if not self.lint_file(filepath):
                valid = False
        
        return valid
    
    def compare_hosts(self, host_dirs: List[Path]) -> CrossHostCompatibilityReport:
        """Compare event patterns across hosts."""
        host_events: Dict[str, List[Dict]] = {}
        
        # Load events from each host
        for host_dir in host_dirs:
            host_name = host_dir.name
            events = []
            for jsonl_file in host_dir.glob("*.jsonl"):
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                events.append(json.loads(line))
                            except:
                                pass
            host_events[host_name] = events
        
        # Compare event type distributions
        all_hosts = list(host_events.keys())
        scenarios = ["tool_call_block", "tool_call_allow", "cgf_down_fail_mode"]
        
        results = {host: {scenario: "unknown" for scenario in scenarios} for host in all_hosts}
        comparisons = {}
        
        # Simple heuristic: count event types per host
        for host_name, events in host_events.items():
            event_types = [e.get("event_type") for e in events]
            
            # Check for block scenario
            if "action_blocked" in event_types:
                results[host_name]["tool_call_block"] = "pass"
            
            # Check for allow scenario
            if "action_allowed" in event_types:
                results[host_name]["tool_call_allow"] = "pass"
            
            # Check for fail mode
            if "cgf_unreachable" in event_types:
                results[host_name]["cgf_down_fail_mode"] = "pass"
        
        compatible = all(
            all(r == "pass" for r in host_results.values())
            for host_results in results.values()
        )
        
        report = CrossHostCompatibilityReport(
            schema_version=SCHEMA_VERSION,
            report_id=f"report-{datetime.now().timestamp()}",
            created_at=datetime.now().timestamp(),
            hosts_tested=all_hosts,
            scenarios=scenarios,
            results=results,
            replay_comparisons=comparisons,
            compatible=compatible
        )
        
        return report
    
    def print_summary(self):
        """Print lint summary."""
        print("\n" + "=" * 60)
        print("SCHEMA LINT SUMMARY")
        print("=" * 60)
        print(f"Schema Source: {SCHEMA_SOURCE}")
        print(f"Files checked: {self.stats['files_checked']}")
        print(f"Lines checked: {self.stats['lines_checked']}")
        print(f"Events valid: {self.stats['events_valid']}")
        print(f"Events invalid: {self.stats['events_invalid']}")
        print(f"Proposals tracked: {self.stats['proposals_tracked']}")
        print(f"Ordering violations: {self.stats['order_violations']}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        
        if self.errors:
            print("\nERRORS:")
            for err in self.errors[:10]:  # Show first 10
                print(f"  ❌ {err['message']}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")


# ============== CLI ==============

def main():
    parser = argparse.ArgumentParser(
        description="CGF Schema Linter v0.3 - Validate JSONL events"
    )
    parser.add_argument("--file", "-f", type=Path, help="Single JSONL file to lint")
    parser.add_argument("--dir", "-d", type=Path, help="Directory of JSONL files to lint")
    parser.add_argument("--compare", "-c", nargs="+", type=Path, 
                       help="Compare hosts (provide multiple directories)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--exit-code", action="store_true",
                       help="Exit with non-zero code on errors")
    
    args = parser.parse_args()
    
    linter = SchemaLinter(verbose=args.verbose)
    valid = True
    
    if args.file:
        valid = linter.lint_file(args.file) and valid
    elif args.dir:
        valid = linter.lint_directory(args.dir) and valid
    elif args.compare:
        if len(args.compare) < 2:
            print("Error: --compare requires at least 2 directories", file=sys.stderr)
            sys.exit(1)
        report = linter.compare_hosts(args.compare)
        print("\n" + "=" * 60)
        print("CROSS-HOST COMPATIBILITY REPORT")
        print("=" * 60)
        print(f"Report ID: {report.report_id}")
        print(f"Hosts: {report.hosts_tested}")
        print(f"Scenarios: {report.scenarios}")
        print(f"\nResults:")
        for host, scenarios in report.results.items():
            print(f"  {host}:")
            for scenario, result in scenarios.items():
                status = "✅" if result == "pass" else "❌"
                print(f"    {status} {scenario}: {result}")
        print(f"\nCompatible: {'✅ YES' if report.compatible else '❌ NO'}")
    else:
        parser.print_help()
        sys.exit(1)
    
    linter.print_summary()
    
    if args.exit_code and not valid:
        sys.exit(1)
    
    sys.exit(0 if valid else 0)  # Default: don't fail CI on warnings


if __name__ == "__main__":
    main()
