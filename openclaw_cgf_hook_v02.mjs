/**
 * openclaw_cgf_hook_v02.mjs - OpenClaw Runtime Hook for CGF Integration v0.2
 * 
 * Intercepts:
 * 1. Tool execution at handleToolsInvokeHttpRequest (gateway-cli-CIYEdmIv.js)
 * 2. Memory writes at updateSessionStore (pi-embedded-helpers-CMf7l1vP.js)
 * 
 * Architecture: Inline hook (non-proxy) — wraps invocation calls directly.
 * 
 * Changes v0.1 → v0.2:
 * - Schema version 0.2.0
 * - Memory write governance for session persistence
 * - Updated event types (canonical enum)
 */

import { createHash } from 'crypto';
import { writeFileSync, appendFileSync, existsSync, mkdirSync } from 'fs';
import { dirname } from 'path';

// ============== CONFIGURATION ==============

const CONFIG = {
  schemaVersion: '0.2.0',
  cgfEndpoint: process.env.CGF_ENDPOINT || 'http://127.0.0.1:8080',
  adapterType: 'openclaw',
  timeoutMs: 500,
  dataDir: './openclaw_cgf_data',
  logLevel: process.env.LOG_LEVEL || 'info'
};

// Ensure data directory exists
if (!existsSync(CONFIG.dataDir)) {
  mkdirSync(CONFIG.dataDir, { recursive: true });
}

// ============== STATE ==============

const state = {
  adapterId: null,
  registered: false,
  eventCount: 0,
  proposalCount: 0,
  session: null
};

// ============== LOGGING ==============

const logLevels = { debug: 0, info: 1, warn: 2, error: 3 };

function log(level, message, meta = {}) {
  if (logLevels[level] >= logLevels[CONFIG.logLevel]) {
    const entry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      ...meta
    };
    console.log(`[CGF-HOOK] [${level.toUpperCase()}] ${message}`, meta);
    
    // Persist to log file
    const logPath = `${CONFIG.dataDir}/hook.log`;
    appendFileSync(logPath, JSON.stringify(entry) + '\n');
  }
}

// ============== EVENT EMISSION ==============

const EVENT_TYPES = {
  // Adapter lifecycle
  ADAPTER_REGISTERED: 'adapter_registered',
  ADAPTER_DISCONNECTED: 'adapter_disconnected',
  
  // Proposal lifecycle
  PROPOSAL_RECEIVED: 'proposal_received',
  PROPOSAL_ENACTED: 'proposal_enacted',
  PROPOSAL_EXPIRED: 'proposal_expired',
  PROPOSAL_REVOKED: 'proposal_revoked',
  
  // Decision lifecycle
  DECISION_MADE: 'decision_made',
  DECISION_REJECTED: 'decision_rejected',
  
  // Enforcement
  ACTION_ALLOWED: 'action_allowed',
  ACTION_BLOCKED: 'action_blocked',
  ACTION_CONSTRAINED: 'action_constrained',
  ACTION_DEFERRED: 'action_deferred',
  ACTION_AUDITED: 'action_audited',
  
  // Errors
  ERRORS: 'errors',
  CONSTRAINT_FAILED: 'constraint_failed',
  CGF_UNREACHABLE: 'cgf_unreachable',
  EVALUATE_TIMEOUT: 'evaluate_timeout',
  
  // Outcomes
  OUTCOME_LOGGED: 'outcome_logged',
  SIDE_EFFECT_REPORTED: 'side_effect_reported'
};

function emitEvent(eventType, payload, proposalId = null, decisionId = null) {
  const event = {
    schema_version: CONFIG.schemaVersion,
    event_type: eventType,
    adapter_id: state.adapterId || 'unregistered',
    timestamp: Date.now() / 1000,
    proposal_id: proposalId,
    decision_id: decisionId,
    payload
  };
  
  const eventPath = `${CONFIG.dataDir}/events.jsonl`;
  appendFileSync(eventPath, JSON.stringify(event) + '\n');
  state.eventCount++;
  
  return event;
}

// ============== HELPERS ==============

function generateId(prefix) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
}

function hashContent(content) {
  if (typeof content === 'string') {
    return createHash('sha256').update(content, 'utf8').digest('hex');
  }
  return createHash('sha256').update(content).digest('hex');
}

async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(timeout);
    return response;
  } catch (error) {
    clearTimeout(timeout);
    throw error;
  }
}

// ============== CGF CLIENT ==============

async function registerWithCGF(hostConfig) {
  const payload = {
    schema_version: CONFIG.schemaVersion,
    adapter_type: CONFIG.adapterType,
    host_config: hostConfig,
    features: ['tool_call', 'memory_write'],
    risk_tiers: {
      high: 'fail_closed',
      medium: 'defer',
      low: 'fail_open'
    },
    timestamp: Date.now() / 1000
  };
  
  try {
    const response = await fetchWithTimeout(
      `${CONFIG.cgfEndpoint}/v1/register`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      },
      5000
    );
    
    if (!response.ok) {
      throw new Error(`Registration failed: ${response.status}`);
    }
    
    const data = await response.json();
    state.adapterId = data.adapter_id;
    state.registered = true;
    
    emitEvent(EVENT_TYPES.ADAPTER_REGISTERED, {
      adapter_id: state.adapterId,
      host_type: hostConfig.host_type,
      version: '0.2.0'
    });
    
    log('info', 'Registered with CGF', { adapterId: state.adapterId });
    return state.adapterId;
  } catch (error) {
    log('error', 'Registration failed', { error: error.message });
    // Local-only mode
    state.adapterId = `local-${generateId('').slice(-12)}`;
    return state.adapterId;
  }
}

async function evaluateWithCGF(proposal, context, signals) {
  const payload = {
    schema_version: CONFIG.schemaVersion,
    adapter_id: state.adapterId,
    host_config: {
      host_type: 'openclaw',
      namespace: 'default',
      capabilities: ['tool_call', 'memory_write'],
      version: '0.2.0'
    },
    proposal,
    context,
    capacity_signals: signals
  };
  
  emitEvent(EVENT_TYPES.PROPOSAL_RECEIVED, {
    proposal_id: proposal.proposal_id,
    action_type: proposal.action_type,
    action_params_hash: hashContent(JSON.stringify(proposal.action_params)).slice(0, 16),
    risk_tier: proposal.risk_tier
  }, proposal.proposal_id);
  
  try {
    const response = await fetchWithTimeout(
      `${CONFIG.cgfEndpoint}/v1/evaluate`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      },
      CONFIG.timeoutMs
    );
    
    if (!response.ok) {
      throw new Error(`Evaluation failed: ${response.status}`);
    }
    
    const data = await response.json();
    const decision = data.decision;
    
    emitEvent(EVENT_TYPES.DECISION_MADE, {
      decision_id: decision.decision_id,
      proposal_id: proposal.proposal_id,
      decision_type: decision.decision,
      confidence: decision.confidence,
      justification: decision.justification
    }, proposal.proposal_id, decision.decision_id);
    
    return decision;
  } catch (error) {
    if (error.name === 'AbortError') {
      emitEvent(EVENT_TYPES.EVALUATE_TIMEOUT, {
        proposal_id: proposal.proposal_id,
        timeout_ms: CONFIG.timeoutMs,
        elapsed_ms: CONFIG.timeoutMs
      }, proposal.proposal_id);
    } else {
      emitEvent(EVENT_TYPES.CGF_UNREACHABLE, {
        proposal_id: proposal.proposal_id,
        endpoint: CONFIG.cgfEndpoint,
        error_type: error.name,
        fail_mode_applied: 'fail_closed'
      }, proposal.proposal_id);
    }
    throw error;
  }
}

async function reportOutcome(proposalId, decisionId, outcome) {
  const payload = {
    schema_version: CONFIG.schemaVersion,
    adapter_id: state.adapterId,
    proposal_id: proposalId,
    decision_id: decisionId,
    ...outcome
  };
  
  try {
    await fetchWithTimeout(
      `${CONFIG.cgfEndpoint}/v1/outcomes/report`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      },
      5000
    );
  } catch (error) {
    // Local fallback
    const localPath = `${CONFIG.data_DIR}/outcomes_local.jsonl`;
    appendFileSync(localPath, JSON.stringify({ ...payload, report_error: error.message }) + '\n');
  }
  
  emitEvent(EVENT_TYPES.OUTCOME_LOGGED, {
    proposal_id: proposalId,
    decision_id: decisionId,
    success: outcome.success,
    duration_ms: outcome.duration_ms
  }, proposalId, decisionId);
}

// ============== TOOL CALL INTERCEPTION ==============

function observeToolProposal(toolName, toolArgs, sessionKey, agentId) {
  state.proposalCount++;
  
  const argsStr = JSON.stringify(toolArgs);
  const argsHash = hashContent(argsStr).slice(0, 32);
  
  const sideEffectTools = new Set(['file_write', 'fs_write', 'write', 'save', 'exec', 'shell', 'eval']);
  const riskTier = sideEffectTools.has(toolName) ? 'high' : 'medium';
  
  return {
    proposal_id: generateId('prop'),
    timestamp: Date.now() / 1000,
    action_type: 'tool_call',
    action_params: {
      tool_name: toolName,
      tool_args_hash: argsHash,
      side_effects_hint: sideEffectTools.has(toolName) ? ['write'] : [],
      idempotent_hint: false,
      resource_hints: []
    },
    context_refs: [sessionKey || 'unknown', agentId || 'unknown'],
    estimated_cost: { tokens: Math.floor(argsStr.length / 4), latency_ms: 500 },
    risk_tier: riskTier,
    priority: 0
  };
}

export async function governanceHookTool(toolName, toolArgs, sessionKey, agentId) {
  // Lazy registration
  if (!state.registered) {
    await registerWithCGF({ host_type: 'openclaw', namespace: 'default' });
  }
  
  const proposal = observeToolProposal(toolName, toolArgs, sessionKey, agentId);
  const context = {
    agent_id: agentId,
    session_id: sessionKey,
    turn_number: 0,
    recent_errors: 0,
    memory_growth_rate: 0
  };
  const signals = {
    token_rate: 0,
    tool_call_rate: 0,
    error_rate: 0,
    memory_growth: 0
  };
  
  const startTime = Date.now();
  
  try {
    const decision = await evaluateWithCGF(proposal, context, signals);
    
    if (decision.decision === 'BLOCK') {
      emitEvent(EVENT_TYPES.ACTION_BLOCKED, {
        decision_id: decision.decision_id,
        proposal_id: proposal.proposal_id,
        justification: decision.justification,
        reason_code: decision.reason_code || 'BLOCKED_BY_POLICY'
      }, proposal.proposal_id, decision.decision_id);
      
      throw new Error(`BLOCKED: ${decision.justification}`);
    }
    
    if (decision.decision === 'ALLOW') {
      emitEvent(EVENT_TYPES.ACTION_ALLOWED, {
        decision_id: decision.decision_id,
        proposal_id: proposal.proposal_id,
        executed_at: Date.now() / 1000
      }, proposal.proposal_id, decision.decision_id);
    }
    
    return {
      allowed: true,
      decision_id: decision.decision_id,
      proposal_id: proposal.proposal_id
    };
    
  } catch (error) {
    // Fail closed for side-effect tools
    const isSideEffect = proposal.action_params.side_effects_hint.includes('write');
    
    if (isSideEffect) {
      emitEvent(EVENT_TYPES.CGF_UNREACHABLE, {
        proposal_id: proposal.proposal_id,
        endpoint: CONFIG.cgfEndpoint,
        error_type: error.name,
        fail_mode_applied: 'fail_closed'
      }, proposal.proposal_id);
      
      throw new Error(`BLOCKED (fail-closed): CGF unavailable for side-effect tool`);
    }
    
    // Fail open for read-only
    emitEvent(EVENT_TYPES.CGF_UNREACHABLE, {
      proposal_id: proposal.proposal_id,
      endpoint: CONFIG.cgfEndpoint,
      error_type: error.name,
      fail_mode_applied: 'fail_open'
    }, proposal.proposal_id);
    
    log('warn', 'CGF unreachable - allowing read-only tool (fail open)');
    return { allowed: true, fail_open: true };
  }
}

// ============== MEMORY WRITE INTERCEPTION ==============

function observeMemoryWrite(namespace, content, sessionKey, agentId, sensitivity = 'medium') {
  state.proposalCount++;
  
  const contentHash = hashContent(content);
  const sizeBytes = content.length;
  
  // Infer sensitivity from size
  if (sensitivity === 'medium' && sizeBytes > 1000000) {
    sensitivity = 'high';
  }
  
  return {
    proposal_id: generateId('prop-mem'),
    timestamp: Date.now() / 1000,
    action_type: 'memory_write',
    action_params: {
      namespace,
      size_bytes: sizeBytes,
      sensitivity_hint: sensitivity,
      content_hash: contentHash,
      context_refs: [sessionKey || 'unknown', agentId || 'unknown'],
      operation: 'update'
    },
    context_refs: [sessionKey || 'unknown', agentId || 'unknown'],
    estimated_cost: { bytes: sizeBytes, latency_ms: 100 },
    risk_tier: sensitivity,
    priority: 0
  };
}

function applyConstraintMemory(constraint, storePath, store, mutator) {
  const constraintType = constraint?.type;
  
  if (constraintType === 'quarantine_namespace') {
    const quarantineNamespace = constraint.params?.target_namespace || '_quarantine_';
    const originalNamespace = constraint.params?.source_namespace || 'default';
    
    log('warn', 'Memory write constrained to quarantine', {
      original: originalNamespace,
      quarantine: quarantineNamespace
    });
    
    // In real implementation, this would redirect to a quarantine path
    // For demo, we flag the store with quarantine metadata
    if (store && typeof store === 'object') {
      store._quarantine_metadata = {
        original_namespace: originalNamespace,
        quarantine_namespace: quarantineNamespace,
        timestamp: Date.now() / 1000
      };
    }
    
    return {
      allowed: true,
      constrained: true,
      quarantined: true,
      target_namespace: quarantineNamespace,
      original_namespace: originalNamespace
    };
  }
  
  // Unknown constraint - fail closed
  emitEvent(EVENT_TYPES.CONSTRAINT_FAILED, {
    constraint_type: constraintType || 'unknown',
    error: `Unknown constraint type: ${constraintType}`,
    fallback_decision: 'BLOCK'
  });
  
  throw new Error(`CONSTRAINT_FAILED: Unknown constraint type ${constraintType}`);
}

/**
 * Governance hook for memory writes.
 * 
 * Intercepts updateSessionStore(storePath, mutator, opts)
 * Call chain: persistSessionEntry() -> updateSessionStore() -> saveSessionStoreUnlocked()
 * 
 * @param storePath - Path to session store file
 * @param store - Current store contents (before mutation)
 * @param mutator - Function that mutates the store
 * @param opts - Options passed to updateSessionStore
 * @returns {Promise<{allowed: boolean, quarantined?: boolean}>}
 */
export async function governanceHookMemoryWrite(storePath, store, mutator, opts = {}) {
  // Lazy registration
  if (!state.registered) {
    await registerWithCGF({ host_type: 'openclaw', namespace: 'default' });
  }
  
  // Estimate content size from store
  const content = JSON.stringify(store);
  const sessionKey = opts?.activeSessionKey || 'unknown';
  
  const proposal = observeMemoryWrite(
    dirname(storePath),
    content,
    sessionKey,
    null,
    opts?.sensitivityHint || 'medium'
  );
  
  const context = {
    agent_id: null,
    session_id: sessionKey,
    turn_number: 0,
    recent_errors: 0,
    memory_growth_rate: proposal.action_params.size_bytes
  };
  const signals = {
    token_rate: 0,
    tool_call_rate: 0,
    error_rate: 0,
    memory_growth: proposal.action_params.size_bytes
  };
  
  const startTime = Date.now();
  
  try {
    const decision = await evaluateWithCGF(proposal, context, signals);
    const decisionId = decision.decision_id;
    
    if (decision.decision === 'BLOCK') {
      const durationMs = Date.now() - startTime;
      
      emitEvent(EVENT_TYPES.ACTION_BLOCKED, {
        decision_id: decisionId,
        proposal_id: proposal.proposal_id,
        justification: decision.justification,
        reason_code: decision.reason_code || 'BLOCKED_BY_POLICY'
      }, proposal.proposal_id, decisionId);
      
      await reportOutcome(proposal.proposal_id, decisionId, {
        executed: false,
        executed_at: Date.now() / 1000,
        duration_ms: durationMs,
        success: false,
        committed: false,
        quarantined: false,
        errors: [decision.justification],
        result_summary: 'Blocked by CGF policy'
      });
      
      throw new Error(`BLOCKED: ${decision.justification}`);
    }
    
    if (decision.decision === 'CONSTRAIN') {
      const constraintResult = applyConstraintMemory(
        decision.constraint,
        storePath,
        store,
        mutator
      );
      
      const durationMs = Date.now() - startTime;
      
      emitEvent(EVENT_TYPES.ACTION_CONSTRAINED, {
        decision_id: decisionId,
        proposal_id: proposal.proposal_id,
        constraint_type: decision.constraint?.type,
        constraint_params: decision.constraint?.params
      }, proposal.proposal_id, decisionId);
      
      await reportOutcome(proposal.proposal_id, decisionId, {
        executed: true,
        executed_at: Date.now() / 1000,
        duration_ms: durationMs,
        success: true,
        committed: true,
        quarantined: true,
        result_summary: 'Quarantined by CGF policy'
      });
      
      return { allowed: true, ...constraintResult };
    }
    
    if (decision.decision === 'ALLOW') {
      const durationMs = Date.now() - startTime;
      
      emitEvent(EVENT_TYPES.ACTION_ALLOWED, {
        decision_id: decisionId,
        proposal_id: proposal.proposal_id,
        executed_at: Date.now() / 1000
      }, proposal.proposal_id, decisionId);
      
      await reportOutcome(proposal.proposal_id, decisionId, {
        executed: true,
        executed_at: Date.now() / 1000,
        duration_ms: durationMs,
        success: true,
        committed: true,
        quarantined: false,
        result_summary: 'Allowed by CGF policy'
      });
      
      return { allowed: true, quarantined: false };
    }
    
    // Unknown decision - default to allow with warning
    log('warn', 'Unknown decision type', { decision: decision.decision });
    return { allowed: true };
    
  } catch (error) {
    if (error.message.startsWith('BLOCKED:') || error.message.startsWith('CONSTRAINT_FAILED:')) {
      throw error;
    }
    
    // CGF unreachable - apply fail mode
    const sensitivity = proposal.action_params.sensitivity_hint;
    const durationMs = Date.now() - startTime;
    
    if (sensitivity === 'high' || sensitivity === 'medium') {
      // Fail closed
      emitEvent(EVENT_TYPES.CGF_UNREACHABLE, {
        proposal_id: proposal.proposal_id,
        endpoint: CONFIG.cgfEndpoint,
        error_type: error.name,
        fail_mode_applied: 'fail_closed'
      }, proposal.proposal_id);
      
      log('error', 'CGF unreachable - blocking memory write (fail closed)', {
        sensitivity,
        storePath
      });
      
      await reportOutcome(proposal.proposal_id, 'fail-mode-outcome', {
        executed: false,
        executed_at: Date.now() / 1000,
        duration_ms: durationMs,
        success: false,
        committed: false,
        quarantined: false,
        errors: ['CGF unreachable - fail closed for ' + sensitivity + ' sensitivity']
      });
      
      throw new Error(`BLOCKED (fail-closed): CGF unavailable for ${sensitivity} sensitivity memory write`);
    }
    
    // Fail open for low sensitivity
    emitEvent(EVENT_TYPES.CGF_UNREACHABLE, {
      proposal_id: proposal.proposal_id,
      endpoint: CONFIG.cgfEndpoint,
      error_type: error.name,
      fail_mode_applied: 'fail_open'
    }, proposal.proposal_id);
    
    log('warn', 'CGF unreachable - allowing memory write (fail open)', {
      sensitivity,
      storePath
    });
    
    await reportOutcome(proposal.proposal_id, 'fail-mode-outcome', {
      executed: true,
      executed_at: Date.now() / 1000,
      duration_ms: durationMs,
      success: true,
      committed: true,
      quarantined: false,
      errors: ['CGF unreachable - fail open for low sensitivity']
    });
    
    return { allowed: true, fail_open: true, quarantined: false };
  }
}

// ============== OPENCLAW INTEGRATION POINT ==============

/**
 * Integration point for OpenClaw's updateSessionStore function.
 * 
 * To integrate into OpenClaw:
 * 1. Import this module in pi-embedded-helpers-CMf7l1vP.js
 * 2. Wrap the mutator call in updateSessionStore:
 * 
 *    import { governanceHookMemoryWrite } from './openclaw_cgf_hook_v02.mjs';
 *    
 *    async function updateSessionStore(storePath, mutator, opts) {
 *      return await withSessionStoreLock(storePath, async () => {
 *        const store = loadSessionStore(storePath, { skipCache: true });
 *        
 *        // CGF GOVERNANCE HOOK
 *        const governance = await governanceHookMemoryWrite(storePath, store, mutator, opts);
 *        if (!governance.allowed) {
 *          throw new Error('Memory write blocked by CGF');
 *        }
 *        
 *        const result = await mutator(store);
 *        await saveSessionStoreUnlocked(storePath, store, opts);
 *        return result;
 *      });
 *    }
 * 
 * This enables governance of all session persistence writes in OpenClaw.
 */

log('info', 'OpenClaw CGF Hook v0.2 loaded', {
  schemaVersion: CONFIG.schemaVersion,
  endpoint: CONFIG.cgfEndpoint
});

export default {
  governanceHookTool,
  governanceHookMemoryWrite,
  CONFIG,
  state
};
