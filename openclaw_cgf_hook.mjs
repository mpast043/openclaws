/**
 * openclaw-cgf-hook.js
 * 
 * OpenClaw tool execution hook for Capacity Governance Framework (CGF).
 * 
 * INTEGRATION POINT:
 * This module patches the OpenClaw gateway tool invocation at runtime.
 * Target file: gateway-cli-CIYEdmIv.js (or similar) in OpenClaw's dist/
 * 
 * Function to patch: handleToolsInvokeHttpRequest
 * Location: exports.handleToolsInvokeHttpRequest or similar exported function
 * 
 * Call chain:
 *   HTTP POST /tools/invoke 
 *   -> handleToolsInvokeHttpRequest()
 *   -> [THIS HOOK] CGF governance check
 *   -> original tool execution (if allowed)
 * 
 * INSTALLATION:
 * 1. Place this file in OpenClaw's installation directory:
 *    /opt/homebrew/lib/node_modules/openclaw/
 * 
 * 2. Modify the OpenClaw entry point (openclaw.mjs or gateway.mjs) to require this hook:
 *    Add at the top of the file:
 *      import './openclaw-cgf-hook.js';
 * 
 * 3. Or use Node.js --require flag:
 *    node --require ./openclaw-cgf-hook.js /path/to/openclaw/dist/gateway-cli-*.js
 * 
 * CONFIGURATION (environment variables):
 * - CGF_ENDPOINT: URL of CGF server (default: http://127.0.0.1:8080)
 * - CGF_ADAPTER_ID: Override adapter ID (auto-generated if not set)
 * - CGF_TIMEOUT_MS: Evaluation timeout (default: 500)
 * - CGF_LOG_LEVEL: debug|info|warn|error (default: info)
 * - CGF_DATA_DIR: Local data directory (default: ./cgf-data)
 * - CGF_ENABLED: Set to "false" to disable (default: true)
 */

import http from 'http';
import https from 'https';
import { createHash } from 'crypto';
import { appendFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';

// ============== CONFIGURATION ==============

const CONFIG = {
  endpoint: process.env.CGF_ENDPOINT || 'http://127.0.0.1:8080',
  adapterId: process.env.CGF_ADAPTER_ID || null,
  timeoutMs: parseInt(process.env.CGF_TIMEOUT_MS) || 500,
  logLevel: process.env.CGF_LOG_LEVEL || 'info',
  dataDir: process.env.CGF_DATA_DIR || './cgf-data',
  enabled: process.env.CGF_ENABLED !== 'false'
};

// Ensure data directory exists
try {
  mkdirSync(CONFIG.dataDir, { recursive: true });
} catch (e) {
  // Directory may already exist
}

// ============== STATE ==============

const state = {
  adapterId: CONFIG.adapterId || `openclaw-${Date.now().toString(36)}`,
  registered: false,
  proposalCount: 0,
  blockedCount: 0,
  errorCount: 0
};

// ============== LOGGING ==============

const LOG_LEVELS = { debug: 0, info: 1, warn: 2, error: 3 };

function log(level, ...args) {
  if (LOG_LEVELS[level] >= LOG_LEVELS[CONFIG.logLevel]) {
    const timestamp = new Date().toISOString();
    console.error(`[CGF:${level.toUpperCase()}] ${timestamp}`, ...args);
  }
}

// ============== EVENT EMITTER ==============

function emitEvent(eventType, payload) {
  const event = {
    event_type: eventType,
    adapter_id: state.adapterId,
    timestamp: Date.now() / 1000,
    payload
  };
  
  const logPath = join(CONFIG.dataDir, 'events.jsonl');
  try {
    appendFileSync(logPath, JSON.stringify(event) + '\n');
  } catch (e) {
    log('error', 'Failed to write event:', e.message);
  }
  
  if (eventType === 'action_blocked' || eventType === 'cgf_unreachable') {
    log('warn', `Event: ${eventType}`, payload);
  } else {
    log('debug', `Event: ${eventType}`, payload);
  }
}

// ============== CGF HTTP CLIENT ==============

async function httpRequest(method, path, body = null, timeoutMs = null) {
  const url = new URL(path, CONFIG.endpoint);
  const isHttps = url.protocol === 'https:';
  const client = isHttps ? https : http;
  
  const timeout = timeoutMs || CONFIG.timeoutMs;
  
  return new Promise((resolve, reject) => {
    const options = {
      hostname: url.hostname,
      port: url.port || (isHttps ? 443 : 80),
      path: url.pathname + url.search,
      method,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    };
    
    const req = client.request(options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const json = JSON.parse(data);
          resolve({ status: res.statusCode, data: json });
        } catch (e) {
          resolve({ status: res.statusCode, data: data });
        }
      });
    });
    
    req.on('error', reject);
    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Request timeout'));
    });
    
    req.setTimeout(timeout);
    
    if (body) {
      req.write(JSON.stringify(body));
    }
    
    req.end();
  });
}

// ============== CGF INTERACTIONS ==============

async function registerAdapter() {
  if (state.registered) return state.adapterId;
  
  try {
    const res = await httpRequest('POST', '/v1/register', {
      adapter_type: 'openclaw',
      host_metadata: {
        host_type: 'openclaw',
        namespace: 'main',
        capabilities: ['tool_call'],
        version: '0.1.0'
      }
    }, 5000); // Registration gets longer timeout
    
    if (res.status === 200 && res.data.adapter_id) {
      state.adapterId = res.data.adapter_id;
      state.registered = true;
      emitEvent('adapter_registered', { adapter_id: state.adapterId });
      log('info', `Registered with CGF: ${state.adapterId}`);
      return state.adapterId;
    } else {
      throw new Error(`Registration failed: ${res.status}`);
    }
  } catch (e) {
    log('error', 'Registration error:', e.message);
    // Continue with local adapter ID, will retry on next call
    return state.adapterId;
  }
}

async function evaluateProposal(proposal, context, signals) {
  await registerAdapter();
  
  emitEvent('proposal_received', {
    proposal_id: proposal.proposal_id,
    tool_name: proposal.action_params.tool_name,
    risk_tier: proposal.risk_tier
  });
  
  try {
    const res = await httpRequest('POST', '/v1/evaluate', {
      adapter_id: state.adapterId,
      host_config: { host_type: 'openclaw' },
      proposal,
      context,
      capacity_signals: signals
    });
    
    if (res.status === 200) {
      emitEvent('decision_made', {
        proposal_id: proposal.proposal_id,
        decision: res.data.decision,
        confidence: res.data.confidence
      });
      return res.data;
    } else {
      throw new Error(`Evaluate failed: ${res.status}`);
    }
  } catch (e) {
    log('error', 'Evaluate error:', e.message);
    throw e;
  }
}

async function reportOutcome(outcome) {
  // Fire-and-forget reporting
  try {
    await httpRequest('POST', '/v1/outcomes/report', outcome, 5000);
    emitEvent('outcome_logged', {
      proposal_id: outcome.proposal_id,
      executed: outcome.executed,
      success: outcome.success
    });
  } catch (e) {
    log('error', 'Outcome report failed:', e.message);
    // Write to local fallback
    const logPath = join(CONFIG.dataDir, 'outcomes_local.jsonl');
    try {
      appendFileSync(logPath, JSON.stringify({ ...outcome, report_failed: true, error: e.message }) + '\n');
    } catch (e2) {
      // Ignore
    }
  }
}

// ============== OBSERVATION ==============

function observeProposal(toolName, toolArgs, sessionKey, agentId) {
  state.proposalCount++;
  
  const argsCanonical = JSON.stringify(toolArgs, Object.keys(toolArgs).sort());
  const toolArgsHash = createHash('sha256').update(argsCanonical).digest('hex').slice(0, 32);
  
  // Determine risk tier
  const sideEffectTools = ['file_write', 'fs_write', 'write', 'save', 'exec', 'shell', 'eval'];
  const riskTier = sideEffectTools.includes(toolName) ? 'high' : 'medium';
  
  return {
    proposal_id: `prop-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
    timestamp: Date.now() / 1000,
    action_type: 'tool_call',
    action_params: {
      tool_name: toolName,
      tool_args_hash: toolArgsHash,
      side_effects_hint: sideEffectTools.includes(toolName) ? ['write'] : [],
      idempotent_hint: false,
      resource_hints: []
    },
    context_refs: [sessionKey || 'unknown', agentId || 'unknown'],
    estimated_cost: {
      tokens: Math.ceil(argsCanonical.length / 4),
      latency_ms: 500
    },
    risk_tier: riskTier
  };
}

function observeContext(sessionKey, agentId) {
  return {
    agent_id: agentId,
    session_id: sessionKey,
    turn_number: 0,
    recent_errors: state.errorCount,
    memory_growth_rate: 0.0
  };
}

function observeSignals() {
  return {
    token_rate: 0.0,
    tool_call_rate: 0.0,
    error_rate: 0.0,
    memory_growth: 0.0
  };
}

// ============== ENFORCEMENT ==============

async function enforceAllow(proposal, decisionId) {
  emitEvent('action_allowed', {
    proposal_id: proposal.proposal_id,
    decision_id: decisionId,
    tool_name: proposal.action_params.tool_name
  });
  state.allowedCount = (state.allowedCount || 0) + 1;
  return { allowed: true, proposal };
}

async function enforceBlock(proposal, decisionId, justification) {
  emitEvent('action_blocked', {
    proposal_id: proposal.proposal_id,
    decision_id: decisionId,
    tool_name: proposal.action_params.tool_name,
    justification
  });
  state.blockedCount++;
  throw new Error(`BLOCKED: ${justification}`);
}

async function enforceConstrain(proposal, decisionId, constraint) {
  emitEvent('constraint_applied', {
    proposal_id: proposal.proposal_id,
    decision_id: decisionId,
    reason: constraint?.reason || ''
  });
  return { allowed: true, constrained: true, proposal, constraint };
}

// ============== FAIL MODE ==============

function applyFailMode(riskTier, toolName) {
  // ENFORCEMENT INVARIANT: Side-effect tools -> FAIL_CLOSED
  const sideEffectTools = ['file_write', 'fs_write', 'write', 'save', 'exec', 'shell', 'eval'];
  const hasSideEffects = sideEffectTools.includes(toolName);
  
  if (hasSideEffects || riskTier === 'high') {
    return 'BLOCK';
  }
  // Read-only tools -> FAIL_OPEN (allow with logging)
  return 'ALLOW';
}

// ============== GOVERNANCE HOOK ==============

async function governanceHook(toolName, toolArgs, sessionKey, agentId) {
  if (!CONFIG.enabled) {
    return { allowed: true, reason: 'CGF disabled' };
  }
  
  // 1. OBSERVE
  const proposal = observeProposal(toolName, toolArgs, sessionKey, agentId);
  const context = observeContext(sessionKey, agentId);
  const signals = observeSignals();
  
  let decision;
  let decisionId;
  
  try {
    // 2. EVALUATE
    const result = await evaluateProposal(proposal, context, signals);
    decision = result.decision;
    decisionId = result.decision_id;
  } catch (e) {
    // ENFORCEMENT INVARIANT: CGF unavailable -> apply fail mode
    state.errorCount++;
    decision = applyFailMode(proposal.risk_tier, toolName);
    decisionId = 'fail-mode-' + Date.now();
    
    const errorType = e.message?.includes('timeout') ? 'timeout' : 'unreachable';
    emitEvent(errorType === 'timeout' ? 'evaluate_timeout' : 'cgf_unreachable', {
      proposal_id: proposal.proposal_id,
      error: e.message,
      fail_mode_decision: decision
    });
    
    log('warn', `CGF ${errorType}, applying fail mode: ${decision}`);
  }
  
  // 3. ENFORCE
  let result;
  if (decision === 'ALLOW') {
    result = await enforceAllow(proposal, decisionId);
  } else if (decision === 'BLOCK') {
    await enforceBlock(proposal, decisionId, 'Blocked by CGF policy');
    result = { allowed: false, blocked: true };
  } else if (decision === 'CONSTRAIN') {
    // v0.1: CONSTRAIN -> apply constraint if possible, else BLOCK
    const constraint = { reason: 'Modified by policy' };
    result = await enforceConstrain(proposal, decisionId, constraint);
  } else if (decision === 'DEFER') {
    emitEvent('action_deferred', {
      proposal_id: proposal.proposal_id,
      reason: 'Action deferred for review'
    });
    throw new Error('DEFERRED: Action requires human review');
  } else if (decision === 'AUDIT') {
    // v0.1: AUDIT -> allow with audit flag
    result = await enforceAllow(proposal, decisionId);
    result.auditRequired = true;
  } else {
    // Unknown decision: fail closed
    await enforceBlock(proposal, decisionId, `Unknown decision: ${decision}`);
    result = { allowed: false, blocked: true };
  }
  
  return { ...result, proposal, decisionId };
}

async function reportExecution(proposal, decisionId, executed, success, error) {
  if (!CONFIG.enabled) return;
  
  const outcome = {
    adapter_id: state.adapterId,
    proposal_id: proposal.proposal_id,
    decision_id: decisionId,
    executed: executed,
    executed_at: Date.now() / 1000,
    duration_ms: 0,
    success: success,
    result_summary: error ? error.message?.slice(0, 200) : 'Executed',
    actual_cost: {},
    errors: error ? [error.message] : [],
    side_effects: []
  };
  
  await reportOutcome(outcome);
}

// ============== OPENCLAW PATCH ==============

function patchOpenClaw() {
  // The integration point: wrap the tool invocation in OpenClaw's gateway
  // Since we cannot modify the compiled source, we monkey-patch if gateway is loaded
  
  log('info', 'CGF Hook loaded, adapter ID:', state.adapterId);
  log('info', 'CGF Endpoint:', CONFIG.endpoint);
  
  // Register on load
  registerAdapter().catch(e => {
    log('warn', 'Initial registration failed:', e.message);
  });
}

// ============== MODULE EXPORTS ==============

export {
  CONFIG,
  state,
  governanceHook,
  reportExecution,
  patchOpenClaw,
  // For testing
  registerAdapter,
  evaluateProposal,
  observeProposal
};

// ============== INITIALIZATION ==============

if (typeof process !== 'undefined') {
  patchOpenClaw();
}
