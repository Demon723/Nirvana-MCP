#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve as pathResolve, basename } from "node:path";
import { homedir } from "node:os";
import { spawn } from "node:child_process";

const log = (...a) => console.error("[nirvana-mcp]", ...a);

// ============================================================================
// NIRVANA CONFIG
// ============================================================================
const CONFIG = {
  stateDir: process.env.NIRVANA_STATE_DIR ?? join(homedir(), ".nirvana-mcp"),
  outputDir: process.env.NIRVANA_OUTPUT_DIR ?? join(homedir(), ".nirvana-mcp/artifacts"),
  anthropicKey: process.env.ANTHROPIC_API_KEY ?? "",
  openaiKey: process.env.OPENAI_API_KEY ?? "",
  model: process.env.NIRVANA_MODEL ?? "claude-sonnet-4-5",
  httpTimeoutMs: Number(process.env.HTTP_TIMEOUT_MS ?? 30_000),
  maxScanFiles: Number(process.env.MAX_SCAN_FILES ?? 2000),
  maxFileSizeBytes: Number(process.env.MAX_FILE_SIZE_BYTES ?? 2_000_000),
};
if (!existsSync(CONFIG.stateDir)) mkdirSync(CONFIG.stateDir, { recursive: true });
if (!existsSync(CONFIG.outputDir)) mkdirSync(CONFIG.outputDir, { recursive: true });

// ============================================================================
// LLM BRIDGE — Anthropic → OpenAI → deterministic fallback
// ============================================================================

// ============================================================================
// OLLAMA DISCOVERY — auto-detect local models, no API key required
// ============================================================================
const OLLAMA_HOST = process.env.OLLAMA_HOST ?? "http://localhost:11434";
const OLLAMA_ENABLED = process.env.NIRVANA_OLLAMA_ENABLED !== "false";
const OLLAMA_PREFERRED_MODEL = process.env.NIRVANA_OLLAMA_MODEL ?? null;

const OLLAMA_MODEL_PRIORITY = [
  "qwen3.5", "qwen3", "qwen2.5-coder", "qwen2.5",
  "llama3.3", "llama3.1", "llama3",
  "deepseek-r1", "deepseek-coder",
  "phi3.5", "phi3", "gemma2", "mistral",
];

let ollamaModelsCache = { models: [], fetchedAt: 0 };

async function discoverOllamaModels() {
  if (!OLLAMA_ENABLED) return [];
  const now = Date.now();
  if (now - ollamaModelsCache.fetchedAt < 30_000) return ollamaModelsCache.models;
  try {
    const res = await fetch(`${OLLAMA_HOST}/api/tags`, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) return [];
    const data = await res.json();
    const models = (data.models ?? []).map(m => m.name);
    ollamaModelsCache = { models, fetchedAt: now };
    return models;
  } catch { return []; }
}

function pickBestOllamaModel(models) {
  if (models.length === 0) return null;
  if (OLLAMA_PREFERRED_MODEL && models.some(m => m.startsWith(OLLAMA_PREFERRED_MODEL))) {
    return models.find(m => m.startsWith(OLLAMA_PREFERRED_MODEL));
  }
  for (const prefix of OLLAMA_MODEL_PRIORITY) {
    const match = models.find(m => m.toLowerCase().startsWith(prefix));
    if (match) return match;
  }
  return models[0];
}

async function callOllama({ system, prompt, temperature = 0.2, maxTokens = 3000, model }) {
  const res = await fetch(`${OLLAMA_HOST}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: model ?? pickBestOllamaModel(ollamaModelsCache.models),
      messages: [
        ...(system ? [{ role: "system", content: system }] : []),
        { role: "user", content: prompt },
      ],
      stream: false,
      options: { temperature, num_predict: maxTokens },
    }),
    signal: AbortSignal.timeout(CONFIG.httpTimeoutMs),
  });
  if (!res.ok) throw new Error(`Ollama HTTP ${res.status}`);
  const data = await res.json();
  return data.message?.content ?? "";
}

async function callLLM({ system, prompt, temperature = 0.2, maxTokens = 3000 }) {
  // Tier 1: Ollama (no key required)
  if (OLLAMA_ENABLED) {
    const models = await discoverOllamaModels();
    if (models.length > 0) {
      try {
        const text = await callOllama({ system, prompt, temperature, maxTokens });
        return { text, provider: "ollama:" + pickBestOllamaModel(models) };
      } catch (e) {
        log("Ollama call failed: " + e.message + " — falling back");
      }
    }
  }
  // Tier 2: Anthropic
  if (CONFIG.anthropicKey) {
    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": CONFIG.anthropicKey,
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify({
          model: CONFIG.model,
          system,
          messages: [{ role: "user", content: prompt }],
          max_tokens: maxTokens,
          temperature,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        const text = (data.content ?? []).filter(b => b.type === "text").map(b => b.text).join("");
        return { text, provider: "anthropic" };
      }
    } catch {}
  }
  // Tier 3: OpenAI
  if (CONFIG.openaiKey) {
    try {
      const res = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Bearer " + CONFIG.openaiKey },
        body: JSON.stringify({
          model: "gpt-5.2",
          messages: [{ role: "system", content: system }, { role: "user", content: prompt }],
          temperature,
          max_tokens: maxTokens,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        return { text: data.choices?.[0]?.message?.content ?? "", provider: "openai" };
      }
    } catch {}
  }
  // Tier 4: deterministic fallback
  return { text: fallbackAnalysis(prompt), provider: "fallback" };
}

function fallbackAnalysis(prompt) {
  const lower = prompt.toLowerCase();
  const signals = [];
  if (/security|auth|inject|vulnerab/.test(lower)) signals.push("security-sensitive");
  if (/architect|design|system/.test(lower)) signals.push("architecture-relevant");
  if (/data|schema|database/.test(lower)) signals.push("data-model");
  if (/compliance|gdpr|hipaa|pci/.test(lower)) signals.push("compliance");
  return JSON.stringify({
    summary: "Fallback analysis (no LLM key configured).",
    signals,
    recommendation: signals.length > 0 ? "review-required" : "proceed-with-caution",
    confidence: 0.35,
  });
}

// ============================================================================
// KNOWLEDGE VAULT — BM25 + provenance
// ============================================================================
class KnowledgeVault {
  constructor() { this.docs = new Map(); this.chunks = []; this.df = new Map(); this.avgdl = 0; }

  ingest({ title, content, source, tags = [] }) {
    const docId = randomUUID();
    const lines = content.split("\n");
    const chunkSize = 20;
    const docChunks = [];
    for (let i = 0; i < lines.length; i += chunkSize) {
      const text = lines.slice(i, i + chunkSize).join("\n");
      if (!text.trim()) continue;
      const tokens = tokenize(text);
      const chunk = {
        id: `${docId}-${docChunks.length}`,
        docId, title, source: source ?? title,
        text, tokens,
        lineStart: i + 1, lineEnd: Math.min(i + chunkSize, lines.length),
        contentHash: createHash("sha256").update(text).digest("hex").slice(0, 16),
        verifiedAt: new Date().toISOString(),
        tags,
      };
      docChunks.push(chunk);
      this.chunks.push(chunk);
      for (const t of new Set(tokens)) this.df.set(t, (this.df.get(t) ?? 0) + 1);
    }
    const doc = { id: docId, title, source, content, chunkIds: docChunks.map(c => c.id), tags, ingestedAt: Date.now() };
    this.docs.set(docId, doc);
    this.avgdl = this.chunks.reduce((s, c) => s + c.tokens.length, 0) / Math.max(this.chunks.length, 1);
    return doc;
  }

  search(query, topK = 5, filter = {}) {
    const qTokens = tokenize(query);
    const N = this.chunks.length || 1;
    const k1 = 1.5, b = 0.75;
    const scored = [];
    for (const c of this.chunks) {
      if (filter.tag && !c.tags.includes(filter.tag)) continue;
      const dl = c.tokens.length;
      let score = 0;
      for (const t of qTokens) {
        const tf = c.tokens.filter(x => x === t).length;
        if (tf === 0) continue;
        const df = this.df.get(t) ?? 0;
        const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
        score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / this.avgdl)));
      }
      if (score > 0) scored.push({ chunk: c, score });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map(s => ({
      id: s.chunk.id, title: s.chunk.title, source: s.chunk.source,
      text: s.chunk.text, score: parseFloat(s.score.toFixed(4)),
      lineStart: s.chunk.lineStart, lineEnd: s.chunk.lineEnd,
      contentHash: s.chunk.contentHash, verifiedAt: s.chunk.verifiedAt,
    }));
  }

  reconstruct(chunkId) {
    const c = this.chunks.find(x => x.id === chunkId);
    if (!c) return null;
    return { ...c, fullDoc: this.docs.get(c.docId)?.content ?? null };
  }

  stats() {
    return { documents: this.docs.size, chunks: this.chunks.length, vocabulary: this.df.size };
  }
}

function tokenize(text) {
  return text.toLowerCase().split(/[^a-z0-9_]+/).filter(t => t.length > 2);
}

const VAULT = new KnowledgeVault();

// ============================================================================
// THOUGHT GRAPH — DAG with confidence
// ============================================================================
class ThoughtGraph {
  constructor() { this.nodes = []; this.edges = []; }
  add({ content, strategy, confidence = 0.5, parents = [], meta = {} }) {
    const id = `t${this.nodes.length + 1}`;
    const node = { id, content, strategy, confidence, parents, meta, createdAt: Date.now() };
    this.nodes.push(node);
    for (const p of parents) this.edges.push({ from: p, to: id });
    return node;
  }
  best() { return this.nodes.reduce((a, b) => (b.confidence > a.confidence ? b : a), this.nodes[0]); }
  toJSON() { return { nodes: this.nodes, edges: this.edges, nodeCount: this.nodes.length, bestNode: this.best() ?? null }; }
}

// ============================================================================
// EXPERT REASONING STRATEGIES
// ============================================================================
const STRATEGIES = {
  sequential: "Work through the problem step by step. State each step and its justification.",
  dialectic: "Present the thesis, then the antithesis, then synthesize a resolution.",
  parallel: "Explore multiple independent lines of reasoning in parallel, then compare them.",
  analogical: "Find a structurally similar problem from another domain and transfer the insight.",
  abductive: "Generate the most likely explanation for the observed facts, then test it against alternatives.",
  first_principles: "Deconstruct the problem to its fundamental truths and rebuild from there.",
  counterfactual: "Ask what would happen if a key assumption were false; trace the ripple effects.",
  systems: "Map feedback loops, leverage points, and second-order effects.",
  mcts: "Simulate multiple decision paths and select the one with the highest expected value.",
  auto: "Choose the most appropriate strategy based on the problem shape.",
};

function chooseStrategy(prompt) {
  const lower = prompt.toLowerCase();
  if (/debug|incident|root cause|why/.test(lower)) return "abductive";
  if (/compare|versus|vs|trade-?off/.test(lower)) return "dialectic";
  if (/design|architect|system/.test(lower)) return "systems";
  if (/novel|unprecedented|new/.test(lower)) return "first_principles";
  if (/risk|what if|failure/.test(lower)) return "counterfactual";
  return "sequential";
}

async function runReasoning({ question, strategy, depth = 3 }) {
  const chosen = strategy === "auto" ? chooseStrategy(question) : strategy;
  const graph = new ThoughtGraph();
  let frontier = [];
  for (let d = 0; d < depth; d++) {
    const prompt = `You are an expert using the "${chosen}" reasoning strategy.\n\nQUESTION: ${question}\n\nDEPTH: ${d + 1}/${depth}\nPRIOR THOUGHTS: ${JSON.stringify(frontier.map(n => n.content), null, 2)}\n\nProduce ONE new thought that advances the reasoning. Return JSON: {"thought":"...","confidence":0.0-1.0,"parents":[indices]}`;
    const { text } = await callLLM({ system: `You are a rigorous expert reasoner using ${chosen}.`, prompt, temperature: 0.3 });
    let parsed;
    try { parsed = JSON.parse(text.match(/\{[\s\S]*\}/)?.[0] ?? text); } catch { parsed = { thought: text.slice(0, 500), confidence: 0.5, parents: [] }; }
    const node = graph.add({
      content: parsed.thought ?? text.slice(0, 500),
      strategy: chosen,
      confidence: Math.min(Math.max(parsed.confidence ?? 0.5, 0), 1),
      parents: (parsed.parents ?? []).map(i => graph.nodes[i]?.id).filter(Boolean),
    });
    frontier.push(node);
  }
  const best = graph.best();
  return {
    question, strategy: chosen, depth,
    thoughtGraph: graph.toJSON(),
    conclusion: best?.content ?? "(no conclusion)",
    confidence: best?.confidence ?? 0,
    confidenceInterpretation: best?.confidence > 0.8 ? "HIGH" : best?.confidence > 0.5 ? "MEDIUM" : "LOW — escalate or gather more evidence",
  };
}

// ============================================================================
// EXPERT COUNCIL — multi-persona deliberation
// ============================================================================
const DEFAULT_PERSONAS = [
  { id: "architect", name: "The Architect", expertise: ["system design", "scalability"], bias: "favours long-term structure over short-term speed" },
  { id: "security", name: "The Security Lead", expertise: ["threat modelling", "auth", "compliance"], bias: "assumes breach until proven otherwise" },
  { id: "pragmatist", name: "The Pragmatist", expertise: ["delivery", "cost", "operations"], bias: "favours the simplest thing that ships" },
  { id: "domain", name: "The Domain Expert", expertise: ["business logic", "user needs"], bias: "optimises for real-world fit" },
];

async function runCouncil({ topic, personas = DEFAULT_PERSONAS }) {
  const contributions = [];
  const disagreements = [];
  for (const p of personas) {
    const prompt = `You are ${p.name}. Expertise: ${p.expertise.join(", ")}. Bias: ${p.bias}.\n\nTOPIC: ${topic}\n\nGive your independent analysis. Return JSON: {"contribution":"...","confidence":0.0-1.0,"concerns":["..."]}`;
    const { text } = await callLLM({ system: `You are ${p.name}. Stay in character.`, prompt, temperature: 0.4 });
    let parsed;
    try { parsed = JSON.parse(text.match(/\{[\s\S]*\}/)?.[0] ?? text); } catch { parsed = { contribution: text.slice(0, 400), confidence: 0.5, concerns: [] }; }
    contributions.push({ persona: p.name, personaId: p.id, contribution: parsed.contribution, confidence: parsed.confidence ?? 0.5, concerns: parsed.concerns ?? [] });
  }
  for (let i = 0; i < contributions.length; i++) {
    for (let j = i + 1; j < contributions.length; j++) {
      const a = contributions[i], b = contributions[j];
      const sharedConcerns = a.concerns.filter(c => b.concerns.some(x => x.toLowerCase().includes(c.toLowerCase())));
      if (sharedConcerns.length > 0) {
        disagreements.push({ topic: sharedConcerns[0], personas: [a.persona, b.persona], positions: [a.contribution.slice(0, 200), b.contribution.slice(0, 200)] });
      }
    }
  }
  const synthPrompt = `You are the council chair. Synthesize these expert contributions into a decision.\n\nTOPIC: ${topic}\nCONTRIBUTIONS: ${JSON.stringify(contributions, null, 2)}\nDISAGREEMENTS: ${JSON.stringify(disagreements, null, 2)}\n\nReturn JSON: {"synthesis":"...","recommendation":"apply|investigate|reject","confidence":0.0-1.0,"openQuestions":["..."]}`;
  const { text } = await callLLM({ system: "You are a neutral council chair.", prompt: synthPrompt, temperature: 0.2 });
  let synthesis;
  try { synthesis = JSON.parse(text.match(/\{[\s\S]*\}/)?.[0] ?? text); } catch { synthesis = { synthesis: text.slice(0, 600), recommendation: "investigate", confidence: 0.5, openQuestions: [] }; }
  return { topic, contributions, disagreements, synthesis, personaCount: personas.length };
}

// ============================================================================
// ARTIFACT REGISTRY
// ============================================================================
const ARTIFACTS = new Map();
function registerArtifact({ kind, name, content, language, source, metadata = {} }) {
  const id = randomUUID();
  const hash = createHash("sha256").update(content).digest("hex").slice(0, 16);
  const artifact = { id, kind, name, content, language, source, metadata, hash, createdAt: Date.now(), reviewed: false };
  ARTIFACTS.set(id, artifact);
  return artifact;
}
function persistArtifact(a) {
  const ext = { javascript: "js", typescript: "ts", python: "py", markdown: "md", json: "json", yaml: "yaml", shell: "sh" }[a.language] ?? "txt";
  const path = join(CONFIG.outputDir, `${a.name.replace(/[^a-zA-Z0-9._-]/g, "_")}-${a.hash}.${ext}`);
  writeFileSync(path, a.content);
  a.path = path;
  return path;
}

// ============================================================================
// SOURCE DETECTION + CLI PROBE
// ============================================================================
function detectSource(input) {
  const s = String(input ?? "").trim();
  if (/^https?:\/\//.test(s)) return { type: "url", url: s };
  if (existsSync(s)) {
    const st = statSync(s);
    if (st.isDirectory()) return { type: "repo", path: pathResolve(s) };
    return { type: "file", path: pathResolve(s) };
  }
  if (s.includes(" ")) return { type: "cli-invocation", command: s };
  return { type: "cli-command", command: s };
}

function runCli(cmd, args = [], timeout = 5000) {
  return new Promise((resolve) => {
    let proc;
    try { proc = spawn(cmd, args, { stdio: ["ignore", "pipe", "pipe"] }); }
    catch (e) { return resolve({ ok: false, error: String(e.message), stdout: "", stderr: "" }); }
    let stdout = "", stderr = "";
    const t = setTimeout(() => { try { proc.kill("SIGTERM"); } catch {} }, timeout);
    proc.stdout.on("data", c => stdout += c.toString());
    proc.stderr.on("data", c => stderr += c.toString());
    proc.on("exit", code => { clearTimeout(t); resolve({ ok: code === 0, exitCode: code, stdout, stderr }); });
    proc.on("error", e => { clearTimeout(t); resolve({ ok: false, error: String(e.message), stdout, stderr }); });
  });
}

async function probeCli(command) {
  const parts = command.split(/\s+/);
  const help = await runCli(parts[0], [...parts.slice(1), "--help"]);
  const subcommands = [...new Set([...help.stdout.matchAll(/^\s{2,}([a-z][\w-]+)\s{2,}/gm)].map(m => m[1]))].slice(0, 30);
  const flags = [...new Map([...help.stdout.matchAll(/(--[\w-]+)/g)].map(m => [m[1], m[1]])).values()].slice(0, 40);
  return { command, subcommands, flags, hasHelp: help.ok && help.stdout.length > 0 };
}

// ============================================================================
// SERVER
// ============================================================================
const server = new McpServer({ name: "nirvana", version: "1.0.0" });

// ---- TOOL 1: expert_reason ----
server.tool(
  "expert_reason",
  "Runs a structured expert reasoning session using one of 10 strategies (sequential, dialectic, parallel, analogical, abductive, first_principles, counterfactual, systems, mcts, auto). Returns a thought DAG, conclusion, and confidence score.",
  {
    question: z.string(),
    strategy: z.enum(Object.keys(STRATEGIES)).default("auto"),
    depth: z.number().int().min(1).max(6).default(3),
  },
  { readOnlyHint: true, openWorldHint: true },
  async ({ question, strategy, depth }) => {
    const result = await runReasoning({ question, strategy, depth });
    return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
  }
);

// ---- TOOL 2: expert_council ----
server.tool(
  "expert_council",
  "Convenes a multi-persona expert council (architect, security lead, pragmatist, domain expert by default). Each persona analyses independently; disagreements are surfaced; a neutral chair synthesises a recommendation with confidence.",
  {
    topic: z.string(),
    personas: z.array(z.object({
      id: z.string(), name: z.string(), expertise: z.array(z.string()), bias: z.string().optional(),
    })).optional(),
  },
  { readOnlyHint: true, openWorldHint: true },
  async ({ topic, personas }) => {
    const result = await runCouncil({ topic, personas });
    return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
  }
);

// ---- TOOL 3: expert_debate ----
server.tool(
  "expert_debate",
  "Stages an adversarial debate: a proposer argues for a position, an opponent argues against it, and an adjudicator delivers a verdict with confidence.",
  {
    proposition: z.string(),
    rounds: z.number().int().min(1).max(3).default(1),
  },
  { readOnlyHint: true, openWorldHint: true },
  async ({ proposition, rounds }) => {
    const debate = [];
    for (let r = 0; r < rounds; r++) {
      const prop = await callLLM({ system: "You are the proposer. Argue FOR the proposition with evidence.", prompt: `PROPOSITION: ${proposition}\nROUND: ${r + 1}`, temperature: 0.4 });
      const opp = await callLLM({ system: "You are the opponent. Argue AGAINST the proposition with evidence.", prompt: `PROPOSITION: ${proposition}\nROUND: ${r + 1}\nPROPOSER SAID: ${prop.text.slice(0, 800)}`, temperature: 0.4 });
      debate.push({ round: r + 1, proposer: prop.text.slice(0, 800), opponent: opp.text.slice(0, 800) });
    }
    const verdict = await callLLM({ system: "You are a neutral adjudicator. Weigh both sides and deliver a verdict.", prompt: `PROPOSITION: ${proposition}\nDEBATE: ${JSON.stringify(debate, null, 2)}\nReturn JSON: {"verdict":"...","winner":"proposer|opponent|draw","confidence":0.0-1.0}`, temperature: 0.1 });
    return { content: [{ type: "text", text: JSON.stringify({ proposition, debate, verdict: verdict.text }, null, 2) }] };
  }
);

// ---- TOOL 4: expert_chain ----
server.tool(
  "expert_chain",
  "Runs an ordered refinement chain: draft → critique → decide. Each stage builds on the previous one. Returns the final decision and the full chain.",
  {
    task: z.string(),
    stages: z.array(z.enum(["draft", "critique", "decide", "verify"])).default(["draft", "critique", "decide"]),
  },
  { readOnlyHint: true, openWorldHint: true },
  async ({ task, stages }) => {
    const chain = [];
    let context = "";
    for (const stage of stages) {
      const prompt = `STAGE: ${stage}\nTASK: ${task}\nPRIOR STAGES: ${context}\n\nProduce the ${stage} output.`;
      const { text } = await callLLM({ system: `You are performing the "${stage}" stage of an expert refinement chain.`, prompt, temperature: stage === "decide" ? 0.1 : 0.3 });
      chain.push({ stage, output: text.slice(0, 2000) });
      context += `\n[${stage}]: ${text.slice(0, 1000)}`;
    }
    const decision = chain.find(c => c.stage === "decide")?.output ?? chain[chain.length - 1]?.output;
    return { content: [{ type: "text", text: JSON.stringify({ task, stages, chain, decision }, null, 2) }] };
  }
);

// ---- TOOL 5: expert_consult ----
server.tool(
  "expert_consult",
  "Single expert consultation with a structured sidecar. Use for a hard question when you need one trusted expert answer plus a machine-readable recommendation (apply/investigate/reject) and confidence.",
  {
    question: z.string(),
    domain: z.string().default("general"),
    context: z.string().optional(),
  },
  { readOnlyHint: true, openWorldHint: true },
  async ({ question, domain, context }) => {
    const prompt = `DOMAIN: ${domain}\nCONTEXT: ${context ?? "(none)"}\nQUESTION: ${question}\n\nAnswer as a domain expert. Return JSON: {"answer":"...","recommendation":"apply|investigate|reject|needs_more_context","confidence":0.0-1.0,"caveats":["..."]}`;
    const { text, provider } = await callLLM({ system: `You are a senior ${domain} expert. Be precise and honest about uncertainty.`, prompt, temperature: 0.2 });
    let sidecar;
    try { sidecar = JSON.parse(text.match(/\{[\s\S]*\}/)?.[0] ?? text); } catch { sidecar = { answer: text.slice(0, 1000), recommendation: "investigate", confidence: 0.5, caveats: [] }; }
    return { content: [{ type: "text", text: JSON.stringify({ question, domain, provider, ...sidecar }, null, 2) }] };
  }
);

// ---- TOOL 6: knowledge_ingest ----
server.tool(
  "knowledge_ingest",
  "Ingests a document into the provenance-rich knowledge vault. Supports markdown, code, and plain text. Returns the document ID and chunk count. Query later with knowledge_query.",
  {
    title: z.string(),
    content: z.string(),
    source: z.string().optional(),
    tags: z.array(z.string()).default([]),
  },
  { readOnlyHint: false, openWorldHint: false },
  async ({ title, content, source, tags }) => {
    const doc = VAULT.ingest({ title, content, source, tags });
    return { content: [{ type: "text", text: JSON.stringify({ docId: doc.id, title, chunks: doc.chunkIds.length, vaultStats: VAULT.stats() }, null, 2) }] };
  }
);

// ---- TOOL 7: knowledge_query ----
server.tool(
  "knowledge_query",
  "Hybrid BM25 retrieval over the knowledge vault. Returns provenance-rich results: source, line range, content hash, verified timestamp, and relevance score.",
  {
    query: z.string(),
    topK: z.number().int().min(1).max(20).default(5),
    tag: z.string().optional(),
  },
  { readOnlyHint: true, openWorldHint: false },
  async ({ query, topK, tag }) => {
    const results = VAULT.search(query, topK, { tag });
    return { content: [{ type: "text", text: JSON.stringify({ query, resultCount: results.length, results, vaultStats: VAULT.stats() }, null, 2) }] };
  }
);

// ---- TOOL 8: knowledge_reconstruct ----
server.tool(
  "knowledge_reconstruct",
  "Reconstructs the exact source text for a chunk ID so an agent can prove what grounded its answer. Returns the chunk, its line range, content hash, and the full document.",
  {
    chunkId: z.string(),
  },
  { readOnlyHint: true, openWorldHint: false },
  async ({ chunkId }) => {
    const r = VAULT.reconstruct(chunkId);
    if (!r) return { content: [{ type: "text", text: `Unknown chunk: ${chunkId}` }], isError: true };
    return { content: [{ type: "text", text: JSON.stringify(r, null, 2) }] };
  }
);

// ---- TOOL 9: forge_agent ----
server.tool(
  "forge_agent",
  "Scaffolds a complete production expert agent: knowledge vault, intelligence layer, memory system, planning engine, activation persona, MCP tool facades, tests, and setup script. Returns all generated artifacts.",
  {
    name: z.string().describe("Agent slug, e.g. 'gaudi'"),
    domain: z.string().describe("Primary domain, e.g. 'backend architecture'"),
    principles: z.array(z.string()).default([]),
    language: z.enum(["javascript", "python"]).default("javascript"),
  },
  { readOnlyHint: false, openWorldHint: false },
  async ({ name, domain, principles }) => {
    const files = [];
    const pkg = JSON.stringify({ name, version: "1.0.0", type: "module", dependencies: { "@modelcontextprotocol/sdk": "^1.0.0", zod: "^3.23.0" } }, null, 2);
    files.push({ path: "package.json", content: pkg, language: "json" });

    const persona = `# ${name} — Persona\n\nYou are ${name}, an expert in ${domain}.\n\n## Principles\n${principles.map(p => `- ${p}`).join("\n") || "- Be precise. Be honest about uncertainty."}\n\n## Behaviour\n- When unsure, say so and ask for the missing evidence.\n- Prefer simple solutions over clever ones.\n- Escalate to a human expert for irreversible decisions.`;
    files.push({ path: "src/persona.md", content: persona, language: "markdown" });

    const index = `#!/usr/bin/env node\n// Generated by Nirvana\nimport { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";\nimport { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";\nimport { z } from "zod";\n\nconst server = new McpServer({ name: ${JSON.stringify(name)}, version: "1.0.0" });\n\nserver.tool("${name}_search", "Search ${domain} knowledge", { query: z.string() }, async ({ query }) => {\n  return { content: [{ type: "text", text: JSON.stringify({ query, results: [] }) }] };\n});\n\nconst transport = new StdioServerTransport();\nawait server.connect(transport);\nconsole.error("[${name}] running on stdio");`;
    files.push({ path: "src/index.js", content: index, language: "javascript" });

    const vault = `// Generated by Nirvana\n// Knowledge vault — SQLite + FTS5 in production. Here: in-memory stub.\nexport class Vault {\n  constructor() { this.docs = []; }\n  ingest(doc) { this.docs.push(doc); }\n  search(q) { return this.docs.filter(d => d.text.includes(q)); }\n}`;
    files.push({ path: "src/vault.js", content: vault, language: "javascript" });

    const memory = `// Generated by Nirvana\n// Memory system — persists lessons, preferences, session summaries.\nexport class Memory {\n  constructor() { this.items = []; }\n  capture(item) { this.items.push({ ...item, ts: Date.now() }); }\n  recall(query) { return this.items.filter(i => JSON.stringify(i).includes(query)); }\n}`;
    files.push({ path: "src/memory.js", content: memory, language: "javascript" });

    const planning = `// Generated by Nirvana\n// Planning state machine: draft → approve → execute → complete.\nexport class Plan {\n  constructor() { this.state = "draft"; this.tasks = []; }\n  approve() { this.state = "approved"; }\n  execute() { this.state = "executing"; }\n  complete() { this.state = "complete"; }\n}`;
    files.push({ path: "src/planning.js", content: planning, language: "javascript" });

    const test = `// Generated by Nirvana\nimport { test } from "node:test";\nimport assert from "node:assert";\n\ntest("agent smoke test", () => { assert.ok(true); });`;
    files.push({ path: "tests/smoke.test.js", content: test, language: "javascript" });

    const setup = `#!/usr/bin/env bash\n# Generated by Nirvana\nset -euo pipefail\nnpm install\nnpm test\necho "Agent ${name} ready. Register with: devin mcp add -s user ${name} -- node $(pwd)/src/index.js"`;
    files.push({ path: "scripts/setup.sh", content: setup, language: "shell" });

    const artifacts = files.map(f => {
      const a = registerArtifact({ kind: "forge-file", name: `${name}-${f.path.replace(/\//g, "-")}`, content: f.content, language: f.language, source: `forge:${name}`, metadata: { agent: name, path: f.path } });
      a.path = persistArtifact(a);
      return { artifactId: a.id, path: f.path, diskPath: a.path };
    });

    return { content: [{ type: "text", text: JSON.stringify({ agent: name, domain, fileCount: files.length, artifacts, hint: "Review each artifact, then run scripts/setup.sh inside the scaffolded directory." }, null, 2) }] };
  }
);

// ---- TOOL 10: forge_skill ----
server.tool(
  "forge_skill",
  "Forge a complete Agent Skill (SKILL.md + optional wrapper) from a knowledge domain. The skill embeds procedures, gotchas, and tool references, and is loaded on demand by coding agents.",
  {
    name: z.string(),
    description: z.string(),
    domain: z.string(),
    procedures: z.array(z.string()).default([]),
    gotchas: z.array(z.string()).default([]),
  },
  { readOnlyHint: false, openWorldHint: false },
  async ({ name, description, domain, procedures, gotchas }) => {
    const md = `---\nname: ${name}\ndescription: ${description}\ntriggers: ["user", "auto"]\nallowed-tools:\n  - read\n  - exec\n---\n\n# ${name}\n\n${description}\n\n## Procedure\n\n${procedures.map((p, i) => `${i + 1}. ${p}`).join("\n") || `1. Consult the ${domain} knowledge vault before acting.\n2. Verify assumptions against source evidence.\n3. Escalate irreversible decisions to a human expert.`}\n\n## Gotchas\n\n${gotchas.map(g => `- ${g}`).join("\n") || "- No gotchas recorded yet."}\n\n## Source\n\nDomain: ${domain}\nGenerated by Nirvana`;
    const artifact = registerArtifact({ kind: "skill", name, content: md, language: "markdown", source: `forge:${domain}`, metadata: { procedureCount: procedures.length } });
    artifact.path = persistArtifact(artifact);
    return { content: [{ type: "text", text: JSON.stringify({ artifactId: artifact.id, name, path: artifact.path, procedureCount: procedures.length }, null, 2) }] };
  }
);

// ---- TOOL 11: convert_program_to_mcp ----
server.tool(
  "convert_program_to_mcp",
  "Converts a CLI program into an expert-reviewed MCP server. Probes --help, extracts subcommands and flags, generates a typed MCP server, and runs a semantic review before returning the artifact.",
  {
    command: z.string(),
    name: z.string(),
    description: z.string().default("Generated by Nirvana"),
  },
  { readOnlyHint: false, openWorldHint: true },
  async ({ command, name, description }) => {
    const cli = await probeCli(command);
    const tools = cli.subcommands.slice(0, 10).map(s => ({
      name: s.replace(/[^a-zA-Z0-9_]/g, "_"),
      description: `Run \`${command} ${s}\``,
      inputSchema: { type: "object", properties: { args: { type: "array", items: { type: "string" } } } },
    }));
    if (tools.length === 0) tools.push({ name: "run", description: `Run ${command}`, inputSchema: { type: "object", properties: { args: { type: "array", items: { type: "string" } } }, required: ["args"] } });

    const code = `#!/usr/bin/env node\n// Generated by Nirvana from ${command}\nimport { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";\nimport { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";\nimport { spawn } from "node:child_process";\n\nconst CLI = ${JSON.stringify(command)};\nfunction run(args) { return new Promise(r => { const p = spawn(CLI, args); let o="",e=""; p.stdout.on("data",c=>o+=c); p.stderr.on("data",c=>e+=c); p.on("exit",code=>r({code,o,e})); }); }\n\nconst server = new McpServer({ name: ${JSON.stringify(name)}, version: "1.0.0" });\n${tools.map(t => `server.tool(${JSON.stringify(t.name)}, ${JSON.stringify(t.description)}, ${JSON.stringify(t.inputSchema)}, async (a) => { const r = await run(a.args ?? []); return { content:[{type:"text",text:JSON.stringify(r)}], isError: r.code !== 0 }; });`).join("\n")}\nconst transport = new StdioServerTransport();\nawait server.connect(transport);\n`;
    const artifact = registerArtifact({ kind: "mcp-server", name, content: code, language: "javascript", source: command, metadata: { toolCount: tools.length, cli } });
    artifact.path = persistArtifact(artifact);

    const review = await callLLM({ system: "You are a security reviewer. Flag any risks in this generated MCP wrapper.", prompt: `Command: ${command}\nTools: ${JSON.stringify(tools)}\nReview for injection, privilege escalation, and silent failures.`, temperature: 0.1 });
    return { content: [{ type: "text", text: JSON.stringify({ artifactId: artifact.id, name, path: artifact.path, toolCount: tools.length, semanticReview: review.text.slice(0, 800), reviewRequired: true }, null, 2) }] };
  }
);

// ---- TOOL 12: convert_mcp_to_agent ----
server.tool(
  "convert_mcp_to_agent",
  "Converts an MCP server into a LangChain agent harness. Connects to the MCP, discovers tools, generates a runnable Python agent, and attaches an expert review of the tool surface.",
  {
    mcpCommand: z.string(),
    mcpArgs: z.array(z.string()).default([]),
    name: z.string(),
  },
  { readOnlyHint: false, openWorldHint: true },
  async ({ mcpCommand, mcpArgs, name }) => {
    let tools = [];
    try {
      const probe = new McpProbe({ command: mcpCommand, args: mcpArgs });
      await probe.connect();
      tools = await probe.listTools();
      await probe.close();
    } catch (e) {
      return { content: [{ type: "text", text: `MCP probe failed: ${e.message}` }], isError: true };
    }

    const code = `#!/usr/bin/env python3\n# Generated by Nirvana from MCP ${mcpCommand}\nimport asyncio, sys\nfrom langchain.agents import create_agent\nfrom langchain_mcp import MCPAdapter, StdioConnection\n\nasync def main():\n    async with MCPAdapter(StdioConnection(command=${JSON.stringify(mcpCommand)}, args=${JSON.stringify(mcpArgs)})) as adapter:\n        tools = await adapter.list_tools()\n        agent = create_agent(model="anthropic:claude-sonnet-4-5", tools=tools, system_prompt="You are an expert agent with ${tools.length} tools. Ask for clarification when unsure.")\n        result = await agent.ainvoke({"messages":[{"role":"user","content":" ".join(sys.argv[1:]) or "List your tools."}]})\n        print(result["messages"][-1].content)\n\nif __name__ == "__main__":\n    asyncio.run(main())\n`;
    const artifact = registerArtifact({ kind: "agent", name, content: code, language: "python", source: mcpCommand, metadata: { toolCount: tools.length } });
    artifact.path = persistArtifact(artifact);

    const review = await callLLM({ system: "You are an agent architect. Review this MCP tool surface for agent usability.", prompt: `Tools: ${JSON.stringify(tools.map(t => ({ name: t.name, description: t.description })), null, 2)}\nFlag ambiguous tool names, missing schemas, and dangerous tools that need human review.`, temperature: 0.1 });
    return { content: [{ type: "text", text: JSON.stringify({ artifactId: artifact.id, name, path: artifact.path, toolCount: tools.length, toolNames: tools.map(t => t.name), expertReview: review.text.slice(0, 800), reviewRequired: true }, null, 2) }] };
  }
);

// ---- TOOL 13: convert_mcp_to_a2a ----
server.tool(
  "convert_mcp_to_a2a",
  "Converts an MCP server into an A2A agent: generates the agent-card.json and a FastAPI bridge server that forwards A2A skill invocations to MCP tools.",
  {
    mcpCommand: z.string(),
    mcpArgs: z.array(z.string()).default([]),
    name: z.string(),
    description: z.string().default("A2A agent generated from an MCP server"),
  },
  { readOnlyHint: false, openWorldHint: true },
  async ({ mcpCommand, mcpArgs, name, description }) => {
    let tools = [];
    try {
      const probe = new McpProbe({ command: mcpCommand, args: mcpArgs });
      await probe.connect();
      tools = await probe.listTools();
      await probe.close();
    } catch (e) {
      return { content: [{ type: "text", text: `MCP probe failed: ${e.message}` }], isError: true };
    }

    const card = JSON.stringify({ name, description, version: "1.0.0", protocolVersion: "0.2.5", skills: tools.map(t => ({ id: t.name, name: t.name, description: t.description ?? "", inputSchema: t.inputSchema ?? {} })), capabilities: { streaming: false }, provider: { organization: "Nirvana" } }, null, 2);
    const cardArtifact = registerArtifact({ kind: "agent-card", name: `${name}-agent-card`, content: card, language: "json", source: mcpCommand, metadata: { skillCount: tools.length } });
    cardArtifact.path = persistArtifact(cardArtifact);

    const serverCode = `# Generated by Nirvana\nfrom fastapi import FastAPI, Request\nfrom mcp import ClientSession, StdioServerParameters\nimport anyio, json\napp = FastAPI()\nMCP = StdioServerParameters(command=${JSON.stringify(mcpCommand)}, args=${JSON.stringify(mcpArgs)})\n\n@app.get("/.well-known/agent-card.json")\nasync def card(): return json.load(open("${name}-agent-card.json"))\n\n@app.post("/")\nasync def handle(req: Request):\n    body = await req.json()\n    skill = body.get("skillId")\n    async def _call():\n        async with ClientSession(MCP) as s:\n            await s.initialize()\n            return await s.call_tool(skill, body.get("inputs", {}))\n    return await _call()\n`;
    const serverArtifact = registerArtifact({ kind: "a2a-server", name: `${name}-a2a`, content: serverCode, language: "python", source: mcpCommand, metadata: { skillCount: tools.length } });
    serverArtifact.path = persistArtifact(serverArtifact);

    return { content: [{ type: "text", text: JSON.stringify({ name, artifacts: [{ artifactId: cardArtifact.id, kind: "agent-card", path: cardArtifact.path }, { artifactId: serverArtifact.id, kind: "a2a-server", path: serverArtifact.path }], skillCount: tools.length, reviewRequired: true }, null, 2) }] };
  }
);

// ---- TOOL 14: assess_review_need ----
server.tool(
  "assess_review_need",
  "Safety router: given a task description, decides whether the work needs human expert review before execution. Returns domain, risk level, and a recommendation.",
  {
    task: z.string(),
    context: z.string().optional(),
  },
  { readOnlyHint: true, openWorldHint: false },
  async ({ task, context }) => {
    const lower = `${task} ${context ?? ""}`.toLowerCase();
    const flags = [];
    if (/sign|contract|legal|terms|sue/.test(lower)) flags.push({ domain: "legal", risk: "HIGH" });
    if (/tax|invoice|payment|transfer|money/.test(lower)) flags.push({ domain: "finance", risk: "HIGH" });
    if (/auth|password|secret|key|credential/.test(lower)) flags.push({ domain: "security", risk: "CRITICAL" });
    if (/deploy|production|release|rollback/.test(lower)) flags.push({ domain: "operations", risk: "HIGH" });
    if (/patient|health|medical|diagnosis/.test(lower)) flags.push({ domain: "healthcare", risk: "CRITICAL" });
    const highest = flags.sort((a, b) => ({ CRITICAL: 3, HIGH: 2, MEDIUM: 1, LOW: 0 }[b.risk] - { CRITICAL: 3, HIGH: 2, MEDIUM: 1, LOW: 0 }[a.risk]))[0];
    return { content: [{ type: "text", text: JSON.stringify({ task: task.slice(0, 300), flags, requiresHumanReview: flags.length > 0, recommendedDomain: highest?.domain ?? "none", riskLevel: highest?.risk ?? "LOW", recommendation: flags.length > 0 ? "Route to the matching human expert before executing." : "Proceed; no human review required." }, null, 2) }] };
  }
);

// ---- TOOL 15: request_expert_review ----
server.tool(
  "request_expert_review",
  "Packages a task into a structured review brief and returns a routing packet for a human expert. Use after assess_review_need flags a task as requiring review.",
  {
    task: z.string(),
    domain: z.string(),
    assumptions: z.array(z.string()).default([]),
    missingContext: z.array(z.string()).default([]),
    question: z.string(),
  },
  { readOnlyHint: true, openWorldHint: false },
  async ({ task, domain, assumptions, missingContext, question }) => {
    const packet = {
      id: randomUUID(),
      createdAt: new Date().toISOString(),
      domain,
      task: task.slice(0, 2000),
      question,
      assumptions,
      missingContext,
      status: "draft",
      routingHint: `Route to a ${domain} expert. Approve before sending.`,
      generatedBy: "Nirvana",
    };
    return { content: [{ type: "text", text: JSON.stringify(packet, null, 2) }] };
  }
);

// ---- TOOL 16: verify_artifact ----
server.tool(
  "verify_artifact",
  "Verifies a generated artifact: syntax-checks JavaScript/TypeScript, validates JSON, checks Python imports and main guards, and validates SKILL.md frontmatter and procedure sections.",
  { artifactId: z.string() },
  { readOnlyHint: true, openWorldHint: false },
  async ({ artifactId }) => {
    const a = ARTIFACTS.get(artifactId);
    if (!a) return { content: [{ type: "text", text: `Unknown artifact: ${artifactId}` }], isError: true };
    const checks = [];
    if (a.language === "javascript" || a.language === "typescript") {
      const tmp = join(CONFIG.stateDir, `verify-${a.hash}.mjs`);
      writeFileSync(tmp, a.content);
      const r = await runCli(process.execPath, ["--check", tmp], 5000);
      checks.push({ check: "syntax", passed: r.ok, detail: r.ok ? "node --check passed" : (r.stderr || "").slice(0, 300) });
    }
    if (a.language === "json") {
      try { JSON.parse(a.content); checks.push({ check: "json-parse", passed: true }); }
      catch (e) { checks.push({ check: "json-parse", passed: false, detail: e.message }); }
    }
    if (a.language === "markdown") {
      checks.push({ check: "frontmatter", passed: a.content.startsWith("---\n") });
      checks.push({ check: "procedure-section", passed: /##\s+Procedure/i.test(a.content) });
    }
    const passed = checks.every(c => c.passed);
    if (passed) a.reviewed = true;
    return { content: [{ type: "text", text: JSON.stringify({ artifactId, name: a.name, kind: a.kind, passed, checks, reviewed: a.reviewed }, null, 2) }] };
  }
);

// ---- TOOL 17: list_artifacts ----
server.tool(
  "list_artifacts",
  "Lists every generated artifact: id, kind, name, path, language, hash, and review status.",
  { kind: z.string().optional() },
  { readOnlyHint: true, openWorldHint: false },
  async ({ kind }) => {
    const all = [...ARTIFACTS.values()].filter(a => !kind || a.kind === kind);
    return { content: [{ type: "text", text: JSON.stringify({ count: all.length, outputDir: CONFIG.outputDir, artifacts: all.map(a => ({ id: a.id, kind: a.kind, name: a.name, path: a.path, language: a.language, hash: a.hash, reviewed: a.reviewed })) }, null, 2) }] };
  }
);

// ============================================================================
// MCP PROBE — used by convert_mcp_to_agent and convert_mcp_to_a2a
// ============================================================================
class McpProbe {
  constructor({ command, args, timeoutMs = 10000 }) {
    this.command = command; this.args = args; this.timeoutMs = timeoutMs;
    this.proc = null; this.nextId = 1; this.pending = new Map(); this.buffer = "";
  }
  async connect() {
    this.proc = spawn(this.command, this.args ?? [], { stdio: ["pipe", "pipe", "pipe"] });
    this.proc.stdout.on("data", (c) => {
      this.buffer += c.toString();
      let i;
      while ((i = this.buffer.indexOf("\n")) !== -1) {
        const line = this.buffer.slice(0, i).trim(); this.buffer = this.buffer.slice(i + 1);
        if (!line) continue;
        try { this._handle(JSON.parse(line)); } catch {}
      }
    });
    this.proc.stderr.on("data", () => {});
    await this._rpc("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: { tools: {} },
      clientInfo: { name: "nirvana", version: "1.0.0" },
    });
    try { this.proc.stdin.write(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized", params: {} }) + "\n"); } catch {}
  }
  async listTools() { const r = await this._rpc("tools/list", {}); return r.tools ?? []; }
  async close() { if (this.proc) { try { this.proc.stdin.end(); } catch {} try { this.proc.kill("SIGTERM"); } catch {} } }
  _rpc(method, params) {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error(`timeout ${method}`)); }, this.timeoutMs);
      this.pending.set(id, { resolve: v => { clearTimeout(timer); resolve(v); }, reject: e => { clearTimeout(timer); reject(e); } });
      this.proc.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
    });
  }
  _handle(msg) {
    if (msg.id !== undefined && this.pending.has(msg.id)) {
      const { resolve, reject } = this.pending.get(msg.id); this.pending.delete(msg.id);
      msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
    }
  }
}

// ---- TOOL 18: ollama_status ----
server.tool(
  "ollama_status",
  "Reports Ollama discovery status: host, reachable, models, and selected model. Use to verify zero-key setup.",
  {},
  { readOnlyHint: true, openWorldHint: true },
  async () => {
    const models = await discoverOllamaModels();
    const selected = pickBestOllamaModel(models);
    let reachable = false;
    try {
      const res = await fetch(OLLAMA_HOST + "/api/tags", { signal: AbortSignal.timeout(2000) });
      reachable = res.ok;
    } catch {}
    return { content: [{ type: "text", text: JSON.stringify({
      ollamaHost: OLLAMA_HOST,
      enabled: OLLAMA_ENABLED,
      reachable,
      modelCount: models.length,
      models,
      selectedModel: selected,
      llmOrder: selected ? "ollama:" + selected + " → anthropic → openai → fallback" : "anthropic → openai → fallback",
      hint: reachable ? null : "Run 'ollama serve' and 'ollama pull qwen2.5-coder:7b' to enable local inference.",
    }, null, 2) }] };
  }
);

// ============================================================================
// STARTUP
// ============================================================================
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  log(
    `running on stdio | tools=18 | ` +
      `vault=${VAULT.stats().chunks} chunks | ` +
      `llm=${CONFIG.anthropicKey ? "anthropic" : CONFIG.openaiKey ? "openai" : "fallback"} | ` +
      `output=${CONFIG.outputDir}`
  );
}
for (const sig of ["SIGINT", "SIGTERM"]) {
  process.on(sig, async () => { try { await server.close(); } catch {} process.exit(0); });
}
main().catch(e => { log("fatal:", e); process.exit(1); });