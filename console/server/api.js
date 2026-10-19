// Dev-only API proxy for the translation widget (Vite middleware, never shipped to the browser).
// - Calls the SageMaker endpoint with the local AWS CLI credentials; no public API exists.
// - Without a deployed endpoint (or with CONSOLE_OFFLINE=1) it serves stored evaluation outputs only.
// - Logs every live request to console/.data/requests.jsonl and publishes drift and cost metrics to CloudWatch.
// - Cost per request is an estimate: request time x the hourly instance price; idle time is reported separately.
// Account IDs and ARNs are masked in anything returned to the browser.

import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fromNodeProviderChain } from '@aws-sdk/credential-providers';
import { SageMakerClient, DescribeEndpointCommand } from '@aws-sdk/client-sagemaker';
import { SageMakerRuntimeClient, InvokeEndpointCommand } from '@aws-sdk/client-sagemaker-runtime';
import { CloudWatchClient, PutMetricDataCommand } from '@aws-sdk/client-cloudwatch';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REGION = 'us-east-1';
const ENDPOINT = 'translation-mlops-nllb-en-din-nus';
// ml.g4dn.xlarge real-time hosting in us-east-1, AWS Price List API (USE1-Host:ml.g4dn.xlarge), 2026-10-19.
const HOURLY_PRICE_USD = 0.736;
const INSTANCE_TYPE = 'ml.g4dn.xlarge';
const TARGETS = new Set(['dik_Latn', 'nus_Latn']);
const MAX_CHARS = 1000;
const LOG = path.resolve(HERE, '../.data/requests.jsonl');
const OFFLINE = process.env.CONSOLE_OFFLINE === '1';

const samples = JSON.parse(readFileSync(path.join(HERE, 'samples.json'), 'utf8'));
const metrics = JSON.parse(readFileSync(path.join(HERE, 'metrics.json'), 'utf8'));

const credentials = fromNodeProviderChain();
const sagemaker = new SageMakerClient({ region: REGION, credentials });
const runtime = new SageMakerRuntimeClient({ region: REGION, credentials });
const cloudwatch = new CloudWatchClient({ region: REGION, credentials });

const redact = (text) => String(text)
  .replace(/arn:aws[\w-]*:[^\s"']+/g, '<arn>')
  .replace(/(?<![0-9a-zA-Z])\d{12}(?![0-9a-zA-Z])/g, '<account-id>');
const words = (text) => text.trim().split(/\s+/).filter(Boolean).length;

let cached = { at: 0, value: null };
async function endpointState() {
  if (OFFLINE) return { status: 'NotDeployed' };
  if (Date.now() - cached.at < 10000) return cached.value;
  let value;
  try {
    const out = await sagemaker.send(new DescribeEndpointCommand({ EndpointName: ENDPOINT }));
    value = { status: out.EndpointStatus, createdAt: out.CreationTime.toISOString() };
  } catch (err) {
    value = err.name === 'ValidationException' ? { status: 'NotDeployed' } : { status: 'Unknown', error: redact(err.message) };
  }
  cached = { at: Date.now(), value };
  return value;
}

function readLog(sinceIso) {
  if (!existsSync(LOG)) return [];
  return readFileSync(LOG, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l)).filter((r) => !sinceIso || r.ts >= sinceIso);
}

async function status() {
  const ep = await endpointState();
  const live = ep.status === 'InService';
  const base = { endpoint: ep, live, instanceType: INSTANCE_TYPE, hourlyPriceUsd: HOURLY_PRICE_USD };
  if (!ep.createdAt) return { ...base, requests: 0, busySeconds: 0, billedSeconds: 0, requestCostUsd: 0, idleCostUsd: 0, totalCostUsd: 0 };
  const rows = readLog(ep.createdAt).filter((r) => r.ok);
  const busySeconds = rows.reduce((s, r) => s + r.ms / 1000, 0);
  // Upper bound: billing is counted from the endpoint creation time, which includes the start-up minutes.
  const billedSeconds = (Date.now() - Date.parse(ep.createdAt)) / 1000;
  const perSecond = HOURLY_PRICE_USD / 3600;
  return {
    ...base,
    requests: rows.length,
    busySeconds,
    billedSeconds,
    requestCostUsd: busySeconds * perSecond,
    idleCostUsd: Math.max(billedSeconds - busySeconds, 0) * perSecond,
    totalCostUsd: billedSeconds * perSecond,
  };
}

async function publishMetrics(entry) {
  const dims = [{ Name: 'EndpointName', Value: ENDPOINT }];
  const data = [
    { MetricName: 'InputWords', Value: entry.input_words, Unit: 'Count' },
    { MetricName: 'OutputInputRatio', Value: entry.output_words / Math.max(entry.input_words, 1), Unit: 'None' },
    { MetricName: 'EstimatedRequestCostUSD', Value: entry.cost_usd, Unit: 'None' },
  ].map((m) => ({ ...m, Dimensions: dims }));
  try {
    await cloudwatch.send(new PutMetricDataCommand({ Namespace: 'TranslationMlops', MetricData: data }));
  } catch (err) {
    console.error('PutMetricData failed:', redact(err.message));
  }
}

async function translate(body) {
  const text = typeof body.text === 'string' ? body.text.trim() : '';
  const tgt = body.tgt_lang;
  if (!text || text.length > MAX_CHARS) return [400, { error: `text must be 1 to ${MAX_CHARS} characters` }];
  if (!TARGETS.has(tgt)) return [400, { error: 'tgt_lang must be dik_Latn or nus_Latn' }];

  const ep = await endpointState();
  if (ep.status !== 'InService') {
    const stored = samples.find((s) => s.source === text && s.direction === `eng_Latn-${tgt}`);
    if (stored) return [200, { translation: stored.finetuned, mode: 'stored', note: 'Stored output from the evaluation run (endpoint not deployed).' }];
    return [503, { error: `Endpoint not deployed (status ${ep.status}). Live translation runs only during the demo session; stored test sentences still work.` }];
  }

  const started = performance.now();
  let entry;
  try {
    const out = await runtime.send(new InvokeEndpointCommand({
      EndpointName: ENDPOINT,
      ContentType: 'application/json',
      Accept: 'application/json',
      Body: JSON.stringify({ inputs: [text], parameters: { tgt_lang: tgt, num_beams: 4, max_new_tokens: 200 } }),
    }));
    const ms = performance.now() - started;
    const result = JSON.parse(Buffer.from(out.Body).toString('utf8'));
    entry = {
      ts: new Date().toISOString(), ok: true, tgt_lang: tgt, ms: Math.round(ms),
      input_words: words(text), output_words: words(result.translations[0]),
      cost_usd: (ms / 3600000) * HOURLY_PRICE_USD,
    };
    log(entry);
    publishMetrics(entry);
    return [200, { translation: result.translations[0], mode: 'live', ms: entry.ms, costUsd: entry.cost_usd }];
  } catch (err) {
    entry = { ts: new Date().toISOString(), ok: false, tgt_lang: tgt, ms: Math.round(performance.now() - started), error: err.name };
    log(entry);
    return [502, { error: redact(err.message) }];
  }
}

function log(entry) {
  mkdirSync(path.dirname(LOG), { recursive: true });
  appendFileSync(LOG, JSON.stringify(entry) + '\n');
}

const readBody = (req) => new Promise((resolve, reject) => {
  let data = '';
  req.on('data', (c) => { data += c; if (data.length > 10000) req.destroy(); });
  req.on('end', () => { try { resolve(data ? JSON.parse(data) : {}); } catch (e) { reject(e); } });
  req.on('error', reject);
});

export function translationApi() {
  return {
    name: 'translation-api',
    configureServer(server) {
      server.middlewares.use('/api', async (req, res) => {
        const send = (code, payload) => {
          res.statusCode = code;
          res.setHeader('content-type', 'application/json');
          res.end(JSON.stringify(payload));
        };
        try {
          if (req.method === 'GET' && req.url === '/status') return send(200, await status());
          if (req.method === 'GET' && req.url === '/evaluation') return send(200, { metrics, samples });
          if (req.method === 'POST' && req.url === '/translate') return send(...(await translate(await readBody(req))));
          send(404, { error: 'not found' });
        } catch (err) {
          send(500, { error: redact(err.message) });
        }
      });
    },
  };
}
