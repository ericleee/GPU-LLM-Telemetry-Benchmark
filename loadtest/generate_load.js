// k6 load test — ramps concurrent users against the inference server so latency,
// throughput, and queue depth bend under load (and the GPU heats up).
//
//   k6 run loadtest\generate_load.js                 # full ramp 1 -> 5 -> 10 -> 20
//   $env:SMOKE=1; k6 run loadtest\generate_load.js   # quick 10s sanity run
//   $env:TARGET="http://127.0.0.1:8000"; ...         # override target
//   $env:STAGE_SECS="60s"; ...                        # longer plateaus (more thermal buildup)
//
// Uses 127.0.0.1 (not localhost) so it hits the server's IPv4 bind directly.

import http from 'k6/http';
import { check } from 'k6';
import { Trend, Counter } from 'k6/metrics';

// Server-reported metrics surfaced into the k6 summary.
const tokensPerSecond = new Trend('server_tokens_per_second');
const generatedTokens = new Counter('server_generated_tokens');

const BASE = __ENV.TARGET || 'http://127.0.0.1:8000';
const PROMPT = __ENV.PROMPT ||
  'Explain what a GPU does and why it is useful for AI, in two sentences.';
const MAX_TOKENS = Number(__ENV.MAX_TOKENS || 64);
const SMOKE = __ENV.SMOKE === '1';
const STAGE = __ENV.STAGE_SECS || '40s';

// Ramp profile: each step holds long enough to read a steady-state plateau in
// Grafana, then a final hold at 20 VUs to let the card heat toward throttling.
const fullStages = [
  { duration: STAGE, target: 1 },
  { duration: STAGE, target: 5 },
  { duration: STAGE, target: 10 },
  { duration: STAGE, target: 20 },
  { duration: '60s', target: 20 },   // sustained peak — watch for thermal throttle
  { duration: '15s', target: 0 },    // ramp down
];
const smokeStages = [
  { duration: '5s', target: 2 },
  { duration: '5s', target: 0 },
];

export const options = {
  scenarios: {
    ramp: {
      executor: 'ramping-vus',
      startVUs: 1,
      stages: SMOKE ? smokeStages : fullStages,
      gracefulRampDown: '15s',
    },
  },
  thresholds: {
    // Meaningful pass/fail: requests should succeed. Latency percentiles are
    // recorded for analysis (set a duration threshold here after a baseline run).
    http_req_failed: ['rate<0.05'],
  },
};

export default function () {
  const payload = JSON.stringify({ prompt: PROMPT, max_tokens: MAX_TOKENS });
  const params = {
    headers: { 'Content-Type': 'application/json' },
    timeout: '120s', // requests queue behind the GPU semaphore under load
  };

  const res = http.post(`${BASE}/generate`, payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
    'produced tokens': (r) => {
      try { return JSON.parse(r.body).generated_tokens > 0; } catch (_) { return false; }
    },
  });

  if (res.status === 200) {
    try {
      const body = JSON.parse(res.body);
      if (body.tokens_per_s) tokensPerSecond.add(body.tokens_per_s);
      if (body.generated_tokens) generatedTokens.add(body.generated_tokens);
    } catch (_) { /* ignore parse errors */ }
  }
  // No sleep: each VU sends back-to-back so N VUs => up to N concurrent requests,
  // which is what builds queue depth behind the GPU-concurrency semaphore.
}
