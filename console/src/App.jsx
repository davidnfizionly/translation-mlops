import { useCallback, useEffect, useState } from 'react';

const DATASET_URL = 'https://huggingface.co/datasets/dayomtechnologies/english-nuer-dinka-parallel-corpus';
const MODEL_URL = 'https://huggingface.co/facebook/nllb-200-distilled-600M';
const LICENSE_URL = 'https://creativecommons.org/licenses/by-nc/4.0/';
const LANGS = { dik_Latn: 'Dinka', nus_Latn: 'Nuer' };
const DIRECTIONS = { 'eng_Latn-dik_Latn': 'English to Dinka', 'eng_Latn-nus_Latn': 'English to Nuer' };
// Demonstration inputs only: their outputs have not been checked by a speaker.
const DEMO_SENTENCES = [
  'Boil water before drinking.',
  'Wash your hands with soap and clean water.',
  'The clinic opens at eight in the morning.',
];

const usd = (v, digits = 4) => `${v.toFixed(digits)} USD`;
const minutes = (s) => `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;

async function api(pathname, options) {
  const res = await fetch(`/api${pathname}`, options);
  return { ok: res.ok, body: await res.json() };
}

function Section({ title, children, aside }) {
  return (
    <section className="rounded-xl border-2 border-slate-300 bg-white p-6 shadow-sm">
      <div className="mb-4 flex items-baseline justify-between gap-4">
        <h2 className="text-2xl font-bold text-slate-900">{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

function Attribution() {
  return (
    <div className="rounded-xl border-2 border-sky-800 bg-sky-50 p-5 text-lg text-slate-900">
      <p>
        <strong>Dataset:</strong>{' '}
        <a className="font-semibold text-sky-900 underline" href={DATASET_URL}>English-Nuer-Dinka Parallel Corpus</a> by Tajnaam,{' '}
        <a className="text-sky-900 underline" href={LICENSE_URL}>CC BY-NC 4.0</a>.{' '}
        <strong>Base model:</strong>{' '}
        <a className="font-semibold text-sky-900 underline" href={MODEL_URL}>NLLB-200 distilled 600M</a> by Meta,{' '}
        <a className="text-sky-900 underline" href={LICENSE_URL}>CC BY-NC 4.0</a>.
      </p>
      <p className="mt-2">
        The fine-tuned model is not published because of these licenses. This page runs locally with no public service.
      </p>
    </div>
  );
}

function Warning() {
  return (
    <div className="rounded-xl border-2 border-amber-700 bg-amber-50 p-5 text-lg text-slate-900" role="note">
      <p>
        <strong>Read before using any output.</strong> The dataset references are very probably machine generated (not
        confirmed by the dataset author). BLEU and chrF measure agreement with those references, not the real quality for a
        Dinka or Nuer speaker. The results are measured on a test split from the same corpus and do not generalize to
        real-world sentences. No speaker has reviewed these outputs: they are not reliable translations and must not be
        used for health, legal or safety information.
      </p>
    </div>
  );
}

function StatusPill({ status }) {
  if (!status) return <span className="text-lg text-slate-600">Checking endpoint...</span>;
  const s = status.endpoint.status;
  const cls = status.live ? 'bg-emerald-700 text-white' : s === 'NotDeployed' ? 'bg-slate-700 text-white' : 'bg-amber-600 text-white';
  const label = status.live ? 'Endpoint InService' : s === 'NotDeployed' ? 'Endpoint not deployed: stored outputs only' : `Endpoint ${s}`;
  return <span className={`rounded-full px-4 py-1 text-lg font-semibold ${cls}`}>{label}</span>;
}

function LiveTranslation({ status, samples, onRequest }) {
  const [tgt, setTgt] = useState('dik_Latn');
  const [text, setText] = useState(DEMO_SENTENCES[0]);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [batch, setBatch] = useState(null);

  const translate = useCallback(async (input, lang) => {
    const { ok, body } = await api('/translate', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text: input, tgt_lang: lang }),
    });
    onRequest();
    return { ok, ...body };
  }, [onRequest]);

  const run = async () => {
    setBusy(true);
    setResult(null);
    setResult({ input: text, lang: tgt, ...(await translate(text, tgt)) });
    setBusy(false);
  };

  const runBatch = async () => {
    setBusy(true);
    const rows = samples.filter((s) => s.direction === `eng_Latn-${tgt}`);
    const done = [];
    for (const row of rows) {
      done.push({ source: row.source, ...(await translate(row.source, tgt)) });
      setBatch({ total: rows.length, done: [...done] });
    }
    setBusy(false);
  };

  return (
    <Section title="Live translation" aside={<StatusPill status={status} />}>
      <div className="mb-4 flex flex-wrap gap-3" role="group" aria-label="Target language">
        {Object.entries(LANGS).map(([code, name]) => (
          <button key={code} onClick={() => setTgt(code)} aria-pressed={tgt === code}
            className={`rounded-lg border-2 px-5 py-2 text-xl font-semibold ${tgt === code ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-400 bg-white text-slate-900'}`}>
            English to {name}
          </button>
        ))}
      </div>
      <p className="mb-2 text-lg text-slate-700">Demonstration sentences (outputs not checked by a speaker):</p>
      <div className="mb-4 flex flex-wrap gap-2">
        {DEMO_SENTENCES.map((s) => (
          <button key={s} onClick={() => setText(s)} className="rounded-lg border-2 border-slate-400 px-3 py-1 text-lg text-slate-900 hover:bg-slate-100">{s}</button>
        ))}
      </div>
      <label className="mb-2 block text-lg font-semibold text-slate-900" htmlFor="source">English input</label>
      <textarea id="source" value={text} onChange={(e) => setText(e.target.value)} maxLength={1000} rows={3}
        className="w-full rounded-lg border-2 border-slate-400 p-3 text-xl text-slate-900" />
      <div className="mt-4 flex flex-wrap gap-3">
        <button onClick={run} disabled={busy || !text.trim()}
          className="rounded-lg bg-sky-800 px-6 py-3 text-xl font-bold text-white disabled:bg-slate-400">Translate</button>
        <button onClick={runBatch} disabled={busy}
          className="rounded-lg border-2 border-sky-800 px-6 py-3 text-xl font-bold text-sky-900 disabled:border-slate-400 disabled:text-slate-500">
          Send the 10 stored test sentences
        </button>
      </div>
      {result && (
        <div className="mt-5 rounded-lg border-2 border-slate-300 bg-slate-50 p-4" data-testid="result">
          {result.ok ? (
            <>
              <p className="text-lg text-slate-700">{LANGS[result.lang]} output ({result.mode === 'live' ? 'live endpoint' : 'stored evaluation output'}):</p>
              <p className="mt-1 text-2xl text-slate-900" lang={result.lang === 'dik_Latn' ? 'din' : 'nus'}>{result.translation}</p>
              {result.mode === 'live'
                ? <p className="mt-2 text-lg text-slate-700">Round trip {result.ms} ms; estimated cost of this request {usd(result.costUsd, 6)} (estimate).</p>
                : <p className="mt-2 text-lg text-slate-700">{result.note}</p>}
            </>
          ) : <p className="text-xl font-semibold text-red-800">{result.error}</p>}
        </div>
      )}
      {batch && (
        <p className="mt-4 text-lg text-slate-800" data-testid="batch">
          Sent {batch.done.length} of {batch.total}: {batch.done.filter((r) => r.ok).length} answered
          {batch.done.some((r) => r.mode === 'live') ? ' by the live endpoint' : ' from stored outputs'}.
        </p>
      )}
    </Section>
  );
}

function CostPanel({ status }) {
  if (!status) return null;
  const rows = status.endpoint.createdAt ? [
    ['Instance', `${status.instanceType}, ${usd(status.hourlyPriceUsd, 3)} per hour`],
    ['Endpoint billed time so far (from creation)', minutes(status.billedSeconds)],
    ['Live requests', String(status.requests)],
    ['Time spent answering requests', `${status.busySeconds.toFixed(1)} s`],
    ['Cost attributed to requests', usd(status.requestCostUsd)],
    ['Average per request', status.requests ? usd(status.requestCostUsd / status.requests, 6) : '-'],
    ['Idle cost (endpoint up, no request)', usd(status.idleCostUsd)],
    ['Total endpoint cost so far', usd(status.totalCostUsd)],
  ] : [
    ['Instance', `${status.instanceType}, ${usd(status.hourlyPriceUsd, 3)} per hour while the endpoint exists`],
    ['Endpoint', 'not deployed: 0 USD per hour now'],
  ];
  return (
    <Section title="Estimated cost (estimate)">
      <table className="w-full text-lg">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k} className="border-b border-slate-200">
              <td className="py-2 pr-4 text-slate-700">{k}</td>
              <td className="py-2 text-right font-semibold text-slate-900" data-testid={k}>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-base text-slate-700">
        Estimate: instance hourly price times request time, plus idle time stated separately. Billing is counted from the
        endpoint creation time, so start-up minutes count as idle. Compared afterwards with Cost Explorer, which lags up to 24 hours.
      </p>
    </Section>
  );
}

function Evaluation({ evaluation }) {
  const [direction, setDirection] = useState('eng_Latn-dik_Latn');
  if (!evaluation) return null;
  const { metrics, samples } = evaluation;
  return (
    <Section title="Before and after fine-tuning (stored evaluation)">
      <table className="mb-3 w-full text-lg">
        <thead>
          <tr className="border-b-2 border-slate-400 text-left text-slate-900">
            <th className="py-2">Direction</th><th>Base BLEU</th><th>Fine-tuned BLEU</th><th>Base chrF</th><th>Fine-tuned chrF</th><th>Looping outputs (base, fine-tuned)</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(metrics.directions).map(([d, m]) => (
            <tr key={d} className="border-b border-slate-200 text-slate-900">
              <td className="py-2 font-semibold">{DIRECTIONS[d]}</td>
              <td>{m.base_bleu.toFixed(2)}</td><td className="font-bold">{m.finetuned_bleu.toFixed(2)}</td>
              <td>{m.base_chrf.toFixed(2)}</td><td className="font-bold">{m.finetuned_chrf.toFixed(2)}</td>
              <td>{metrics.loops[d].base}%, {metrics.loops[d].finetuned}%</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-base text-slate-700">
        {metrics.test_rows} held-out test sentences per direction, same decoding for both models (beam {metrics.decoding.num_beams},
        at most {metrics.decoding.max_new_tokens} new tokens, {metrics.decoding.dtype}). sacreBLEU signatures: BLEU {metrics.signatures.bleu};
        chrF {metrics.signatures.chrf}. Scores measure agreement with very probably machine-generated references.
      </p>
      <div className="my-4 flex gap-3" role="group" aria-label="Sample direction">
        {Object.entries(DIRECTIONS).map(([d, name]) => (
          <button key={d} onClick={() => setDirection(d)} aria-pressed={direction === d}
            className={`rounded-lg border-2 px-4 py-2 text-lg font-semibold ${direction === d ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-400 text-slate-900'}`}>{name}</button>
        ))}
      </div>
      <p className="mb-3 text-base text-slate-700">
        10 test sentences per direction, not hand-picked: 5 drawn with a fixed seed from the best quarter and 5 from the worst
        quarter by sentence chrF of the fine-tuned output. Sample pairs from the English-Nuer-Dinka Parallel Corpus by Tajnaam (CC BY-NC 4.0).
      </p>
      <div className="space-y-3">
        {samples.filter((s) => s.direction === direction).map((s) => (
          <div key={`${s.direction}-${s.row_id}`} className="rounded-lg border-2 border-slate-300 p-4 text-lg">
            <p className="mb-1 font-semibold text-slate-900">
              <span className={`mr-2 rounded px-2 py-0.5 text-white ${s.label === 'good' ? 'bg-emerald-700' : 'bg-red-800'}`}>{s.label === 'good' ? 'Best quarter' : 'Worst quarter'}</span>
              sentence chrF {s.sentence_chrf}{s.looping ? ' - output loops' : ''}
            </p>
            <dl className="grid grid-cols-[10rem_1fr] gap-x-3 gap-y-1 text-slate-900">
              <dt className="text-slate-700">English</dt><dd>{s.source}</dd>
              <dt className="text-slate-700">Reference</dt><dd>{s.reference}</dd>
              <dt className="text-slate-700">Base model</dt><dd className="break-words">{s.base}</dd>
              <dt className="text-slate-700">Fine-tuned</dt><dd className="font-semibold break-words">{s.finetuned}</dd>
            </dl>
          </div>
        ))}
      </div>
    </Section>
  );
}

export default function App() {
  const [status, setStatus] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const refresh = useCallback(() => { api('/status').then((r) => setStatus(r.body)); }, []);

  useEffect(() => {
    api('/evaluation').then((r) => setEvaluation(r.body));
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [refresh]);

  return (
    <main className="mx-auto max-w-6xl space-y-6 p-6">
      <h1 className="text-4xl font-bold text-slate-900">English to Dinka and Nuer: fine-tuned NLLB-200, local demo</h1>
      <Attribution />
      <Warning />
      <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <LiveTranslation status={status} samples={evaluation?.samples ?? []} onRequest={refresh} />
        <CostPanel status={status} />
      </div>
      <Evaluation evaluation={evaluation} />
    </main>
  );
}
