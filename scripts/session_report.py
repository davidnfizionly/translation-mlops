"""Cost and drift report for one endpoint demo session.

Cost (estimates): the endpoint bills per instance hour from creation to
deletion. The time spent answering requests (round trip measured by the local
proxy) is attributed to requests; the rest is idle time, stated separately.
The proxy's request count is cross-checked with the CloudWatch Invocations sum.

Drift signals: input length (words) and output/input word ratio of the live
requests, compared with the held-out test split references.

Usage: python scripts/session_report.py --created ISO --deleted ISO [--test data/processed/test.jsonl]
"""

import argparse
import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path

import boto3

REPO = Path(__file__).resolve().parents[1]
ENDPOINT = "translation-mlops-nllb-en-din-nus"
HOURLY_PRICE_USD = 0.736  # ml.g4dn.xlarge real-time hosting, us-east-1, AWS Price List API (USE1-Host), 2026-10-19
FIELD = {"dik_Latn": "din", "nus_Latn": "nuer"}


def parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def quantiles(values):
    if not values:
        return "n=0"
    q = statistics.quantiles(values, n=10) if len(values) > 1 else [values[0]] * 9
    return f"n={len(values)} mean {statistics.mean(values):.2f} p50 {statistics.median(values):.2f} p90 {q[8]:.2f}"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--created", required=True, help="endpoint creation time (UTC ISO)")
    p.add_argument("--deleted", required=True, help="endpoint deletion time (UTC ISO)")
    p.add_argument("--log", default=str(REPO / "console" / ".data" / "requests.jsonl"))
    p.add_argument("--test", default=str(REPO / "data" / "processed" / "test.jsonl"))
    p.add_argument("--region", default="us-east-1")
    args = p.parse_args()

    created, deleted = parse(args.created), parse(args.deleted)
    rows = [json.loads(line) for line in open(args.log, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if created <= parse(r["ts"]) <= deleted]
    ok = [r for r in rows if r["ok"]]
    billed = (deleted - created).total_seconds()
    busy = sum(r["ms"] for r in ok) / 1000
    per_second = HOURLY_PRICE_USD / 3600

    cw = boto3.Session(region_name=args.region).client("cloudwatch")
    points = cw.get_metric_statistics(
        Namespace="AWS/SageMaker", MetricName="Invocations",
        Dimensions=[{"Name": "EndpointName", "Value": ENDPOINT}, {"Name": "VariantName", "Value": "AllTraffic"}],
        StartTime=created - timedelta(minutes=5), EndTime=deleted + timedelta(minutes=5), Period=60, Statistics=["Sum"],
    )["Datapoints"]
    latency = cw.get_metric_statistics(
        Namespace="AWS/SageMaker", MetricName="ModelLatency",
        Dimensions=[{"Name": "EndpointName", "Value": ENDPOINT}, {"Name": "VariantName", "Value": "AllTraffic"}],
        StartTime=created - timedelta(minutes=5), EndTime=deleted + timedelta(minutes=5), Period=3600,
        ExtendedStatistics=["p50", "p90"],
    )["Datapoints"]

    print(f"Endpoint session {args.created} to {args.deleted}: {billed:.0f} s billed (creation to deletion, estimate)")
    print(f"Hourly price: {HOURLY_PRICE_USD} USD (ml.g4dn.xlarge real-time hosting, us-east-1)")
    print(f"Requests: {len(rows)} sent by the proxy, {len(ok)} answered; CloudWatch Invocations sum: {sum(d['Sum'] for d in points):.0f}")
    ms = [r["ms"] for r in ok]
    print(f"Round-trip latency ms (proxy): {quantiles(ms)}")
    for d in latency:
        print(f"ModelLatency ms (CloudWatch, per hour bucket {d['Timestamp']:%H:%M}): p50 {d['ExtendedStatistics']['p50'] / 1000:.0f}, p90 {d['ExtendedStatistics']['p90'] / 1000:.0f}")
    print(f"Time answering requests: {busy:.1f} s, {busy * per_second:.4f} USD attributed to requests (estimate)")
    if ok:
        print(f"Average per request: {busy * per_second / len(ok):.6f} USD (estimate)")
    print(f"Idle time: {billed - busy:.0f} s, {(billed - busy) * per_second:.4f} USD (estimate)")
    print(f"Endpoint total: {billed * per_second:.4f} USD (estimate; Cost Explorer comparison later, lags up to 24 h)")
    print()

    test = [json.loads(line) for line in open(args.test, encoding="utf-8")]
    print("Drift signals (live requests vs held-out test split references):")
    for tgt, field in FIELD.items():
        live = [r for r in ok if r["tgt_lang"] == tgt]
        ref_in = [len(t["english"].split()) for t in test]
        ref_ratio = [len(t[field].split()) / max(len(t["english"].split()), 1) for t in test]
        print(f"  {tgt} input words   live {quantiles([r['input_words'] for r in live])} | test {quantiles(ref_in)}")
        print(f"  {tgt} output/input  live {quantiles([r['output_words'] / max(r['input_words'], 1) for r in live])} | test {quantiles(ref_ratio)}")


if __name__ == "__main__":
    main()
