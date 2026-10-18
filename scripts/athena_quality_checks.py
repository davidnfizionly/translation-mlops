"""Run the named Athena data quality queries of the data stack and print the results.

Uses the CLI credentials (AWS_PROFILE). Prints, for each query, the result
table, the data scanned and the estimated cost at 5 USD per TB scanned
(Athena bills at least 10 MB per query).
"""

import argparse
import sys
import time

import boto3

PRICE_PER_TB = 5.0
MIN_BYTES = 10 * 1024 * 1024


def run(athena, workgroup, database, name, sql):
    qid = athena.start_query_execution(
        QueryString=sql, WorkGroup=workgroup, QueryExecutionContext={"Database": database}
    )["QueryExecutionId"]
    while True:
        ex = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        state = ex["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(2)
    scanned = ex.get("Statistics", {}).get("DataScannedInBytes", 0)
    billed = max(scanned, MIN_BYTES)
    print(f"== {name}  state {state}  scanned {scanned / 1e6:.1f} MB  est. cost {billed / 1e12 * PRICE_PER_TB:.5f} USD")
    if state != "SUCCEEDED":
        print("   " + ex["Status"].get("StateChangeReason", ""))
        return billed, False
    rows = athena.get_query_results(QueryExecutionId=qid)["ResultSet"]["Rows"]
    for row in rows:
        print("   " + " | ".join(c.get("VarCharValue", "") for c in row["Data"]))
    return billed, True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workgroup", default="translation-mlops")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    athena = boto3.client("athena", region_name=args.region)
    ids = athena.list_named_queries(WorkGroup=args.workgroup)["NamedQueryIds"]
    queries = athena.batch_get_named_query(NamedQueryIds=ids)["NamedQueries"]
    total, ok = 0, True
    for q in sorted(queries, key=lambda q: q["Name"]):
        billed, success = run(athena, args.workgroup, q["Database"], q["Name"], q["QueryString"])
        total += billed
        ok = ok and success
    print(f"\nTotal billed bytes {total / 1e6:.1f} MB, estimated cost {total / 1e12 * PRICE_PER_TB:.5f} USD")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
