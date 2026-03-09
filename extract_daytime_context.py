"""
Extract daytime context from Bedrock batch query output JSONL.

Input format (Bedrock batch output):
  {"recordId": "2019-01-01_night", "modelOutput": {"content": [
      {"type": "thinking", ...},
      {"type": "text", "text": "today is 2019-01-01, likely categories: ..."}
  ]}}

Output format (compatible with load_date_context()):
  {"recordId": "2019-01-01_night", "modelOutput": {"content": [
      {"text": "today is 2019-01-01, likely categories: ..."}
  ]}}
"""

import json
import argparse


def extract(input_path, output_path):
    records = []
    skipped = 0
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            record_id = data.get("recordId", "")
            content_list = data.get("modelOutput", {}).get("content", [])

            # Find the "text" type entry (skip "thinking")
            text_value = None
            for entry in content_list:
                if entry.get("type") == "text":
                    text_value = entry.get("text", "")
                    break

            if not text_value:
                skipped += 1
                continue

            records.append({
                "recordId": record_id,
                "modelOutput": {"content": [{"text": text_value}]},
            })

    records.sort(key=lambda r: r["recordId"])

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Extracted {len(records)} records, skipped {skipped}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Bedrock batch output JSONL")
    parser.add_argument("--output", required=True, help="Cleaned context JSONL")
    args = parser.parse_args()
    extract(args.input, args.output)