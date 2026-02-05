#!/usr/bin/env python3
"""
Script to replace the content text with record-ID only in JSONL files.

This transforms entries from:
    {"recordId": "2024-01-15", "modelOutput": {"content": [{"text": "Some description..."}]}}
To:
    {"recordId": "2024-01-15", "modelOutput": {"content": [{"text": "2024-01-15"}]}}

Usage:
    python replace_content_with_recordid.py <input_jsonl> <output_jsonl>

Example:
    python replace_content_with_recordid.py \
        ./ID_generation/preprocessing/raw_data/Amazon/Beauty_date_context.jsonl \
        ./ID_generation/preprocessing/raw_data/Amazon/Beauty_date_context_id_only.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def replace_content_with_recordid(input_path: str, output_path: str) -> None:
    """
    Read a JSONL file and replace content text with recordId.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON, skipping. Error: {e}", 
                      file=sys.stderr)
                skipped_count += 1
                continue
            
            # Get the recordId
            record_id = data.get("recordId")
            if not record_id:
                print(f"Warning: Line {line_num} has no 'recordId', skipping.", 
                      file=sys.stderr)
                skipped_count += 1
                continue
            
            # Replace the content text with recordId
            # Navigate to modelOutput.content[0].text and replace it
            if "modelOutput" in data and "content" in data["modelOutput"]:
                content_list = data["modelOutput"]["content"]
                if content_list and len(content_list) > 0:
                    if "text" in content_list[0]:
                        content_list[0]["text"] = record_id
                    else:
                        # Add text field if it doesn't exist
                        content_list[0]["text"] = record_id
                else:
                    # Create content structure if empty
                    data["modelOutput"]["content"] = [{"text": record_id}]
            else:
                # Create the entire modelOutput structure if missing
                data["modelOutput"] = {"content": [{"text": record_id}]}
            
            # Write the modified entry
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            processed_count += 1
    
    print(f"Done! Processed {processed_count} entries, skipped {skipped_count} entries.")
    print(f"Output written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Replace content text with recordId in JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python replace_content_with_recordid.py input.jsonl output.jsonl
    
    python replace_content_with_recordid.py \\
        ./ID_generation/preprocessing/raw_data/Amazon/Beauty_date_context.jsonl \\
        ./ID_generation/preprocessing/raw_data/Amazon/Beauty_date_context_id_only.jsonl
        """
    )
    parser.add_argument(
        "input_jsonl",
        help="Path to the input JSONL file"
    )
    parser.add_argument(
        "output_jsonl", 
        help="Path to the output JSONL file"
    )
    
    args = parser.parse_args()
    
    replace_content_with_recordid(args.input_jsonl, args.output_jsonl)


if __name__ == "__main__":
    main()
