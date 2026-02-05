#!/usr/bin/env python3
"""
Script to analyze timestamp coverage in the Beauty dataset and show quarterly counts.

Usage:
    python analyze_timestamp_coverage.py
"""

import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime


def parse_amazon_reviews(filepath):
    """Parse Amazon reviews JSON gzip file."""
    timestamps = []
    g = gzip.open(filepath, "r")
    for line in g:
        line = line.replace(b"true", b"True").replace(b"false", b"False")
        try:
            entry = eval(line)
            if "unixReviewTime" in entry:
                timestamps.append(int(entry["unixReviewTime"]))
        except:
            continue
    g.close()
    return timestamps


def timestamp_to_quarter(timestamp):
    """Convert Unix timestamp to quarter string (e.g., '2014-Q1')."""
    dt = datetime.utcfromtimestamp(timestamp)
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{quarter}"


def analyze_quarterly_coverage(timestamps):
    """Analyze timestamp coverage by quarter."""
    quarterly_counts = Counter()
    
    for ts in timestamps:
        quarter = timestamp_to_quarter(ts)
        quarterly_counts[quarter] += 1
    
    return quarterly_counts


def display_results(quarterly_counts, dataset_name):
    """Display quarterly coverage results."""
    print(f"\n{'='*60}")
    print(f"Timestamp Coverage Analysis for {dataset_name}")
    print(f"{'='*60}\n")
    
    if not quarterly_counts:
        print("No timestamp data found.")
        return
    
    # Sort by quarter
    sorted_quarters = sorted(quarterly_counts.items())
    
    # Calculate stats
    total_interactions = sum(quarterly_counts.values())
    min_year = int(sorted_quarters[0][0].split('-')[0])
    max_year = int(sorted_quarters[-1][0].split('-')[0])
    
    print(f"Total Interactions: {total_interactions:,}")
    print(f"Time Range: {sorted_quarters[0][0]} to {sorted_quarters[-1][0]}")
    print(f"Number of Quarters: {len(quarterly_counts)}")
    print(f"\n{'Quarter':<15} {'Count':>10} {'Percentage':>12}")
    print('-' * 40)
    
    for quarter, count in sorted_quarters:
        percentage = (count / total_interactions) * 100
        print(f"{quarter:<15} {count:>10,} {percentage:>11.2f}%")
    
    # Show yearly aggregates
    print(f"\n{'='*60}")
    print("Yearly Aggregates")
    print(f"{'='*60}\n")
    
    yearly_counts = defaultdict(int)
    for quarter, count in quarterly_counts.items():
        year = quarter.split('-')[0]
        yearly_counts[year] += count
    
    print(f"{'Year':<10} {'Count':>10} {'Percentage':>12}")
    print('-' * 35)
    
    for year in sorted(yearly_counts.keys()):
        count = yearly_counts[year]
        percentage = (count / total_interactions) * 100
        print(f"{year:<10} {count:>10,} {percentage:>11.2f}%")
    
    print(f"\n{'='*60}\n")


def main():
    # Path to Beauty dataset
    raw_data_path = "/Users/limuyang/recsys/liger/ID_generation/preprocessing/raw_data/amazon"
    review_file = os.path.join(raw_data_path, "reviews_Beauty_5.json.gz")
    
    if not os.path.exists(review_file):
        print(f"Error: Review file not found at {review_file}", file=sys.stderr)
        print("Please ensure the Beauty dataset is downloaded.", file=sys.stderr)
        sys.exit(1)
    
    print("Loading timestamp data from Beauty dataset...")
    timestamps = parse_amazon_reviews(review_file)
    
    if not timestamps:
        print("Error: No timestamps found in the dataset.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(timestamps):,} interactions with timestamps.")
    
    print("Analyzing quarterly coverage...")
    quarterly_counts = analyze_quarterly_coverage(timestamps)
    
    display_results(quarterly_counts, "Amazon Beauty")
    
    # Optionally save to JSON
    output_file = "/Users/limuyang/recsys/liger/beauty_quarterly_coverage.json"
    with open(output_file, 'w') as f:
        json.dump(dict(sorted(quarterly_counts.items())), f, indent=2)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
