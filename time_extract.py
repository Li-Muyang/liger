import gzip
import json
from datetime import datetime

path = "/home/ec2-user/recsys/liger/ID_generation/preprocessing/raw_data/amazon/reviews_Beauty_5.json.gz"

dates = set()
with gzip.open(path, 'rt') as f:
    for line in f:
        r = json.loads(line)
        dates.add(datetime.fromtimestamp(r['unixReviewTime']).strftime('%Y-%m-%d'))

sorted_dates = sorted(dates)

output_path = "/home/ec2-user/recsys/liger/ID_generation/preprocessing/raw_data/amazon/reviews_Beauty_5_dates.json"
with open(output_path, 'w') as f:
    json.dump(sorted_dates, f, indent=2)

print(f"Saved to {output_path}")
print(f"\n=== Stats ===")
print(f"Unique dates: {len(sorted_dates)}")
print(f"Time span: {sorted_dates[0]} to {sorted_dates[-1]}")
