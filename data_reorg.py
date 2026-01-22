import gzip
import json
from datetime import datetime
from collections import defaultdict

path = "/home/ec2-user/recsys/liger/ID_generation/preprocessing/raw_data/amazon/reviews_Toys_and_Games_5.json.gz"

reviews_by_user = defaultdict(list)

with gzip.open(path, 'rt') as f:
    for line in f:
        r = json.loads(line)
        reviews_by_user[r['reviewerID']].append({
            'reviewText': r.get('reviewText', ''),
            'summary': r.get('summary', ''),
            'overall': r['overall'],
            'reviewTime': datetime.fromtimestamp(r['unixReviewTime']).strftime('%Y-%m-%d %H:%M:%S'),
            '_sort_key': r['unixReviewTime']
        })

for uid in reviews_by_user:
    reviews_by_user[uid].sort(key=lambda x: x['_sort_key'])
    for r in reviews_by_user[uid]:
        del r['_sort_key']

output_path = "/home/ec2-user/recsys/liger/ID_generation/preprocessing/raw_data/amazon/reviews_Toys_and_Games_5_organized.json"
with open(output_path, 'w') as f:
    json.dump(dict(reviews_by_user), f, indent=2)

print(f"Saved to {output_path}")

# Stats
num_reviewers = len(reviews_by_user)
review_counts = [len(v) for v in reviews_by_user.values()]
num_reviews = sum(review_counts)

print(f"\n=== Stats ===")
print(f"Reviewers: {num_reviewers}")
print(f"Reviews: {num_reviews}")
print(f"Reviews per reviewer: min={min(review_counts)}, max={max(review_counts)}, avg={num_reviews/num_reviewers:.2f}")
