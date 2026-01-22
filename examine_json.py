import gzip
import json

path = "/home/ec2-user/recsys/liger/ID_generation/preprocessing/raw_data/steam/steam_reviews.json.gz"

with gzip.open(path, 'rt') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(json.loads(line))