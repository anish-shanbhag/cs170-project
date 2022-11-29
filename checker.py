## checks the scores of our results compared to webscraped results
from helper import *

import json
with open("scores.json") as f:
    scores = json.load(f)
    
short, medium, long = [], [], []
for name in range(300):
    short_name = "small" + str(name)
    if os.path.isfile(f"outputs/{short_name}.out"):
        our_score = check_score(short_name)
        best_score = scores[short_name]
        if ((our_score - best_score) / best_score) > 0.01:
            short.append((our_score - best_score) / best_score)
    medium_name = "medium" + str(name)
    if os.path.isfile(f"outputs/{medium_name}.out"):
        our_score = check_score(medium_name)
        best_score = scores[medium_name]
        medium.append((our_score - best_score) / best_score)
    long_name = "large" + str(name)
    if os.path.isfile(f"outputs/{long_name}.out"):
        our_score = check_score(long_name)
        best_score = scores[long_name]
        long.append((our_score - best_score) / best_score)
        