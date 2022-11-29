## checks the scores of our results compared to webscraped results
from helper import *

import json
with open("scores.json") as f:
    scores = json.load(f)
    
short, medium, long = [], [], []
for name in range(300):
    short_name = "small" + str(name)
    if os.path.isfile(f"{short_name}.out"):
        old_score = score(read_output(read_input(f"inputs/{short_name}.in"), f"outputs/{short_name}.out"))
        c_score = score(read_output(read_input(f"inputs/{short_name}.in"), f"{short_name}.out"))
        if c_score < old_score:
            with open(f"{short_name}.out") as f:
                with open(f"outputs/{short_name}.out", "w") as f2:
                    f2.write(f.read())
        best_score = scores[short_name]
        if ((old_score - best_score) / best_score) > 0.01:
            short.append(name)
        
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
        
        
def checkandreplace(short_name):
    stored = score(read_output(read_input(f"inputs/{short_name}.in"), f"outputs/{short_name}.out"))
    ## for all files that start with "short_name" in the current directory, check if the score is better than the stored one
    for file in os.listdir():
        if file.startswith(short_name):
            our_score = score(read_output(read_input(f"inputs/{short_name}.in"), file))
            if our_score < stored:
                print("TRUE", our_score)
                with open(file) as f:
                    with open(f"outputs/{short_name}.out", "w") as f2:
                        f2.write(f.read())
                stored = our_score
                
    ## delete all files that start with "short_name" in the current directory
    for file in os.listdir():
        if file.startswith(short_name):
            os.remove(file)
        