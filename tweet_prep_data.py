# 57 lines of code
import csv
import util
import pickle as pk
import numpy as np

# column indices
SENTIMENT_COL = 1
CONFIDENCE_COL = 2
TWEET_COL = 10

# Probabilities
TEST_P = 0.1

# Confidence required
REQUIRED_CONFIDENCE = 0.5

# Vector length
VECLEN = 2000

# The file we are reading the tweets from
INPATH = "Tweets.csv"

# Create an output file for each phase
phases = ["train", "test"]
writers = {}
for phase in phases:
    f = open(f"{phase}_tweet.csv", "w", encoding="utf8", newline="\n")
    writer = csv.writer(f)
    writers[phase] = writer

# Read all the tweets
with open(INPATH, "r", encoding="utf8") as f:
    reader = csv.reader(f)

    # Skip header
    next(reader)

    # Count the words as we go
    count_dict = {}

    # Count how many are discarded for low confidence
    discarded_count = 0
    saved_count = 0
    for row in reader:

        # Make sure the row has at least 11 columns
        if len(row) < 11:
            continue

        # Skip rows with low confidence
        if float(row[CONFIDENCE_COL]) < REQUIRED_CONFIDENCE:
            discarded_count += 1
            continue

        # Get tweet
        tweet = row[TWEET_COL]

        # Convert to a list of lowercase words
        wordlist = util.str_to_list(tweet)

        # Which file should it go into?
        r = np.random.rand()
        if r < TEST_P:
            destination = "test"
        else:
            destination = "train"

        # Get sentiment
        sentiment = util.labels.index(row[SENTIMENT_COL])

        # Write it out
        writers[destination].writerow([tweet, sentiment])
        saved_count += 1

        # Count the words in the list
        for w in wordlist:
            if w not in count_dict:
                count_dict[w] = 1
            else:
                count_dict[w] += 1

print(f"Kept {saved_count} rows, discarded {discarded_count} rows")

# Make a list of words in descending order of frequency
word_pairs = list(count_dict.items())
word_pairs.sort(key=lambda x: x[1], reverse=True)
vocab_list = [word_pairs[i][0] for i in range(VECLEN)]
print(f"Most common 32 words are {vocab_list[:32]}")

# Save out the vocabulary
print(f"Wrote {len(vocab_list)} words to {util.vocab_list_path}.")
with open(util.vocab_list_path, "wb") as f:
    pk.dump(vocab_list, f)
