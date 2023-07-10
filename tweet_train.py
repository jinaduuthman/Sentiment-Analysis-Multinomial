import csv
import util
import pickle as pk
import numpy as np

# What is the probability for unseen words?
DEFAULT_P = 0.5

# Read the word list
## Your code here
file = open("vocablist_tweet.pkl", "rb")
vocab_list = pk.load(file)


# Note how many words are in the vocabulary
veclen = len(vocab_list)

# Make a dictionary (word->index) for faster lookup
# Convert dictionary for faster lookup
vocab_lookup = {}
for i, word in enumerate(vocab_list):
    vocab_lookup[word] = i

# Create an np array for counting each word
# sums[3][1] will be the total number of vocab_list[3]
# in sentiment 1 tweets
sums = np.zeros((veclen, 3), dtype=np.double)

# Create an array for counting tweets by sentiment
# tweet_counts[0] will be the total number of negative tweets
tweet_counts = np.zeros(3, dtype=int)

# Step through the train.csv file
with open("train_tweet.csv", "r", encoding="utf8") as f:
    reader = csv.reader(f)

    # Keep track of how many tweets we skipped
    # because they had no words in our vocabulary
    skipped_tweet_count = 0

    for row in reader:

        # Skip rows that don't have two entries
        if len(row) != 2:
            continue

        # Get the tweet and its associated sentiment
        tweet = row[0]
        sentiment = int(row[1])

        # Convert the tweet to a list of words
        wordlist = util.str_to_list(tweet)  ## Your code here (use util.py)

        # Convert the list of words into a word count vector
        word_counts = util.counts_for_wordlist(
            wordlist, vocab_lookup
        )  ## Your code here (use util.py)

        # Skip tweets with no common words
        if word_counts is None:
            skipped_tweet_count += 1
            continue

        # Add the word counts to the sums for the appropriate sentiment
        # (You don't need a loop here)
        ## Your code here
        if sentiment == 0:
            for j in range(veclen):
                sums[j][0] += word_counts[j]
        elif sentiment == 1:
            for j in range(veclen):
                sums[j][1] += word_counts[j]
        elif sentiment == 2:
            sums[j][2] += word_counts[j]

        # Increment the count of the sentiment
        ## Your code here
        if sentiment == 0:
            tweet_counts[0] += 1
        elif sentiment == 1:
            tweet_counts[1] += 1
        elif sentiment == 2:
            tweet_counts[2] += 1

print(f"Skipped {skipped_tweet_count} tweets: had no words from vocabulary")

# Zeros are draconian
# Replace any zeros in sums with DEFAULT_P
## Your code here
sums[sums == 0] = DEFAULT_P

# From sums, get the total number of counted
# words for each sentiment
totals = np.sum(sums, axis=0)  ## Your code here
assert totals.shape == (3,), "totals is an incorrect shape"

# Compute the word frequencies
# Create Arrays of zeros
word_vector_freq = np.zeros((veclen, 3), dtype=np.double)
for i in range(veclen):
    word_vector_freq[i][0] = sums[i][0] / totals[0]
    word_vector_freq[i][1] = sums[i][1] / totals[1]
    word_vector_freq[i][2] = sums[i][2] / totals[2]

word_frequencies = word_vector_freq  ## Your code here
sum_word_freqencies = np.sum(word_frequencies, axis=0)


assert np.all(
    np.isclose(word_frequencies.sum(axis=0), np.array([1.0, 1.0, 1.0]))
), "Word frequencies for a sentiment do not sum to one"

# Take the log of the word frequencies
log_word_frequencies = np.log(word_frequencies)  ## Your code here

# Compute the priors
sum_tweet_counts = np.sum(tweet_counts)
sentiment_frequencies = np.zeros(3, dtype=np.double)
for k in range(3):
    sentiment_frequencies[k] = tweet_counts[k] / sum_tweet_counts
# sentiment_frequencies = ## Your code here

assert sentiment_frequencies.shape == (
    3,
), "sentiment_frequencies is an incorrect shape"
assert np.isclose(
    sentiment_frequencies.sum(), 1.0
), "sentiment frequencies do not sum to one"

# Print out the priors
print("*** Tweets by sentiment ***")
for i in range(3):
    print(f"\t{i} ({util.labels[i]}): {sentiment_frequencies[i] * 100.0:.1f}%")

# Compute the log of the priors
log_sentiment_frequencies = np.log(sentiment_frequencies)  ## Your code here

assert log_word_frequencies.shape == (
    2000,
    3,
), "log_word_frequencies is an incorrect shape"
assert log_sentiment_frequencies.shape == (
    3,
), "log_sentiment_frequencies is an incorrect shape"

# Write out the logs of the word and sentiment frequencies in a single pickle file
## Your code here
logs_word_sentiment = (
    {}
)  # Creating a dictionary to hold the scaler and the model(logreg)
logs_word_sentiment["log_word_frequencies"] = log_word_frequencies
logs_word_sentiment["log_sentiment_frequencies"] = log_sentiment_frequencies
logs_word_sentiment["word_frequencies"] = word_frequencies
logs_word_sentiment["sentiment_frequencies"] = sentiment_frequencies

pk.dump(
    logs_word_sentiment, open("frequencies_tweet.pkl", "wb")
)  # Save the model to pickle file


# Just for fun, print out the most positive and most negative words
# by taking the difference between a wols
# rd's frequency in sentiment 0 tweets
# and its frequency in sentiment 2 tweets
happy_angry = word_frequencies[:, 0] - word_frequencies[:, 2]
happiest_to_angriest = np.argsort(happy_angry)

## Your code here
positive_words_list = happiest_to_angriest[:10]
negative_words_list = happiest_to_angriest[-10:]

print("Positive words:")
positive_words = [
    vocab_list[positive_words_list[w]] for w in range(len(positive_words_list))
]

for a in positive_words:
    print("\t", a, end="")

print("\nNegative words:")
negative_words = [
    vocab_list[negative_words_list[w]] for w in range(len(negative_words_list))
]

for a in negative_words:
    print("\t", a, end="")
