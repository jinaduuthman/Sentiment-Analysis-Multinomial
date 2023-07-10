# 40 lines of code
from nltk.corpus import stopwords
import numpy as np
import pickle as pk
import sys
import string


# Shared constants
vocab_list_path = "vocablist_tweet.pkl"
frequencies_path = "frequencies_tweet.pkl"
labels = ["negative", "neutral", "positive"]

# Initialization
ignore = [
    "united",
    "usairways",
    "americanair",
    "southwestair",
    "jetblue",
    "virginamerica",
]
# Put the stop words in a set
stop_words = set(stopwords.words("english"))

# Add the airline names
stop_words.update(ignore)

# Create a translator that deletes the punctuation and digits
translator = str.maketrans("", "", string.punctuation + string.digits)


def str_to_list(instring: str) -> list[str]:
    global stop_words

    # Get rid of leading and trailing whitespace
    tweet = instring.strip()

    # Get rid of punctuation and digits
    tweet = tweet.translate(translator)

    # Make lowercase and split into words
    tweetwords = tweet.lower().split()

    # Put non-stop words in a list
    wordlist = []
    for w in tweetwords:
        if w not in stop_words:
            wordlist.append(w)

    # Return the list
    return wordlist


def counts_for_wordlist(wordlist: list[str], vocab_lookup: dict[str, int]) -> np.array:

    # Create an empty array as long as the vocabulary
    count_vec = np.zeros(len(vocab_lookup))

    # Step through each word in the list
    # and look up its index in the vocabulary
    is_zero = True
    for word in wordlist:
        index = vocab_lookup.get(word)
        if index is None:
            continue
        count_vec[index] += 1
        is_zero = False

    # Did this tweet have no words in the vocabulary?
    if is_zero:
        # Let the caller know that the vector would have been zero
        return None
    else:
        # Return the counts
        return count_vec
