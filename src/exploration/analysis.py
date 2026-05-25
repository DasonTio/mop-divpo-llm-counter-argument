# %% Cell 1: Setup
import os

from convokit import Corpus, download

# %% Cell 2: Load Corpus
# ConvoKit downloads to ~/.convokit/saved-corpora by default
corpus_name = "conversations-gone-awry-cmv-corpus"
try:
    # This will load it from the local path since it's already downloaded
    corpus = Corpus(filename=download(corpus_name))
    print(f"Successfully loaded {corpus_name}")
except Exception as e:
    print(f"Error loading corpus: {e}")

# %% Cell 3: Basic Exploration
print(f"Number of Speakers: {len(corpus.get_speaker_ids())}")
print(f"Number of Utterances: {len(corpus.get_utterance_ids())}")
print(f"Number of Conversations: {len(corpus.get_conversation_ids())}")

# %% Cell 4: Preview first conversation
first_conv_id = corpus.get_conversation_ids()[0]
first_conv = corpus.get_conversation(first_conv_id)
print(f"\nExample Conversation ID: {first_conv_id}")
print(
    f"Number of utterances in this conversation: {len(first_conv.get_utterance_ids())}"
)
