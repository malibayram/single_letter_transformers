"""Train a small BPE tokenizer and upload it to Hugging Face.

Install once:
    pip install transformers tokenizers huggingface_hub
    hf auth login

Then edit TEXT_FILE and REPO_ID below and run:
    python train_hf_tokenizer.py
"""

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

TEXT_FILE = "datasets/bpe-dataset-corpus.txt"
REPO_ID = "alibayram/my-tokenizer-demo"
VOCAB_SIZE = 2 ** 13

SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]


# 1. Create and train a byte-level BPE tokenizer.
backend = Tokenizer(BPE(unk_token="<unk>"))
backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
backend.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=ByteLevel.alphabet(),
)
backend.train([TEXT_FILE], trainer)

# 2. Wrap it so Transformers and AutoTokenizer understand it.
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=backend,
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<bos>",
    eos_token="<eos>",
)

# 3. Save locally and upload to the Hub.
tokenizer.save_pretrained("my-tokenizer")
tokenizer.push_to_hub(REPO_ID)

# 4. Load it exactly as other users will load it.
loaded = AutoTokenizer.from_pretrained(REPO_ID)
example = "Hello! This is a very small tokenizer example."

token_ids = loaded.encode(example)
print("Tokens:", loaded.convert_ids_to_tokens(token_ids))
print("Token IDs:", token_ids)
print("Decoded:", loaded.decode(token_ids))
