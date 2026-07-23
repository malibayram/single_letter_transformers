# single_letter_transformers

Tiny, **from-scratch** implementations of modern LLM architectures, each small enough
to read top to bottom and train on a laptop CPU in under a minute. Every model learns
the same toy task — **generate Turkish first names one letter at a time** — so you can
compare architectures directly instead of getting lost in scale.

Four language-model architectures, same task, side by side — plus a fifth
folder that is a different beast entirely (a four-model audio pipeline):

| Folder                     | Architecture           | What makes it different                                                                               |
| -------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------- |
| [`qwen3/`](qwen3/)         | **Qwen3 dense**        | The clean modern baseline: RMSNorm, QK-Norm, GQA, RoPE, SwiGLU, pre-norm                              |
| [`qwen3_5/`](qwen3_5/)     | **Qwen3.5 hybrid**     | Adds **Gated DeltaNet** linear-attention layers, interleaved with full attention                      |
| [`gemma4/`](gemma4/)       | **Gemma**              | **Sandwich norm**, local/global **sliding-window** attention, dual RoPE, **GeGLU**, embedding scaling |
| [`deepseek3/`](deepseek3/) | **DeepSeek-V3 sparse** | **MLA** compressed-KV attention + **MoE** with a shared expert and **aux-loss-free** load balancing   |
| [`acestep/`](acestep/)     | **ACE-Step v1.5**      | Not an LM: a **planner LM + FSQ bridge + diffusion DiT + VAE** that turn a letter into a **waveform** |

> The goal is _clarity_, not fidelity. Each folder implements the architecture's
> signature ideas in the simplest correct form; large-scale tricks (MatFormer,
> chunked-parallel DeltaNet, 256-expert routing, 256k vocab, multi-token
> prediction, etc.) are intentionally left out. See each folder's notes for what
> is simplified.

## Bonus: fine-tuning techniques

[`lora/`](lora/) is a **teaching module, not another architecture.** It freezes any of
the four models above and shows how **LoRA, rsLoRA, DoRA, VeRA, and PiSSA** re-steer it
by training a tiny add-on (as few as **704 numbers** — smaller than its own README).
The adapter code is **architecture-agnostic**: the same `inject()` wraps the
`nn.Linear`s of qwen3 (GQA), gemma (sliding-window), or deepseek (MLA) unchanged —
only the target layer names differ, and those live in one small registry. Every file
is commented line by line, and [`lora/by_hand.py`](lora/by_hand.py) verifies each
concept on pencil-friendly numbers your students can check by hand. See
[`lora/README.md`](lora/README.md).

## Bonus: quantization from numbers to packed weights

[`quantization/`](quantization/) is a second **teaching and practice module**. It
starts with `scale`, `zero_point`, rounding and clipping, then implements real
INT8 storage and signed/unsigned **packed INT4** (two values per byte), a
float-weight-free `QuantizedLinear`, PTQ/QAT with an STE, NF4, and the 144-byte
Q4_K super-block structure. It now also includes a structurally faithful
**Q8_0** implementation: 32 signed INT8 codes plus one FP16 scale (34 bytes,
8.5 bits/weight), a float-weight-free `Q8_0Linear`, hand calculations, and a
dedicated TinyQwen evaluation. Its demos compare FP32/Q8_0/INT4 and isolate the
central ideas behind GPTQ, AWQ and SmoothQuant. Every numerical claim is
asserted in runnable CPU examples; the documentation also explains why
readable PyTorch storage code is not an optimized low-bit inference kernel. See
[`quantization/README.md`](quantization/README.md).

## Design rules

- **One module = one file.** `config → rms_norm → rotary → attention → mlp → block → model`.
- **Each folder is self-contained** — copies of the shared simple modules live in each, so
  you can study one architecture without jumping between directories.
- **No `transformers` dependency.** Just `torch`.
- Character-level: every token is a single Turkish letter; `\n` marks the start/end of a name.

## Data

- [`data/isimler.txt`](data/isimler.txt) — raw names (UPPERCASE, some multi-name lines)
- [`data/temizle_isimler.py`](data/temizle_isimler.py) — lowercases (correct Turkish `I/İ`),
  splits multi-name lines, dedupes
- [`data/temiz_isimler.txt`](data/temiz_isimler.txt) — **921 cleaned names**, one per line
  (29 Turkish letters + newline)

Regenerate the clean file: `cd data && python3 temizle_isimler.py`

## Run it

Each folder works the same way. From inside a model folder:

```bash
cd qwen3        # or qwen3_5, gemma4, or deepseek3
python3 train.py        # trains a few thousand steps, prints loss, samples names, saves a checkpoint
python3 generate.py 20      # generate 20 names
python3 generate.py 20 0.7  # lower temperature = safer / more common names
```

> [`acestep/`](acestep/) is the exception: same `train.py` entry point (it trains four
> stages in series, ~80s on CPU), but `python3 generate.py a` takes a **letter** and writes
> `out.wav` instead of printing names. See its own README for the four-region walkthrough.

> This project uses **pyenv**, pinned to **Python 3.13.3** via [`.python-version`](.python-version)
> (run `pyenv install 3.13.3` once if needed). The old `.venv/` is not used.

## Use in a Jupyter notebook

Start the notebook from the repo root (`jupyter lab` or `jupyter notebook`). Each model
folder uses flat imports (`from model import ...`), so add the folder you want to
`sys.path` first.

> **One architecture per kernel.** All model folders share module names
> (`model.py`, `config.py`, …). To switch architectures, **restart the kernel** so the
> right modules get imported.

**Train from a cell** (simplest — just run the script):

```python
!cd qwen3 && python train.py      # or qwen3_5, gemma4, or deepseek3
```

**Or train inline** (handy for tweaking hyperparameters live):

```python
import sys, torch
sys.path.insert(0, "qwen3")                      # pick ONE: qwen3 / qwen3_5 / gemma4 / deepseek3
from config import ModelConfig
from model import TinyQwen                         # TinyQwen35 / TinyGemma / TinyDeepSeek in the others
from tokenizer import CharTokenizer

tok = CharTokenizer.from_file("data/temiz_isimler.txt")
data = torch.tensor(tok.encode(open("data/temiz_isimler.txt", encoding="utf-8").read()))

cfg = ModelConfig(vocab_size=tok.vocab_size)
model = TinyQwen(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

block = 16
for step in range(1, 2001):
    ix = torch.randint(len(data) - block - 1, (64,))
    x = torch.stack([data[i:i+block] for i in ix])
    y = torch.stack([data[i+1:i+1+block] for i in ix])
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0:
        print(step, round(loss.item(), 3))

torch.save({"model": model.state_dict(), "chars": tok.chars, "cfg": cfg}, "qwen3/tiny_qwen.pt")
```

**Load a trained checkpoint and generate names:**

```python
import sys, torch
sys.path.insert(0, "qwen3")                      # match the architecture of the checkpoint
from model import TinyQwen
from tokenizer import CharTokenizer

ckpt = torch.load("qwen3/tiny_qwen.pt", map_location="cpu", weights_only=False)
tok = CharTokenizer(ckpt["chars"])
model = TinyQwen(ckpt["cfg"]); model.load_state_dict(ckpt["model"]); model.eval()

start = torch.full((10, 1), tok.eos_id, dtype=torch.long)   # every name starts at EOS/newline
out = model.generate(start, max_new_tokens=model.cfg.max_seq_len,
                     temperature=0.8, eos_id=tok.eos_id)     # stops each row at EOS
for row in out.tolist():
    print(tok.decode(row[1:]).split("\n")[0])
```

Checkpoint paths and model classes per folder:

| Folder      | Model class    | Checkpoint                   |
| ----------- | -------------- | ---------------------------- |
| `qwen3`     | `TinyQwen`     | `qwen3/tiny_qwen.pt`         |
| `qwen3_5`   | `TinyQwen35`   | `qwen3_5/tiny_qwen35.pt`     |
| `gemma4`    | `TinyGemma`    | `gemma4/tiny_gemma.pt`       |
| `deepseek3` | `TinyDeepSeek` | `deepseek3/tiny_deepseek.pt` |

## Results (toy run, CPU)

All four drop from the uniform baseline loss (`ln 30 ≈ 3.40`) to ~0.5 and produce
plausible Turkish names:

| Model       | Params             | Final loss | Sample names                  |
| ----------- | ------------------ | ---------- | ----------------------------- |
| Qwen3       | ~20k               | ~0.45      | nurhan, oktay, nalan, bedriye |
| Qwen3.5     | ~42k               | ~0.54      | nevin, şebnem, orhan, cemal   |
| Gemma       | ~63k               | ~0.56      | selda, zeliha, erkan, rabia   |
| DeepSeek-V3 | ~48k (~36k active) | ~0.58      | zafer, habibe, ferman, cömert |

## The architectures in one paragraph each

**Qwen3 (dense).** A standard pre-norm decoder. Each block does
`x += attention(norm(x)); x += mlp(norm(x))`. Attention uses Grouped-Query Attention
(fewer key/value heads than query heads), applies RMSNorm to per-head queries and keys
(**QK-Norm**) and then **RoPE**, and the MLP is **SwiGLU**. See [`qwen3/`](qwen3/).

**Qwen3.5 (hybrid).** Same skeleton, but most layers replace softmax attention with
**Gated DeltaNet** — a linear-attention layer that keeps one fixed-size memory matrix and
updates it token by token with the _gated delta rule_ (decay gate + write gate + error
correction). Only every 4th layer keeps full attention. See
[`qwen3_5/gated_deltanet.py`](qwen3_5/gated_deltanet.py).

**Gemma.** Normalizes **before and after** each sub-layer ("sandwich norm"), interleaves
**5 local sliding-window layers : 1 global layer**, uses a small RoPE base for local layers
and a large one for global, swaps SwiGLU for **GeGLU**, scales embeddings by
`sqrt(hidden_size)`, and adds small **per-layer embeddings**. See [`gemma4/`](gemma4/).

**DeepSeek-V3 (sparse).** Swaps both halves of the block. Attention becomes
**MLA**: each token is compressed into a tiny **KV latent** and every head's K/V
are re-expanded from it (only the latent would ever be cached), with position
carried by a few **decoupled RoPE** dims. The MLP becomes **MoE**: a sigmoid
router sends each token to its **top-2** of 4 small experts (plus one always-on
shared expert), and load is balanced **aux-loss-free** by nudging a per-expert
selection bias after every step. See [`deepseek3/`](deepseek3/).

**ACE-Step v1.5 (the outlier).** Not a single LM and not the names task — it is a
**two-brain audio pipeline**, so here each Turkish letter is a _tag_ whose "song" is a tiny
waveform. A low-resolution autoregressive **planner LM** (the reused TinyQwen) writes a coarse
**5Hz code** blueprint; an **FSQ** bridge turns those discrete codes into a continuous "source
latent"; a **diffusion DiT** denoises noise → **25Hz latent** in a few **flow-matching** steps
(self-attention for time coherence + cross-attention onto the tag and skeleton); and a small
**Oobleck VAE** decodes the latent to a **waveform**. Run `python3 generate.py a` (not a
count) — it traces the shape through all four regions and writes `out.wav`. See
[`acestep/`](acestep/).
