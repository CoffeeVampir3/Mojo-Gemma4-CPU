# checkpoint_downloader

Small `uv`-based downloader that pulls a Hugging Face model snapshot natively
(`huggingface_hub.snapshot_download`, accelerated with `hf_xet` high-performance
transfer) into `../checkpoints/<name>`, reproducing the repo's file layout.

## Usage

Default — download `google/gemma-4-26B-A4B` into `checkpoints/gemma-4-26B-A4B`:

```
cd checkpoint_downloader
uv run download.py
```

The result mirrors the source repo:

```
checkpoints/gemma-4-26B-A4B/
  .gitattributes
  config.json
  generation_config.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
  model.safetensors.index.json
  processor_config.json
  README.md
  tokenizer_config.json
  tokenizer.json
```

Re-running resumes partial downloads and skips files already present.

## Options

- `--repo` — Hugging Face repo id (default `google/gemma-4-26B-A4B`).
- `--name` — destination subdirectory under `checkpoints` (default: repo basename).
- `--checkpoints` — destination root (default `../checkpoints`).
- `--revision` — branch, tag, or commit sha.
- `--token` — Hugging Face access token (or set `HF_TOKEN`). Gemma is gated, so
  you must accept the license on the model page and supply a token.
- `--allow` / `--ignore` — glob patterns to restrict the download, e.g.
  `--ignore "*.safetensors"` for config/tokenizer only.

Example, weights-free config + tokenizer pull:

```
uv run download.py --ignore "*.safetensors"
```
