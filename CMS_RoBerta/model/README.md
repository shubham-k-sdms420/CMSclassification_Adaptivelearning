# Model Directory — Keep All Files

**Do not delete or remove any files from this directory.**

This folder contains the trained CMS RoBERTa classification model and its assets. All files here are required for the model to load and run.

## Required files (keep all)

| File | Purpose |
|------|---------|
| `config.json` | Model configuration (architecture, labels, etc.) |
| `tokenizer_config.json` | Tokenizer configuration |
| `tokenizer.json` | Tokenizer vocabulary and settings |
| `label2id.json` | Category name → label ID mapping |
| `model.safetensors` or `pytorch_model.bin` | Model weights (if present) |

Plus any other files you have placed here (e.g. `special_tokens_map.json`).

## Notes

- The integration and API load the model from this path. Removing or renaming files here will break the service.
- If you add or replace model weights, keep the filenames expected by the loader (`model.safetensors` or `pytorch_model.bin`).

**Keep all files in this directory.**
