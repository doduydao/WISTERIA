# WISTERIA

WISTERIA is a codebase and toolkit for temporal relation extraction (predicting temporal relations between two events or event/time expressions) on datasets such as I2B2, TBD, TDD, and MATRES. The repository contains preprocessing utilities, dataset classes with careful token alignment between spaCy and HuggingFace tokenizers, a BERT-based model with cross-attention and biaffine scoring, and training/evaluation utilities including top-k attention analysis.

## Key components

- `preprocess/` — scripts to read and parse TimeML/TimeBank/TBD/TDD files and convert them into tabular formats (CSV/DataFrame).
  - `TBPreprocess/TimeMLParser.py`: TimeML parser (TimeBank/TimeBank-Dense) and functions to clean text and update character spans.
  - `readTBD.py`, `readTDD.py`, etc.: dataset-specific loaders that produce DataFrames suitable for training.

- `source/` — main implementation
  - `model.py`: `TemporalRelationModelV2` — a BERT encoder plus positional/context encoder, cross-attention modules, biaffine scorer, and fusion heads. Supports returning auxiliary logits and top-k attention tokens for analysis.
  - `data.py`: dataset classes (`CustomTextDatasetTDD`, `CustomTextDatasetI2B2`), token alignment utilities (BERT ↔ spaCy), spaCy masks and entity mask builders, and `create_dataloader`.
  - `analysis_topk.py`: training loop (`train_model`), evaluation utilities (`evaluate`, `grid_search`) and dataset-specific evaluation logic (I2B2 closure-based evaluation, TDD/TBD/MATRES metrics).
  - `utilsv2.py`: helper functions for building analysis datasets, token normalization, and visualization helpers (POS/dep/morph distributions).

- `top-k/` — example outputs from top-k attention analysis (e.g. `MATRES_D4A.json`, `TBD_D4A.json`).

- Notebooks in `source/` — example Jupyter notebooks for experiments and analysis (e.g. `script_TBD_BERT.ipynb`, `script_MATRES_BERT.ipynb`).

## Requirements (recommended)

- Python 3.8+ (3.9/3.10 recommended)
- PyTorch (the repository was tested with PyTorch 1.4 in the provided environment)
- transformers (HuggingFace)
- spaCy (and an English model such as `en_core_web_sm` or `en_core_web_trf`)
- scikit-learn, pandas, tqdm, matplotlib, seaborn

A minimal `requirements.txt` has been added to the repository; install dependencies in a virtual environment and download the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Quick start

1. Prepare the raw TimeML data and run the dataset loader to generate CSV/DFs:

```bash
python preprocess/readTBD.py
# or: python preprocess/readTDD.py
```

2. Open one of the notebooks in `source/` to run experiments interactively, or use the provided functions from Python scripts (see examples below).

3. Minimal programmatic example to instantiate the model:

```python
from transformers import BertTokenizerFast, BertModel
from source.model import TemporalRelationModelV2

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
model = TemporalRelationModelV2(bert, tokenizer, hidden_ffn_dim=512)
```

To build DataLoaders that use spaCy ↔ BERT alignment, see `source/data.py` and call `create_dataloader(...)` with an appropriate `spacy_nlp`, `tokenizer`, and parameters such as `MAX_LENGTH`, `window_size`, and `batch_size`.

## Training and evaluation

- Training is implemented in `source/analysis_topk.py` via `train_model(...)`. Main arguments include: `top_k`, `train_dataloader`, `validation_dataloader`, `optimizer`, `criterion`, `device`, and `num_epochs`.
- Evaluation is handled by `evaluate(...)` which supports dataset-specific metrics (I2B2 closure-based evaluation when `test_df` is provided, and dataset-appropriate metrics for TDD/TBD/MATRES).
- The model supports returning top-k attention tokens (`return_topk_tokens=True`) for detailed analysis and visualization.

## Notebooks and analysis

Use the notebooks in `source/` for interactive experiments, model debugging, and plotting attention/top-k analyses. The `utilsv2.py` helpers convert model logs into analysis DataFrames and plotting functions.

## Data

The repository expects TimeML-format inputs for datasets such as TimeBank, TimeBank-Dense, TBD, TDD, and MATRES. Preprocessing scripts in `preprocess/` convert `.tml` files to DataFrames with columns such as entity spans, text, and labels. Adjust `data_root` variables in the loader scripts as needed to match your local dataset paths.

## Technical notes

- Token alignment: the code aligns HuggingFace tokenizer subword tokens with spaCy tokens using `spacy` Alignment utilities; this allows building word-level masks and pooling subword embeddings up to word tokens.
- Model architecture: combination of entity pooling, word-level context encoder, cross-attention from entities/pair to word-context, top-k pooling, and a late fusion of biaffine/context/fusion heads.
- Training: the training loop supports masking out `VAGUE` labels (if present) and weighting auxiliary losses from auxiliary heads.

## Possible improvements / next steps

- Add a command-line training script (`train.py`) and a configuration file for reproducible experiments.
- Provide a pinned `requirements.txt` (or `environment.yml`) matching specific CUDA/PyTorch combos for reproducible environments.
- Add unit tests and a small end-to-end smoke test to verify data loading → model → training loop.

## Contact

If you want me to generate a `train.py`, a `environment.yml` that reproduces your conda env, or pin `transformers`/`spaCy` versions compatible with PyTorch 1.4, tell me which option you prefer and I will add it.

---
This README was generated based on a quick analysis of the codebase. If you want a more detailed step-by-step example for a specific dataset and setup (for example: train on TBD with BERT-base, top_k=5, batch_size=8), tell me and I'll add a runnable script and commands.
