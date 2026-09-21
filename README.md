# Lidstone Unigram Language Model

A CS MSc Applied Probabilistic Models exercise from 2024. The project trains unigram language models on a Reuters-derived corpus, applies Lidstone smoothing, selects the smoothing factor with the lowest validation perplexity, and evaluates the selected model on a held-out test set.

## Tech Stack

- Python 3.10 or newer
- Python standard library for corpus parsing, probability estimation, and model selection
- pytest and Ruff for validation and code quality
- uv for reproducible environment and package management

## Setup

```bash
git clone https://github.com/KobieHazon/msc-applied-probabilistic-models-lidstone.git
cd msc-applied-probabilistic-models-lidstone
uv sync --dev
```

## Usage

Run the experiment with the supplied development and test data:

```bash
uv run apm-lidstone data/develop.txt data/test.txt honduras output.txt
```

The output contains the requested event counts, word probabilities, validation perplexities, selected smoothing factor, and final test perplexity. A reference run is preserved in `results/output.txt`.

## Testing

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The test suite includes focused model and corpus-parser tests plus a regression run over the complete supplied dataset.

## Repository Structure

- `assignment/`: supplied exercise brief and an output-format template
- `data/`: supplied development and test corpora
- `solution/`: my Python implementation
- `results/`: output reproduced by the solution
- `tests/`: focused tests and the full-data regression check
