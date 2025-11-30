---
sidebar_position: 1
---

# CLI

Command-line interface for Benchwise.

## Installation

How to install the Benchwise CLI.

The CLI is included when you install Benchwise:

```bash
pip install benchwise
```

## Available Commands

A list of commands available through the Benchwise CLI.

### List Models

```bash
benchwise list models
```

### List Metrics

```bash
benchwise list metrics
```

### Run Evaluation

```bash
benchwise eval gpt-4 claude-3-opus --dataset data.json --metrics accuracy rouge_l
```

### Validate Dataset

```bash
benchwise validate dataset.json
```

### Compare Results

```bash
benchwise compare results1.json results2.json --metric accuracy
```

## Examples

Practical examples of using Benchwise CLI commands.

```bash
# Run QA evaluation
benchwise eval gpt-4 --dataset qa.json --metrics accuracy

# Compare multiple result files
benchwise compare run1.json run2.json run3.json

# Validate dataset format
benchwise validate my_dataset.json
```

## See Also

- [Configuration](../advanced/configuration.md)
- [Getting Started](../getting-started/installation.md)
