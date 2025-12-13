"""
Tests for documentation examples.

This module extracts and tests Python code examples from documentation markdown files
to ensure they are syntactically correct and execute without errors.
"""

import ast
import re
import pytest
from pathlib import Path
from typing import List


def extract_code_blocks_from_md(markdown_file: Path) -> List[tuple]:
    """
    Extract all Python code blocks from a markdown file.
    Returns list of (code, block_number, line_number) tuples.
    """
    with open(markdown_file, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r"```python\n(.*?)```"
    matches = re.finditer(pattern, content, re.DOTALL)

    code_blocks = []
    for i, match in enumerate(matches, 1):
        code = match.group(1)
        line_number = content[: match.start()].count("\n") + 1
        code_blocks.append((code, i, line_number))

    return code_blocks


def get_doc_files() -> List[Path]:
    """Get all markdown documentation files with code examples."""
    docs_dir = Path(__file__).parent.parent / "docs" / "docs" / "examples"

    if not docs_dir.exists():
        return []

    return sorted(docs_dir.glob("*.md"))


def prepare_code_for_testing(code: str) -> str:
    """
    Prepare documentation code for testing by replacing real models with mocks.
    """
    model_replacements = {
        '"gpt-4"': '"mock-gpt-4"',
        '"gpt-3.5-turbo"': '"mock-gpt-3.5"',
        '"gpt-4o-mini"': '"mock-gpt-4o-mini"',
        '"claude-3-opus"': '"mock-claude-opus"',
        '"claude-3-sonnet"': '"mock-claude-sonnet"',
        '"claude-3-5-sonnet-20241022"': '"mock-claude-sonnet"',
        '"claude-3-haiku"': '"mock-claude-haiku"',
        '"claude-3-5-haiku-20241022"': '"mock-claude-haiku"',
        '"claude-opus-4-1"': '"mock-claude-opus"',
        '"gemini-pro"': '"mock-gemini-pro"',
    }

    modified_code = code
    for real, mock in model_replacements.items():
        modified_code = modified_code.replace(real, mock)

    # Replace placeholder dataset loading
    if 'load_dataset("data/' in modified_code:
        modified_code = modified_code.replace(
            'load_dataset("data/qa_1000.json")',
            'create_qa_dataset(questions=["Q1?"], answers=["A1"], name="test")',
        )
        modified_code = modified_code.replace(
            'load_dataset("data/news_articles.json")',
            'create_summarization_dataset(documents=["Doc1"], summaries=["Sum1"], name="news")',
        )

    return modified_code


# Generate test parameters from all documentation files
doc_files = get_doc_files()
test_params = []

for doc_file in doc_files:
    blocks = extract_code_blocks_from_md(doc_file)
    for code, block_num, line_num in blocks:
        test_params.append((doc_file.name, block_num, line_num, code))


@pytest.mark.parametrize(
    "filename,block_num,line_num,code",
    test_params,
    ids=[f"{f}:block_{b}:L{line}" for f, b, line, _ in test_params],
)
def test_documentation_code_syntax(filename, block_num, line_num, code):
    """
    Test that all code examples in documentation have valid Python syntax.

    This is a basic sanity check - if this fails, the example cannot be run.
    """
    try:
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(
            f"Syntax error in {filename} block {block_num} (line {line_num}):\n"
            f"  Line {e.lineno}: {e.msg}\n"
            f"  {e.text}"
        )


@pytest.mark.slow
@pytest.mark.mock
@pytest.mark.parametrize(
    "filename,block_num,line_num,code",
    test_params,
    ids=[f"{f}:block_{b}:L{line}" for f, b, line, _ in test_params],
)
def test_documentation_code_execution(filename, block_num, line_num, code):
    """
    Test that code examples can be executed without errors (using mock models).

    Note: Some examples are intentionally incomplete (just function definitions)
    and will be skipped.
    """
    # Skip examples that are just function definitions without execution
    if "@evaluate(" in code and "asyncio.run" not in code:
        pytest.skip("Incomplete example (defines functions only)")

    # Skip examples that require external data files
    if 'load_dataset("data/' in code and "create_" not in prepare_code_for_testing(
        code
    ):
        pytest.skip("Requires external data files")

    # Prepare code with mock models
    prepared_code = prepare_code_for_testing(code)

    # Execute the code
    try:
        exec_globals = {"__name__": "__main__"}
        exec(prepared_code, exec_globals)
    except Exception as e:
        pytest.fail(
            f"Execution error in {filename} block {block_num} (line {line_num}):\n"
            f"  {type(e).__name__}: {str(e)}"
        )


@pytest.mark.smoke
def test_documentation_examples_exist():
    """Verify that documentation example files exist and contain code blocks."""
    docs_dir = Path(__file__).parent.parent / "docs" / "docs" / "examples"

    assert docs_dir.exists(), f"Documentation examples directory not found: {docs_dir}"

    doc_files = list(docs_dir.glob("*.md"))
    assert len(doc_files) > 0, "No documentation markdown files found"

    total_blocks = 0
    for doc_file in doc_files:
        blocks = extract_code_blocks_from_md(doc_file)
        total_blocks += len(blocks)

    assert total_blocks > 0, "No Python code blocks found in documentation"
    print(
        f"\nFound {len(doc_files)} documentation files with {total_blocks} code blocks"
    )


if __name__ == "__main__":
    # Run just the smoke test
    pytest.main([__file__, "-k", "test_documentation_examples_exist", "-v"])
