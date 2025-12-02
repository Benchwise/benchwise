#!/usr/bin/env python3
"""
Test all code examples from a documentation file with REAL models.

This script extracts all Python code blocks from a documentation markdown file
and runs each one as a separate test with real OpenAI and Google models.

The script can find documentation files in multiple ways:
- Absolute path: /path/to/file.md
- Relative to project root: docs/docs/examples/classification.md
- Just filename (searches docs/ tree): classification.md

Usage:
    # Using just filename (searches in docs/ directory)
    python test_single_doc_file.py classification.md

    # Using relative path from project root
    python test_single_doc_file.py docs/docs/examples/classification.md
    python test_single_doc_file.py README.md
    python test_single_doc_file.py docs/docs/getting-started/quickstart.md

    # Syntax check only (no API calls)
    python test_single_doc_file.py --syntax-only classification.md

    # Save test results to files
    python test_single_doc_file.py --save-results classification.md
"""

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple


def extract_code_blocks(markdown_file: Path) -> List[Tuple[str, int, int]]:
    """
    Extract all Python code blocks from a markdown file.
    Returns list of (code, block_number, line_number) tuples.
    """
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'```python\n(.*?)```'
    matches = re.finditer(pattern, content, re.DOTALL)

    code_blocks = []
    for i, match in enumerate(matches, 1):
        code = match.group(1)
        line_number = content[:match.start()].count('\n') + 1
        code_blocks.append((code, i, line_number))

    return code_blocks


def prepare_code_for_real_models(code: str) -> str:
    """
    Replace model names with real OpenAI and Google models.
    Ensures we use exactly 2 models: gpt-3.5-turbo and gemini-2.5-flash
    """
    import re

    # Find all @evaluate decorators and replace models to ensure diversity
    def replace_evaluate_models(match):
        decorator = match.group(0)

        # Extract the content inside @evaluate(...)
        content = re.search(r'@evaluate\((.*)\)', decorator, re.DOTALL)
        if not content:
            return decorator

        params = content.group(1)

        # Split by comma, but be careful with nested structures
        # Extract all quoted strings (model names)
        model_pattern = r'"([^"]+)"'
        models = re.findall(model_pattern, params)

        if not models:
            return decorator

        # Always use exactly 2 models: gpt-3.5-turbo and gemini-2.5-flash
        # Take first N models and replace them, but cap at 2
        num_models = min(len(models), 2)
        new_models = ['"gpt-3.5-turbo"', '"gemini-2.5-flash"'][:num_models]

        # If there was only 1 model originally, keep it as 1 model
        if len(models) == 1:
            new_models = ['"gpt-3.5-turbo"']

        # Find any kwargs (parameters with =)
        # Split params and identify non-string parts (kwargs)
        kwargs = []
        # Remove all quoted strings and see what's left
        params_without_strings = re.sub(r'"[^"]*"', '', params)
        if '=' in params_without_strings:
            # Extract kwargs
            kwargs_match = re.search(r',?\s*(\w+\s*=\s*[^,)]+(?:,\s*\w+\s*=\s*[^,)]+)*)\s*$', params)
            if kwargs_match:
                kwargs.append(kwargs_match.group(1))

        # Reconstruct the decorator
        result = '@evaluate(' + ', '.join(new_models)
        if kwargs:
            result += ', ' + ', '.join(kwargs)
        result += ')'

        return result

    # Replace all @evaluate decorators
    modified_code = re.sub(r'@evaluate\([^)]+\)', replace_evaluate_models, code)

    # Replace placeholder dataset loading with actual datasets
    if 'load_dataset("data/qa_1000.json")' in modified_code:
        # Add import if not present
        if 'from benchwise' in modified_code and 'create_qa_dataset' not in modified_code:
            modified_code = modified_code.replace(
                'from benchwise import',
                'from benchwise import create_qa_dataset,'
            )
        modified_code = modified_code.replace(
            'load_dataset("data/qa_1000.json")',
            'create_qa_dataset(questions=["What is AI?", "What is ML?"], answers=["Artificial Intelligence", "Machine Learning"], name="qa_test")'
        )

    if 'load_dataset("data/news_articles.json")' in modified_code:
        # Add import if not present
        if 'from benchwise' in modified_code and 'create_summarization_dataset' not in modified_code:
            modified_code = modified_code.replace(
                'from benchwise import',
                'from benchwise import create_summarization_dataset,'
            )
        modified_code = modified_code.replace(
            'load_dataset("data/news_articles.json")',
            'create_summarization_dataset(documents=["Article about AI.", "Article about ML."], summaries=["AI summary", "ML summary"], name="news")'
        )

    return modified_code


def check_syntax(code: str) -> Tuple[bool, str]:
    """Check if Python code has valid syntax."""
    import ast
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Parse error: {str(e)}"


def run_code_sync(code: str, timeout: int = 90) -> Tuple[bool, str, str]:
    """Run code in subprocess and capture output."""
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        # Run in subprocess
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent
        )

        # Cleanup
        import os
        os.unlink(temp_file)

        output = result.stdout
        error = result.stderr

        if result.returncode == 0:
            return True, output, None
        else:
            return False, output, error

    except subprocess.TimeoutExpired:
        return False, "", f"Timeout after {timeout}s"
    except Exception as e:
        return False, "", f"Error: {str(e)}"


def test_code_block(code: str, block_num: int, line_num: int, syntax_only: bool = False) -> Tuple[bool, str]:
    """Test a single code block."""
    # Check syntax
    syntax_valid, syntax_error = check_syntax(code)
    if not syntax_valid:
        print(f"❌ SYNTAX ERROR")
        return False, f"Syntax Error: {syntax_error}"

    if syntax_only:
        print(f"✅ SYNTAX VALID")
        return True, None

    # Prepare code with real models
    prepared_code = prepare_code_for_real_models(code)

    # Skip incomplete examples (just function definitions without execution)
    if '@evaluate(' in prepared_code and 'asyncio.run' not in prepared_code:
        print(f"⏭️  SKIPPED (incomplete example - defines functions only)")
        return True, "Skipped: Incomplete example"

    # Run the code
    print(f"⏳ Running test...", end=" ", flush=True)
    start_time = time.time()
    success, output, error = run_code_sync(prepared_code, timeout=90)
    duration = time.time() - start_time

    if success:
        print(f"✅ PASSED ({duration:.2f}s)")
        return True, output
    else:
        print(f"❌ FAILED ({duration:.2f}s)")
        return False, error or output


def main():
    import argparse
    import json
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Test Python code examples from a documentation file")
    parser.add_argument('file', help='Documentation file to test. Can be:\n'
                                     '  - Relative path from project root (e.g., docs/docs/examples/classification.md)\n'
                                     '  - Absolute path (e.g., /path/to/file.md)\n'
                                     '  - Just filename (will search in docs/ directory tree)')
    parser.add_argument('--syntax-only', action='store_true', help='Only check syntax')
    parser.add_argument('--save-results', action='store_true', help='Save test results to files')
    args = parser.parse_args()

    # Find the documentation file
    project_root = Path(__file__).parent
    file_arg = Path(args.file)

    # Try different strategies to find the file
    doc_file = None

    # Strategy 1: Absolute path
    if file_arg.is_absolute() and file_arg.exists():
        doc_file = file_arg

    # Strategy 2: Relative to project root
    elif (project_root / file_arg).exists():
        doc_file = project_root / file_arg

    # Strategy 3: Search in docs directory tree
    else:
        docs_dir = project_root / 'docs'
        if docs_dir.exists():
            # Search for the file in docs directory tree
            for candidate in docs_dir.rglob(file_arg.name if file_arg.name else args.file):
                if candidate.is_file():
                    doc_file = candidate
                    break

    if doc_file is None or not doc_file.exists():
        print(f"❌ Error: File not found: {args.file}")
        print(f"\nSearched in:")
        print(f"  - Absolute path: {file_arg if file_arg.is_absolute() else 'N/A'}")
        print(f"  - Relative to project: {project_root / file_arg}")
        print(f"  - In docs/ directory tree")
        return 1

    # Get relative path for display
    try:
        display_path = doc_file.relative_to(project_root)
    except ValueError:
        display_path = doc_file

    print(f"\n🧪 Testing Documentation Examples")
    print(f"📄 File: {display_path}")

    if args.syntax_only:
        print("⚙️  Mode: Syntax check only")
    else:
        print("⚙️  Mode: Full execution with REAL models")
        print("🤖 Models: gpt-3.5-turbo, gemini-2.5-flash")
        print("⚠️  Note: This will make actual API calls and incur costs")

    # Extract code blocks
    code_blocks = extract_code_blocks(doc_file)

    if not code_blocks:
        print(f"\n❌ No Python code blocks found in {args.file}")
        return 1

    print(f"📝 Total code blocks: {len(code_blocks)}\n")
    print("=" * 80)

    # Test each code block
    results = []
    for code, block_num, line_num in code_blocks:
        print(f"\n{'=' * 80}")
        print(f"TEST {block_num}/{len(code_blocks)}: Block {block_num} (Line {line_num})")
        print("=" * 80)

        success, output_or_error = test_code_block(code, block_num, line_num, args.syntax_only)
        results.append((block_num, success, output_or_error))

        # Show output
        if success and output_or_error and output_or_error.strip() and not args.syntax_only:
            print("\n📋 OUTPUT:")
            print("-" * 80)
            output_lines = output_or_error.strip().split('\n')
            for line in output_lines[:50]:  # Show first 50 lines
                print(f"  {line}")
            if len(output_lines) > 50:
                print(f"  ... ({len(output_lines) - 50} more lines)")
            print("-" * 80)
        elif not success and output_or_error:
            print("\n❌ ERROR:")
            print("-" * 80)
            error_lines = output_or_error.split('\n')
            for line in error_lines[:40]:  # Show first 40 lines
                print(f"  {line}")
            if len(error_lines) > 40:
                print(f"  ... ({len(error_lines) - 40} more lines)")
            print("-" * 80)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)

    total = len(results)
    passed = sum(1 for _, success, _ in results if success)
    failed = total - passed

    print(f"\nFile: {display_path}")
    print(f"Total: {total} code blocks")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {passed/total*100:.1f}%")

    # Show failures
    if failed > 0:
        print(f"\n{'-' * 80}")
        print("FAILED TESTS")
        print("-" * 80)
        for block_num, success, output_or_error in results:
            if not success:
                print(f"\n❌ Block {block_num}")
                if output_or_error:
                    print(f"   {output_or_error[:200]}")

    print(f"\n{'=' * 80}\n")

    # Save results if requested
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a clean base name from the file path
        base_name = doc_file.stem  # Gets filename without extension

        # Create results directory
        results_dir = Path(__file__).parent / 'test_results'
        results_dir.mkdir(exist_ok=True)

        # Save JSON results (detailed)
        json_file = results_dir / f"{base_name}_{timestamp}.json"
        json_data = {
            "file": str(display_path),
            "full_path": str(doc_file),
            "timestamp": datetime.now().isoformat(),
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed/total*100,
            "syntax_only": args.syntax_only,
            "results": [
                {
                    "block": block_num,
                    "success": success,
                    "output": output_or_error[:500] if output_or_error else None,  # Truncate long outputs
                }
                for block_num, success, output_or_error in results
            ]
        }

        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"💾 JSON results saved to: {json_file}")

        # Save Markdown summary
        md_file = results_dir / f"{base_name}_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write(f"# Test Results: {display_path}\n\n")
            f.write(f"**File:** `{doc_file}`\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Models:** gpt-3.5-turbo, gemini-2.5-flash\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- Total Tests: {total}\n")
            f.write(f"- ✅ Passed: {passed}\n")
            f.write(f"- ❌ Failed: {failed}\n")
            f.write(f"- Success Rate: {passed/total*100:.1f}%\n\n")

            if failed > 0:
                f.write(f"## Failed Tests\n\n")
                for block_num, success, output_or_error in results:
                    if not success:
                        f.write(f"### Block {block_num}\n\n")
                        f.write(f"```\n{output_or_error[:300] if output_or_error else 'No error details'}\n```\n\n")

        print(f"📝 Markdown summary saved to: {md_file}")

        # Save to latest file (overwrite)
        latest_json = results_dir / f"{base_name}_latest.json"
        with open(latest_json, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"📌 Latest results: {latest_json}")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
