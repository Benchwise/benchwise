#!/usr/bin/env python3
"""
Test runner for BenchWise SDK with detailed output

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --basic      # Run only basic tests (no slow/API tests)
    python run_tests.py --coverage   # Run with coverage report
    python run_tests.py --file test_core.py  # Run specific file
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path


def run_command_with_output(cmd, description):
    """Run a command and show real-time output"""
    print(f"\n🔄 {description}...")
    print(f"📋 Command: {' '.join(cmd)}")
    print("=" * 80)

    start_time = time.time()

    try:
        # Run without capturing output so we see it in real-time
        result = subprocess.run(cmd, text=True)

        end_time = time.time()
        duration = end_time - start_time

        print("=" * 80)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully in {duration:.2f}s")
            return True
        else:
            print(
                f"❌ {description} failed with exit code {result.returncode} after {duration:.2f}s"
            )
            return False

    except KeyboardInterrupt:
        print(f"\n⚠️ {description} interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in {description}: {e}")
        return False


def check_dependencies():
    """Check if test dependencies are installed"""
    print("🔍 Checking test dependencies...")

    required_packages = {"pytest": "pytest", "pytest-asyncio": "pytest_asyncio"}

    missing = []
    available = []

    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            available.append(package_name)
        except ImportError:
            missing.append(package_name)

    if available:
        print(f"✅ Available: {', '.join(available)}")

    if missing:
        print(f"❌ Missing: {', '.join(missing)}")
        print("💡 Install with: pip install -e '.[dev]'")
        return False

    print("✅ All required dependencies are available")
    return True


def check_optional_dependencies():
    """Check optional test dependencies"""
    print("\n🔍 Checking optional dependencies...")

    optional_packages = {
        "rouge-score": "rouge_score",
        "sacrebleu": "sacrebleu",
        "bert-score": "bert_score",
        "sentence-transformers": "sentence_transformers",
    }

    available = []
    missing = []

    for package_name, import_name in optional_packages.items():
        try:
            __import__(import_name)
            available.append(package_name)
        except ImportError:
            missing.append(package_name)

    if available:
        print(f"✅ Available: {', '.join(available)}")
    if missing:
        print(f"⚠️ Missing (some tests may be skipped): {', '.join(missing)}")
        print("💡 Install all with: pip install -e '.[all]'")

    return len(available) > 0


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced test runner for BenchWise SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_tests.py                    # Run all tests
    python run_tests.py --basic            # Basic tests only
    python run_tests.py --coverage         # With coverage
    python run_tests.py --file test_core   # Specific file
    python run_tests.py --test accuracy    # Specific test pattern
    """,
    )

    parser.add_argument(
        "--basic",
        action="store_true",
        help="Run only basic tests (exclude slow/API tests)",
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Run with coverage report"
    )
    parser.add_argument("--file", help="Run specific test file (e.g., test_core.py)")
    parser.add_argument("--test", help="Run tests matching pattern")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--no-header", action="store_true", help="Skip header information"
    )

    args = parser.parse_args()

    if not args.no_header:
        print("🧪 BenchWise SDK Test Runner")
        print("=" * 40)

    # Check if we're in the right directory
    if not Path("benchwise").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   Expected to find 'benchwise' directory here")
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check optional dependencies
    has_optional = check_optional_dependencies()
    if not has_optional and not args.basic:
        print("\n⚠️ Warning: No optional dependencies found.")
        print("   Some tests may fail. Consider using --basic flag.")
        print("   Or install dependencies with: pip install rouge-score sacrebleu")

    # Build pytest command
    cmd_parts = ["python", "-m", "pytest"]

    # Add verbosity
    if args.verbose:
        cmd_parts.extend(["-v", "-s"])
    else:
        cmd_parts.extend(["-v"])

    # Add test selection based on arguments
    test_description = "all tests"
    if args.basic:
        # Don't use markers, just run specific test files
        cmd_parts.extend(
            [
                "tests/test_core.py",
                "tests/test_datasets.py",
                "tests/test_models.py",
                "tests/test_results.py",
                "tests/test_config.py",
            ]
        )
        test_description = "basic tests (core functionality)"
    elif args.file:
        if not args.file.startswith("tests/"):
            if not args.file.startswith("test_"):
                args.file = f"test_{args.file}"
            args.file = f"tests/{args.file}"
        if not args.file.endswith(".py"):
            args.file += ".py"
        cmd_parts.append(args.file)
        test_description = f"tests from {args.file}"
    else:
        cmd_parts.append("tests/")

    # Add test pattern if specified
    if args.test:
        cmd_parts.extend(["-k", args.test])
        test_description += f" matching '{args.test}'"

    # Add coverage if requested
    if args.coverage:
        cmd_parts.extend(
            [
                "--cov=benchwise",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
            ]
        )

    # Add useful pytest options
    cmd_parts.extend(
        [
            "--tb=short",  # Shorter traceback format
            "--color=yes",  # Colored output
            "--durations=5",  # Show 5 slowest tests
            "--maxfail=10",  # Stop after 10 failures
            "--disable-warnings",  # Cleaner output
        ]
    )

    # Show configuration
    if not args.no_header:
        print("\n📋 Test Configuration:")
        print(f"   • Mode: {'Basic' if args.basic else 'Full'}")
        print(f"   • Coverage: {'Yes' if args.coverage else 'No'}")
        print(f"   • Target: {test_description}")
        print(f"   • Verbose: {'Yes' if args.verbose else 'No'}")

    # Run tests
    success = run_command_with_output(cmd_parts, f"Running {test_description}")

    # Summary
    print("\n📊 Test Results Summary:")
    if success:
        print("🎉 Status: ALL TESTS PASSED!")
        if args.coverage:
            print("📈 Coverage report: htmlcov/index.html")
        print("\n✨ Great job! Your code is working correctly.")
    else:
        print("💥 Status: SOME TESTS FAILED!")
        print("\n🔧 Next steps:")
        print("   • Check the error messages above")
        print("   • Run with --verbose for more details")
        print("   • Try --basic to run core tests only")
        print("   • Run specific files with --file <filename>")

    print("\n" + "=" * 80)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
