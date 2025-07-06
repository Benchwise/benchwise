from setuptools import setup, find_packages

core_requirements = [
    "pydantic>=2.0.0",
    "httpx>=0.24.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "requests>=2.28.0",
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]

optional_requirements = {
    "metrics": [
        "rouge-score>=0.1.2",
        "sacrebleu>=2.3.0",
        "bert-score>=0.3.13",
        "nltk>=3.8.0",
    ],
    "llm-apis": [
        "openai>=1.0.0",
        "anthropic>=0.7.0",
        "google-generativeai>=0.3.0",
    ],
    "transformers": [
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-mock>=3.10.0",
        "pytest-cov>=4.0.0",
        "ruff>=0.1.6",
        "pre-commit>=3.0.0",
        "mypy>=1.0.0",
        "psutil>=5.9.0",
    ],
}

optional_requirements["all"] = [
    req for reqs in optional_requirements.values() for req in reqs
]


# Read version from __init__.py
def get_version():
    with open("benchwise/__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip("\"'")
    return "0.1.0-alpha"


setup(
    name="benchwise",
    version=get_version(),
    description="The GitHub of LLM Evaluation - Python SDK",
    long_description="BenchWise SDK for LLM evaluation and benchmarking",
    long_description_content_type="text/plain",
    author="Bhuvnesh Sharma",
    author_email="bhuvnesh875@gmail.com",
    url="https://github.com/devilsautumn/benchwise",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=core_requirements,
    extras_require=optional_requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    keywords=["llm", "evaluation", "benchmarking", "ai", "ml"],
    entry_points={
        "console_scripts": [
            "benchwise=benchwise.cli:main",
        ],
    },
)
