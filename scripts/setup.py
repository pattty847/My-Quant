# setup.py
from setuptools import setup, find_packages

setup(
    name="quant_system",
    version="0.1.0",
    description="Quantitative Trading System with LLM Integration",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "ccxt>=3.0.0",
        "ta>=0.10.0",  # Technical analysis indicators
        "requests>=2.28.0",
        "python-dotenv>=1.0.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "viz": [
            "plotly>=5.13.0",
            "kaleido>=0.2.1",  # For static image export in plotly
        ],
    },
    entry_points={
        "console_scripts": [
            "quant-system=main:start_cli",
            "quant-api=main:start_api",
        ],
    },
    python_requires=">=3.8",
)