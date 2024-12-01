from setuptools import setup, find_packages

setup(
    name="monica-rag",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests>=2.25.0",
        "numpy>=1.19.0",
        "sentence-transformers>=2.0.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ]
    },
    python_requires=">=3.8",
)