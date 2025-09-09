"""
Setup configuration for MeshAI Core package
"""

from setuptools import setup, find_packages
import os

# Read the version
version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
if os.path.exists(version_file):
    with open(version_file, 'r') as f:
        version = f.read().strip()
else:
    version = "1.0.0"

# Read the README
readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
with open(readme_file, 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="meshai-core",
    version=version,
    author="MeshAI Labs",
    author_email="engineering@meshai.dev",
    description="Core libraries and utilities for MeshAI services",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meshailabs/meshai-core",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "asyncpg>=0.27.0",
        "bcrypt>=4.0.0",
        "redis>=4.5.0",
        "cachetools>=5.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "meshai-validate=meshai_core.auth.cli:main",
        ],
    },
)