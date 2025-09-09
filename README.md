# MeshAI Core

Central repository for shared libraries, utilities, and packages used across MeshAI services.

## Structure

```
meshai-core/
├── meshai_core/            # Main package directory
│   ├── __init__.py
│   ├── auth/              # Authentication and authorization utilities
│   │   ├── __init__.py
│   │   └── api_key_validator.py   # Shared API key validation library
│   ├── database/          # Database utilities
│   │   ├── __init__.py
│   │   └── models.py      # Shared database models
│   └── utils/             # Common utilities
│       ├── __init__.py
│       ├── logging.py     # Standardized logging
│       └── metrics.py     # Metrics collection
├── setup.py               # Package setup for pip installation
├── requirements.txt       # Core dependencies
└── VERSION               # Version tracking
```

## Installation

### As a Git Submodule

```bash
cd your-service
git submodule add https://github.com/yourusername/meshai-core.git core
git submodule init
git submodule update
```

### As a Python Package

```bash
pip install git+https://github.com/yourusername/meshai-core.git
```

### For Development

```bash
pip install -e git+https://github.com/yourusername/meshai-core.git#egg=meshai-core
```

## Usage

### API Key Validator

```python
from meshai_core.auth.api_key_validator import MeshAIAPIKeyValidator

validator = MeshAIAPIKeyValidator(
    db_url="postgresql://...",
    redis_url="redis://...",
    service_keys=["key1", "key2"],
    admin_keys=["admin1"],
    enable_caching=True,
    enable_circuit_breaker=True
)

await validator.initialize()
result = await validator.validate(api_key)
```

## Version

Current version: 1.0.0

## License

Proprietary - MeshAI Labs
