# CaseForge

**GUI-first desktop application for generating software-engineering benchmark test cases using Qwen models.**

CaseForge is a standalone helper tool that uses a configured Qwen model to INVENT new benchmark cases for later import into AuditPup Model Evaluation.

## Installation and Launching

On Linux, install the supplied `caseforge_<version>_amd64.deb` with your
desktop software installer. After installation, launch **CaseForge** from the
application menu under **Development**. No terminal, Python installation, or
virtual environment is required for normal use.

Configure the provider endpoint, model, optional API-key environment variable,
generation settings, and output folder in the graphical interface. Generate,
preview, and save benchmark cases with the visible GUI controls.

Windows source and installer packaging are outside the scope of this Linux
package pass; the application code remains cross-platform.

## Features

- **Graphical Interface**: No CLI commands to remember - everything is configurable through the GUI
- **Two-Stage Generation**: Plans are validated for diversity before full case generation
- **Diversity Protection**: Every generated case must be meaningfully different from existing cases
- **Private Ground Truth**: Expected answers stored separately from benchmark prompts
- **Resume Support**: Continue generating cases across sessions without losing diversity tracking
- **Configurable Conclusion Distribution**: Control mix of BUG, CORRECT, UNSUPPORTED, NEEDS_CONTEXT, and INTENTIONAL cases

## GUI Sections

### Model / Provider Configuration
- API Endpoint (base URL)
- Model ID
- API key (optional, entered graphically and not saved)
- API Key environment variable (optional)
- Test Connection button
- Connection status indicator

### Generation Settings
- Number of cases (1-100)
- Language selection (multi-select)
- Difficulty checkboxes (Easy, Medium, Hard)
- Max retries per case
- Output folder chooser

### Expected Result Distribution
- Concrete Defect (BUG): ~40%
- Correct / Already Handled: ~30%
- Unsupported by Evidence: ~15%
- Needs Additional Context: ~10%
- Intentional Behavior: ~5%

### Generated Cases Panel
- Table view with Title, Language, Domain, Difficulty, Result Type, Status
- Preview selected cases
- Delete from list (files preserved)
- Save Selected / Save All verification

## How It Works

### Two-Stage Generation

1. **Stage 1 - Case Plan**: The model produces structured JSON describing a proposed case including title, language, technical domain, primary concept, failure mechanism, expected conclusion, difficulty, code shape, and semantic signature.

2. **Diversity Validation**: The plan is checked against all previously accepted cases. Rejected if:
   - Primary concept duplicates or is too similar (>85%) to existing
   - Same failure mechanism with similar concept (>60%)
   - Semantic signature substantially equivalent (>90%)
   - Code sample too similar (>75%)

3. **Stage 2 - Full Case**: Only after passing diversity validation, the model creates the complete markdown benchmark prompt.

### Output Files

Each accepted case produces:
- `generated_tests/001_<name>.md` - The benchmark prompt (for AuditPup import)
- `generator_state.json` - Private ground truth and metadata (never shared)

The `.md` files contain ONLY the prompt text. Expected answers, conclusions, and generation metadata are stored privately in the state file.

## Model Configuration

CaseForge supports any OpenAI-compatible chat completions endpoint:

### Local Qwen Server (LM Studio, etc.)
- Base URL: `http://localhost:1234/v1`
- Model: `qwen` or your model name
- API Key: leave empty for a local server

### Hosted Provider
- Base URL: Provider's endpoint
- Model: Provider's model identifier
- API Key: enter it in the graphical field; it is not saved
- API Key Env: optional environment-variable fallback, e.g., `OPENAI_API_KEY`

## Diversity Rules

CaseForge rejects candidates that reuse substantially the same:
- Primary concept
- Failure mechanism  
- Semantic signature
- Reasoning challenge

Changing only programming language, variable names, filenames, surface syntax, or domain nouns does NOT make a case distinct.

## Existing Concepts (Occupied)

The following concepts from existing benchmarks are treated as occupied:
- Canonicalized value computed but discarded before enqueue/use
- Immutable update correctly captured and used
- Write failure incorrectly reports success
- Swallowed read/IO failure leaves invalid/null state
- Resource/stream not closed
- Null/blank guard correctly protects later access
- Parameterized SQL query already safe
- Explicit sealed success/failure result correctly handled
- Guard/bounds check occurs after dangerous access
- Correct try/finally behavior
- Local-only counter with no demonstrated race
- Intentional error-handling behavior

## Technical Domains

Supported domains include:
- Error handling
- Resource management
- Bounds/indexing
- Nullability
- Immutability
- State management
- Collections
- Concurrency
- Locking
- Async control flow
- Transactions
- SQL/database use
- Serialization/parsing
- Caching
- Filesystem behavior
- Numeric conversion
- Ownership/lifetime
- API contracts
- Iterator/stream lifecycle
- Validation
- Cleanup
- Data transformation
- Exception propagation

## Developer packaging and testing

Maintainers can build the Linux package with the repository-local environment:

```bash
./.venv/bin/python packaging/build_deb.py
```

This produces the one-directory frozen application under `dist/CaseForge/` and
the installable package at `dist/caseforge_<version>_amd64.deb`.

Run automated tests (no live provider required) with the same environment:

```bash
./.venv/bin/python -m pytest
```

Tests cover:
- Backend component integration
- Diversity validation
- Storage round-trip
- Output numbering and file writing
- SHA-256 recording
- Ground truth privacy
- Provider configuration
- Error handling
- Resume behavior
- GUI/backend separation

## Limitations

- Requires working Qwen endpoint; quality depends on model
- Uses deterministic similarity (SequenceMatcher, Jaccard) - no embeddings
- Gives up after max retries rather than creating bogus cases
- Generated code not compiled/validated
- Sequential generation (single-threaded)

## Project Structure

```
benchmark_case_generator/
    __init__.py          # Package metadata
    client.py            # OpenAI-compatible API client
    diversity.py         # Similarity checking and validation
    generation.py        # Two-stage generation logic
    gui.py               # PySide6 graphical interface
    models.py            # Data classes (CasePlan, GeneratedCase)
    resources.py         # Cross-platform resources and user-state paths
    storage.py           # State manifest and file management
    theme.py             # Centralized branding colors

assets/
    icon.ico             # Application/window icon
    rivet_logo.png       # Intended TaskPuppyKreations logo

debian/
    control             # Debian source metadata
    caseforge.desktop   # Application-menu entry

packaging/
    build_deb.py        # Maintainer-only frozen/.deb build

tests/
    test_client.py
    test_diversity.py
    test_generation.py
    test_gui.py
    test_models.py
    test_storage.py

pyproject.toml
CaseForge.spec
README.md
```

## License

MIT
