# CLAUDE.md — How We Work Together

This file describes the collaboration patterns, conventions, and workflow
preferences. Claude Code should follow these when assisting on any project.

---

## Project Wiki — read this first

**At the start of every session, read `wiki/wiki.md`** — the durable, aggregate memory of this
repo (where we are, what we've concluded, decisions/conventions, where things live). It survives
context compaction and new sessions; use it to pick up exactly where we left off instead of
re-deriving state from the notebook. **One wiki per repository:** `wiki/wiki.md`.

**Claude updates it autonomously (then notes it in one line), incrementally, so it's always
current** — that, not a last-second flush, is what survives compaction. **Never ask permission
to update the wiki — just do it and give the one-line notice.** Triggers:

- A **finding settles** — an analysis reaches a conclusion worth remembering next session.
- A **decision/definition is made or changed** — a parameter, cohort definition, approach, naming, pivot.
- A **reusable artifact stabilizes** — a new `functions.py` helper, a config knob, a notebook section moved to `.py`.
- On **session wind-down**, do a reconcile pass; update on **explicit request** ("update the wiki").

Do **not** update on every message / edit / intermediate run / exploratory question — record
*settled* state, not the play-by-play.

When writing: **reconcile, don't blindly append** (fix anything new results contradicted); add the
**date + provenance** (cell id / output file / config key); **aggregate only** — findings, counts,
gene names, gene-sets, metrics — **never** SMILES / compound IDs / per-compound values; then give a
**one-line notice**.

---

## Development Workflow

### 1. Prototype in Jupyter, then move to Python

- New features start in a Jupyter notebook — test interactively, inspect outputs, iterate
- Once stable, move logic to the appropriate `.py` file
- Keep the notebook as a step-by-step demo with per-step inspection and checkpoints
- Use `%autoreload 2` so code changes in `.py` files are picked up automatically

### 2. Config over code

- All tuneable parameters live in a YAML config file — never hardcode values in Python
- When adding a new parameter: add it to the YAML with a comment, read it with `.get("key", default)`, update docs
- To run the same code on a different use case, copy the YAML and change the values

### 3. Keep things organized

- One function per concern — no monolithic functions doing multiple things
- Shared helpers in a dedicated `functions.py` (or `utils.py`)
- Prompts in a dedicated `prompts.py` — never inline LLM prompts in logic code
- Config in YAML, logic in Python, prompts as data
- **Jupyter notebooks:** consolidate all imports at the top in one cell — no scattered imports throughout the notebook

### 4. Separate run from assertions in test notebooks

- Cell 1: run the code / load from checkpoint
- Cell 2: assertions and validation
- This way assertions can be tweaked without re-running expensive operations

---

## Dependency Management

### Python Packages and Conda Environments

- **NEVER install packages to the base conda environment** — always use a dedicated environment
- The base conda environment is shared and can break existing libraries if modified
- When packages need to be installed:
  1. Ask the user to create/activate their own conda environment (e.g., `llms`)
  2. OR inform the user that packages need to be installed and let them handle it
  3. NEVER run `pip install` or `conda install` to the base environment without explicit permission
- The user will manually create environments like: `conda create -n llms python=3.12`
- Document any new dependencies by adding them to `requirements.txt`

---

## Data Privacy & Local-Only Execution

**The project's data, SMILES, labels, screening results, and any derived features must never leave this machine.** This is a hard rule, not a preference.

### Do

- **Run all models locally** — RDKit, sklearn, XGBoost, LightGBM, ChemProp, PyTorch (CPU/GPU local), local ONNX, etc.
- **Use `WebSearch` / `WebFetch` only for library documentation, blog posts, and public references** — never with project data in the query string or POST body.
- **Pin library versions** in `requirements.txt` so the local environment is reproducible without re-fetching from the internet later.

### Don't

- **Don't send project data to any cloud LLM API.** No OpenAI / Anthropic / Gemini / Cohere / Mistral / Together / Replicate calls that include compound IDs, SMILES, predictions, or labels — not even "anonymised" snippets.
- **Don't import packages that phone home by default** without first checking. If a library has telemetry, opt out (e.g. `WANDB_MODE=offline`, `MLFLOW_TRACKING_URI=file:./mlruns`).
- **Don't paste data into web tools.** No diagram renderers, no pastebins, no gists, no shared notebooks — even if "just for visualisation".
- **Don't sync output folders to cloud storage.** Keep `data/`, `output/`, `autoresearch/logs/`, and `tests/files/` out of any auto-syncing path.

### Claude Code itself is a cloud LLM — the agent's tool outputs cross to Anthropic

Claude Code (the assistant editing this codebase) runs on Anthropic's API. Every `Read` / `Bash` / `Grep` result is sent to Anthropic for inference, regardless of subscription tier (Pro / Team / Enterprise). Subscription tier only changes retention and training terms; it does **not** change whether data crosses the wire.

Operational rules for the assistant in this project:

- **Don't `Read`** files that contain chemical structures: `data/chemical_libs/*.csv`, raw SDFs, fingerprint matrices, anything with a `smiles` / `mol` / `inchi` column.
- **Don't run `Bash`** that prints rows of `serac_df`, `df_raw`, `MF_features`, or any DataFrame with a `smiles` column. Schema is fine (`df.columns.tolist()`, `df.shape`, `df.dtypes`); values are not (`df.head()`).
- **Don't `Grep`** file *contents* for compound IDs (`SRB-XXXXXXX-NNN` format). Searching file names / paths is fine.
- **Don't echo** notebook outputs that render structure thumbnails, top-K SMILES tables, or per-compound logfc.
- **Don't `Read` Jupyter notebooks (or any file with saved/executed outputs) that may contain structures.** Opening a `.ipynb` returns its cell *outputs* — `df.head()` tables, structure thumbnails, top-K SMILES, per-compound values — which cross the wire even if you only wanted the code. To inspect or edit a notebook, extract **cell sources only** (strip `outputs`) with a small script, or edit by known cell-id; never read the whole notebook blind. Keep notebook outputs cleared before any tooling reads them.
- **Aggregate-only is fine**: gene names (HGNC is public), per-gene R² values, plate IDs, compound counts, model hyperparameters, code, configs.
- **Public reference SMILES are fine** for unit tests: ethanol (`CCO`), benzene (`c1ccccc1`), aspirin, or anything from RDKit's example data.
- **Schema-then-synthesize for testing**: to test code against a real file's *shape*, read only its schema — column names and dtypes (`pd.read_csv(path, nrows=0).columns`, `df.dtypes`) — never the values, then **generate synthetic data matching that schema** (random floats, fake `C_001`/`G_001` IDs, public SMILES like `CCO`) and run the test on the synthetic frame. This keeps real values off the wire while still exercising the code. Prefer this over reading rows.
- **Run heavy diagnostics in the user's notebook, not in your own process**: when a check needs `df_raw`/`measure`/large exports, hand the user a snippet to run in the live kernel (frames already in memory) rather than loading the files yourself — avoids both the data crossing the wire and OOM-killing the machine.
- If chemistry inspection is genuinely needed, ask the user to paste an *anonymised* summary (`C_001`, `G_001` with the mapping kept local), rather than reading the source file.

This applies even with a Teams / Enterprise subscription. Zero-Data-Retention (ZDR) addenda reduce retention but don't eliminate the in-flight transit — the rule above is operational hygiene that's independent of which contract was signed.

### When evaluating a new dependency

Before adding a library to `requirements.txt`, confirm:

1. It runs entirely offline once installed (no API key required, no auto-update calls during inference).
2. It has no opt-in telemetry that's enabled by default; if it does, the opt-out is documented in `requirements.txt` next to the entry.
3. The model weights are downloaded once at install time and cached locally (no per-inference download).

If a tool would meaningfully accelerate the work but only runs as a hosted API, **flag it and wait for explicit approval** rather than using it silently.

---

## Coding Conventions

### Do

- **Read before writing** — always read a file before editing it
- **Check for existing patterns** — match the style already in the codebase
- **Write simplified code** — the fewest clear statements: inline single-use intermediates, derive lists directly, collapse verbose literals; no redundant variables or steps. Prefer a vectorized/built-in expression over a hand-rolled helper function when equally clear (e.g. `.str.extract(...).astype(float)` over a custom `apply` function). Simplify before reporting done — readability first (simple ≠ cryptic one-liner).
- **Keep comments to 1 line** — 2 lines max, and only if really necessary. Applies to inline/block comments (incl. notebook cell-top `##` comments); docstrings are exempt.
- **Encapsulate state properly** — use classes or function parameters instead of global variables
- **Update docs when changing code** — keep documentation in sync
- **Break complex tasks into a todo list** — write out the steps, get approval, then execute
- **Save checkpoints** after expensive operations so work can be resumed
- **Use descriptive docstrings on test functions** — describe input, expected output, rationale
- **Add a line comment above each assert** — briefly describe what is being checked
- **Verify every code change before reporting it as done** — after editing, run a quick check appropriate to the change: `grep` to confirm a string was removed/added everywhere it should be, a one-liner `python -c "..."` smoke test for new/modified Python functions, a notebook cell re-run for plotting tweaks, etc. Never claim a change works based on the diff alone.

### Don't

- Don't hardcode parameters — put them in config
- Don't use global variables — use classes, function parameters, or return values to pass state
- Don't truncate data flowing between components — truncation is only for display
- Don't add features beyond what was asked
- Don't create new files unless necessary — prefer editing existing ones
- Don't write documentation files unless explicitly requested
- Don't add error handling, fallbacks, or abstractions for hypothetical future requirements
- **Don't delete any files or directories without explicit permission** — even if they seem unused or large (like virtualenvs, cache directories, or data files), always ask first before removing them

---

## Testing

### Approach

- Write tests that can run from saved checkpoints (no need to re-run expensive operations)
- Hard asserts check required fields and data types
- LLM structure validation checks output matches the prompt schema
- Mirror notebook test structure in `.py` unit tests so both stay in sync

### When adding a new component

1. Prototype and test in the notebook first
2. Move to `.py` file once stable
3. Add unit tests that mirror the notebook cells
4. Update documentation

---

## Communication Preferences

- **Be concise** — lead with the answer, skip preamble
- **Show don't tell** — code snippets over long explanations
- **Ask before big changes** — propose the approach, wait for approval
- **Let me know what you find** before implementing — especially for refactors
- **Don't add features I didn't ask for** — stay focused on the request
- **Update docs alongside code** — documentation should always match the code
- **When a task is complex, decompose it into a todo list first** — then execute step by step
