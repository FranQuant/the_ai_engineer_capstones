<img src="https://theaiengineer.dev/tae_logo_gw_flatter.png" width=35% align=right>

# AI Agents & Automation — Code Companion

This repository accompanies the book *AI Agents & Automation — Engineering Intelligent Workflows*.
It curates the runnable source code examples and the Colab-ready notebooks so you can explore,
modify, and extend every concept introduced in the text without digging through the manuscript.

## Repository Layout

- `code/`: Python source files grouped by chapter (`code/examples/`) and figure generation
  scripts (`code/figures/`). Each script includes inline comments and docstrings to explain the
  control flow and the architectural decisions behind the implementation.
- `notebooks/`: Chapter-aligned Jupyter notebooks that mirror the narrative, provide executable
  walkthroughs, and contain additional guidance between cells.
- `requirements.txt`: Minimal dependency set required for the examples and notebooks. Optional
  integrations (e.g., LangChain, LlamaIndex) are marked clearly inside the code.
- `tools/`: Utility scripts (for example, validation helpers) that ship with the companion repo.

## Getting Started

1. Clone the repository from GitHub once it is published.
2. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Open the notebooks folder in JupyterLab or run the Python scripts directly.

### Notebook Tips

- Every notebook is self-contained and prefers standard libraries or widely available packages.
- Execute cells sequentially; many notebooks build small helper classes inline for clarity.
- Optional integrations (such as framework snapshots) are guarded with explanatory comments and
  can be skipped if the dependency is not available.

## Disclaimer

The code, notebooks, and utilities in this repository are provided for educational and illustrative
purposes only. They come with no warranties or guarantees of correctness, fitness for a particular
purpose, or suitability in production. Use at your own risk. Agentic systems — especially those that
operate autonomously — can carry material risks, including unintended actions, data loss or leakage,
excessive resource usage, regulatory non-compliance, and safety violations. Always validate outputs,
apply least-privilege principles, instrument observability, and put strong safeguards in place before
deployment. By using any code or patterns described here, you acknowledge and accept these risks. The
authors and publisher assume no responsibility for any consequences arising from their use.

### Contributing

If you spot an issue or want to contribute improvements, feel free to open a pull request in the
companion repository. Please follow the coding guidelines described in the book’s `RULES.md`.

---

© Dr. Yves J. Hilpisch | The Python Quants GmbH  \
AI-Powered by GPT-5.

