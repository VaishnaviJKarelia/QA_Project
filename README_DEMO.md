Agentic RAG Demo (QA_Project)

Quick start (macOS):

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your Google API key (required): the app no longer falls back to a hard-coded key for security.

```bash
export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

Note: The app will automatically load a `.env` file if present (using python-dotenv). You can create a `.env` file locally by copying `.env.example` and filling in your key; the app will load it on startup so you don't need to export the variable manually.
4. Run the Streamlit app:

```bash
streamlit run main.py
```

Notes
- The agent runs with verbose=True, so Thought/Action/Observation traces will be printed to the terminal where you launched Streamlit; use this during demos to show agent reasoning.
- There's a mocked test at `tests/test_agent_mock.py` which patches the RetrievalQA chain and ChatGoogleGenerativeAI so you can run the test without network/API calls.

Run tests:

```bash
pytest -q
```

If you want me to:
- Add a GitHub Actions workflow to run the tests automatically.
- Replace the fallback hard-coded API key in `main.py` with a safer pattern or throw if not present.
- Pin exact working versions for your environment after you confirm which langchain/google SDK versions you have.
