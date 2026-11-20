<div align="center">
  <img src="logo.png" alt="ExamForge Logo" width="120"/>
  <h1>ExamForge</h1>
  <p><strong>AI-powered Exam Question Generator</strong></p>
</div>

<br>

ExamForge is a professional web application that allows educators to upload a syllabus PDF, automatically extract its structure, and use AI to dynamically generate a complete exam question paper based on a custom marks distribution.

## Features

- **Dynamic Syllabus Parsing:** Automatically extracts text from uploaded PDFs and uses AI to identify topics and chapters without assuming rigid formats.
- **Custom Exam Configuration:** Define your own exam name, subject, difficulty (Easy, Medium, Hard), and exact question distribution (e.g., 4 questions of 5 marks, 2 questions of 10 marks).
- **Dual AI Backends:**
  - **Ollama (Local):** Run 100% offline, private, and free using local models like `llama3` or `qwen`. Auto-detects installed models.
  - **OpenAI (Cloud):** Use powerful models like `gpt-4o-mini` or `gpt-4o` for maximum quality and speed via API key.
- **Smart Generation:** Generates non-repetitive, topic-balanced questions adapting depth to the assigned marks.
- **PDF Export:** Downloads the final generated question paper as a clean, professionally formatted PDF.
- **Light & Dark Themes:** Clean slate-and-indigo interface built on Streamlit that follows your system's light or dark setting.

## Screenshots

Screenshots follow your GitHub theme: switch GitHub between light and dark mode to see both.

### 1. Upload & Analyze Syllabus
Upload your syllabus and select your AI backend (Local or Cloud).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/screenshots/upload-dark.png">
  <img alt="ExamForge upload tab with discovered syllabus topics" src="assets/screenshots/upload-light.png">
</picture>

### 2. Exam Configuration
Define your exact exam marks and difficulty level.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/screenshots/configuration-dark.png">
  <img alt="Exam configuration with exam name, difficulty and marks distribution" src="assets/screenshots/configuration-light.png">
</picture>

### 3. Generated Question Paper
Instantly generate and review balanced questions tailored to your marks distribution.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/screenshots/generated-paper-dark.png">
  <img alt="Generated question paper with questions, topics and marks" src="assets/screenshots/generated-paper-light.png">
</picture>

### 4. PDF Export
Download a professionally formatted question paper ready for print.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/screenshots/export-dark.png">
  <img alt="Export tab with the Download Question Paper PDF button" src="assets/screenshots/export-light.png">
</picture>

## Setup Instructions

### Prerequisites
- Python 3.11+
- (Optional) [Ollama](https://ollama.com/) for local AI generation.

### Installation

1. **Clone the repository** (or navigate to the folder):
   ```bash
   cd examforge
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Configure settings**:
   Copy `.env.example` to `.env` and fill in what you need, for example an OpenAI key:
   ```env
   OPENAI_API_KEY=sk-...
   ```
   *Note: You can also just type your API key directly into the app sidebar!*
   All settings are listed in [Configuration](#configuration).

### Running the App

Start the Streamlit server:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Local AI Guide (Ollama)

If you prefer to run ExamForge entirely offline:
1. Download and install [Ollama](https://ollama.com/).
2. Pull a model in your terminal (e.g., Qwen 2.5):
   ```bash
   ollama pull qwen2.5:7b
   ```
3. Ensure the server is running (`ollama serve`).
4. ExamForge will automatically detect your local models!

## Configuration

Settings are read from environment variables, or from a `.env` file for local use (real environment variables take precedence). See [`.env.example`](.env.example). All are optional.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | Server's OpenAI key, used when a visitor doesn't enter their own. Never sent to the browser. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address. `OLLAMA_HOST` is also accepted. |
| `EXAMFORGE_PASSWORD` | none | Password required to use the app. **Set this before exposing the app publicly.** |
| `EXAMFORGE_MAX_AI_REQUESTS_PER_HOUR` | `120` | App-wide cap on AI requests using the server's key or Ollama. Visitors using their own key aren't counted. `0` = unlimited. |
| `EXAMFORGE_LLM_TIMEOUT` | `120` | Seconds before an AI request times out. |
| `EXAMFORGE_MAX_QUESTIONS` | `50` | Most questions allowed in one paper. |
| `EXAMFORGE_MAX_MARKS_PER_QUESTION` | `100` | Most marks allowed for one question. |

Uploads are limited to 10 MB (`maxUploadSize` in [`.streamlit/config.toml`](.streamlit/config.toml)).

## Deployment

ExamForge ships with a [`Dockerfile`](Dockerfile). The image runs as a non-root user and includes a health check on `/_stcore/health`.

```bash
docker build -t examforge .
docker run -d -p 8501:8501 \
  -e EXAMFORGE_PASSWORD='choose-a-strong-password' \
  -e OPENAI_API_KEY='sk-...' \
  examforge
```

To use Ollama running on the host machine instead, pass `-e OLLAMA_BASE_URL=http://host.docker.internal:11434` (on Linux, also add `--add-host=host.docker.internal:host-gateway`).

**Before making the app public:**
- **Set `EXAMFORGE_PASSWORD`.** Without it, anyone who can reach the app can spend your OpenAI key. The app logs a warning at startup if a server key is set without a password.
- **Serve it over HTTPS**, behind a reverse proxy or a hosting platform that provides TLS, so the password isn't sent in plain text.
- **Keep the usage cap** (`EXAMFORGE_MAX_AI_REQUESTS_PER_HOUR`) sized to your budget, and set a spending limit in your OpenAI account as a backstop.

Logs go to standard output, in the form `timestamp LEVEL examforge: message` (for example failed sign-ins, usage-limit hits, and AI errors).

## Tech Stack
- **Frontend:** Streamlit
- **Backend Logic:** Python
- **PDF Extraction:** pdfplumber, pypdf
- **AI Integration:** OpenAI Python SDK (compatible with OpenAI & Ollama)
- **PDF Generation:** ReportLab
- **Data Handling:** Pandas

## Testing

Run the unit tests:
```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests
```

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the tests and builds and health-checks the Docker image on every push and pull request.

Tested using [TestGrid.io](https://testgrid.io)
