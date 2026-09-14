# 🎓 RAG-Based AI Teaching Assistant

An intelligent **Retrieval-Augmented Generation (RAG)** system that turns video course lectures into a searchable, AI-powered teaching assistant. Ask questions about your course content and get precise answers pointing to the exact video and timestamp.

---

## 📖 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Step 1 – Collect Videos](#step-1--collect-videos)
  - [Step 2 – Extract Audio](#step-2--extract-audio)
  - [Step 3 – Transcribe Audio to JSON](#step-3--transcribe-audio-to-json)
  - [Step 4 – Merge Chunks](#step-4--merge-chunks)
  - [Step 5 – Generate Embeddings](#step-5--generate-embeddings)
  - [Step 6 – Ask Questions](#step-6--ask-questions)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

When learning from long video courses, it's hard to find *where* a specific topic was taught. This project solves that problem by:

1. **Transcribing** all course videos (supports Hindi → English translation)
2. **Chunking & embedding** the transcriptions into vector representations
3. **Retrieving** the most relevant chunks for any user question using cosine similarity
4. **Generating** a natural-language answer via an LLM, pointing you to the exact video and timestamp

---

## How It Works

```
 ┌──────────┐     ┌──────────┐     ┌───────────┐     ┌────────────┐
 │  Videos  │────▶│  Audio   │────▶│ Whisper   │────▶│ JSON       │
 │ (.mp4)   │     │ (.mp3)   │     │ Transcribe│     │ Subtitles  │
 └──────────┘     └──────────┘     └───────────┘     └─────┬──────┘
                                                           │
                                                           ▼
                                                   ┌──────────────┐
                                                   │ Merge Chunks │
                                                   │ (groups of 5)│
                                                   └─────┬────────┘
                                                         │
                                                         ▼
 ┌──────────────┐     ┌──────────────┐     ┌─────────────────────┐
 │ User Query   │────▶│ BGE-M3       │────▶│ Cosine Similarity   │
 │              │     │ Embedding    │     │ Search (Top 5)      │
 └──────────────┘     └──────────────┘     └──────────┬──────────┘
                                                      │
                                                      ▼
                                              ┌───────────────┐
                                              │ LLM (Groq)    │
                                              │ Generate      │
                                              │ Answer        │
                                              └───────────────┘
```

---

## Architecture

| Stage | Script | Description |
|-------|--------|-------------|
| **1. Video → Audio** | `video_to_mp3.py` | Extracts audio tracks from video files using FFmpeg |
| **2. Audio → Text** | `mp3_to_json.py` | Transcribes audio using OpenAI Whisper (`large-v2`), translating Hindi to English |
| **3. Merge Chunks** | `merge_chunks.py` | Combines every 5 subtitle segments into larger chunks for better context |
| **4. Text → Vectors** | `preprocessed_json.py` | Generates vector embeddings using Ollama's BGE-M3 model, saves as `.joblib` |
| **5. Query & Answer** | `process_incoming.py` | Embeds the user query, finds top-5 similar chunks, sends to LLM for answer |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Speech-to-Text** | [OpenAI Whisper](https://github.com/openai/whisper) (large-v2) |
| **Embeddings** | [BGE-M3](https://huggingface.co/BAAI/bge-m3) via [Ollama](https://ollama.com/) |
| **Vector Search** | Cosine similarity with scikit-learn |
| **LLM Inference** | [Groq API](https://groq.com/) |
| **Audio Extraction** | [FFmpeg](https://ffmpeg.org/) |
| **Data Storage** | Pandas DataFrame serialized with joblib |

---

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9+**
- **FFmpeg** – for video-to-audio conversion
  ```bash
  # Windows (via Chocolatey)
  choco install ffmpeg

  # macOS
  brew install ffmpeg

  # Ubuntu/Debian
  sudo apt install ffmpeg
  ```
- **Ollama** – for running the embedding model locally
  ```bash
  # Install from https://ollama.com/
  # Then pull the embedding model:
  ollama pull bge-m3
  ```
- **Groq API Key** – sign up at [console.groq.com](https://console.groq.com/) to get a free API key

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Gouravkumar532/RAG-based-ai-teaching-assistant.git
   cd RAG-based-ai-teaching-assistant
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirments.txt
   ```

4. **Install Whisper** (required for transcription)
   ```bash
   pip install openai-whisper
   ```

5. **Set up environment variables**

   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

6. **Start Ollama** (must be running for embeddings)
   ```bash
   ollama serve
   ```

---

## Usage

### Step 1 – Collect Videos

Place all your course video files (`.mp4`, `.mkv`, etc.) into the `videos/` folder.

```
videos/
├── Sigma Web Dev Tutorial #1 Installing VS Code - 1080p.mp4
├── Sigma Web Dev Tutorial #2 Your First HTML Website - 1080p.mp4
├── Sigma Web Dev Tutorial #15 Inline, Internal & External CSS - 1080p.mp4
└── ...
```

> **Note:** Video filenames must follow the pattern `... Tutorial #<number> ... - ...` for the script to extract the tutorial number correctly.

---

### Step 2 – Extract Audio

Convert all videos to MP3 audio files:

```bash
python video_to_mp3.py
```

**What it does:** Uses FFmpeg to extract the audio track from each video and saves it as `audios/<number>_<title>.mp3`.

**Output example:**
```
1 Sigma Web Dev Tutorial #1
2 Sigma Web Dev Tutorial #2
15 Sigma Web Dev Tutorial #15
```

---

### Step 3 – Transcribe Audio to JSON

Transcribe each audio file into timestamped subtitle chunks:

```bash
python mp3_to_json.py
```

**What it does:**
- Loads OpenAI Whisper `large-v2` model
- Transcribes Hindi audio and translates it to English
- Saves each transcription as a JSON file with timestamped segments

**Output JSON structure:**
```json
{
  "chunks": [
    {
      "number": "1",
      "title": "Installing VS Code & How Websites Work",
      "start": 0.0,
      "end": 3.54,
      "text": "From today's video, we will start the Sigma Web Development course."
    },
    {
      "number": "1",
      "title": "Installing VS Code & How Websites Work",
      "start": 3.54,
      "end": 8.12,
      "text": "So first of all, let me tell you what is web development."
    }
  ],
  "text": "Full transcription text..."
}
```

> **⏱️ Note:** This step can take a long time depending on the number and length of videos. The Whisper `large-v2` model requires a GPU for reasonable speed.

---

### Step 4 – Merge Chunks

Combine small subtitle segments into larger, more meaningful chunks:

```bash
python merge_chunks.py
```

**What it does:** Groups every 5 consecutive subtitle segments into a single chunk, preserving the start/end timestamps of the group. This improves retrieval quality by giving each chunk more context.

**Before merging:** Each chunk is ~3-5 seconds of speech (~10-20 words)
**After merging:** Each chunk is ~15-25 seconds of speech (~50-100 words)

---

### Step 5 – Generate Embeddings

Convert all text chunks into vector embeddings:

```bash
python preprocessed_json.py
```

**What it does:**
- Reads all merged JSON files from `newjsons/`
- Sends text to Ollama's BGE-M3 model in batches of 200
- Creates a Pandas DataFrame with columns: `number`, `title`, `start`, `end`, `text`, `chunk_id`, `embedding`
- Saves everything as `embeddings.joblib`

> **⚠️ Requirement:** Ollama must be running (`ollama serve`) with the `bge-m3` model pulled.

---

### Step 6 – Ask Questions

Query the teaching assistant:

```bash
python process_incoming.py
```

**What it does:**
1. Loads the pre-computed embeddings from `embeddings.joblib`
2. Prompts you to ask a question
3. Embeds your question using BGE-M3
4. Finds the top 5 most similar chunks using cosine similarity
5. Constructs a prompt with the relevant chunks and sends it to the LLM
6. Returns a natural-language answer pointing you to the right video and timestamp

**Example interaction:**
```
Ask a question : How do websites work?

Great question! The concept of how websites work is covered in
Video #1 - "Installing VS Code & How Websites Work" starting around
the 11:34 mark (694 seconds). In this section, the instructor explains
the client-server model — the browser (like Chrome) acts as the client,
and it communicates with a server to fetch web pages. You'll learn about
how the browser requests pages and the server responds with the content.

I'd recommend watching from timestamp 11:30 to about 13:00 in Video #1
for a complete understanding of client-server architecture!
```

---

## Project Structure

```
RAG-based-ai-teaching-assistant/
│
├── video_to_mp3.py          # Step 1: Extract audio from videos (FFmpeg)
├── mp3_to_json.py           # Step 2: Transcribe audio → JSON (Whisper)
├── merge_chunks.py          # Step 3: Merge small chunks into larger ones
├── preprocessed_json.py     # Step 4: Generate embeddings (Ollama BGE-M3)
├── process_incoming.py      # Step 5: Query interface & LLM response
│
├── videos/                  # Input: Place course videos here
├── audios/                  # Generated: Extracted MP3 files
├── jsons/                   # Generated: Raw transcription JSONs
├── newjsons/                # Generated: Merged chunk JSONs
├── embeddings.joblib        # Generated: DataFrame with embeddings
│
├── prompt.txt               # Generated: Last prompt sent to LLM
├── response.txt             # Generated: Last LLM response
│
├── requirments.txt          # Python dependencies
├── .env                     # API keys (not tracked in git)
├── .gitignore               # Git ignore rules
└── readme.md                # This file
```

---

## Configuration

| Variable | Where | Description |
|----------|-------|-------------|
| `GROQ_API_KEY` | `.env` | Your Groq API key for LLM inference |
| `bge-m3` | `preprocessed_json.py`, `process_incoming.py` | Ollama embedding model (change if using a different model) |
| `large-v2` | `mp3_to_json.py` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v2`) |
| `n = 5` | `merge_chunks.py` | Number of segments to merge per chunk |
| `top_Results = 5` | `process_incoming.py` | Number of similar chunks to retrieve |
| `BATCH_SIZE = 200` | `preprocessed_json.py` | Batch size for embedding generation |

---

## Example

Here's a complete end-to-end example using the [Sigma Web Development Course](https://www.youtube.com/playlist?list=PLu0W_9lII9agq5TBjMJPsEB_Xhl9TM-r7):

```bash
# 1. Place all downloaded Sigma Web Dev videos in videos/

# 2. Extract audio
python video_to_mp3.py

# 3. Transcribe (this will take a while!)
python mp3_to_json.py

# 4. Merge chunks for better context
python merge_chunks.py

# 5. Generate embeddings (make sure Ollama is running)
python preprocessed_json.py

# 6. Start asking questions!
python process_incoming.py
```

**Sample questions you can ask:**
- *"How to add CSS to an HTML page?"*
- *"What is flexbox and how to use it?"*
- *"How do websites work?"*
- *"What is the difference between inline and block elements?"*
- *"How to create a form in HTML?"*

---

## Contributing

Contributions are welcome! Here are some ideas for improvement:

- [ ] Add a web-based UI (Flask/Streamlit)
- [ ] Support multiple languages beyond Hindi
- [ ] Add support for YouTube URL input (auto-download)
- [ ] Implement a persistent vector database (e.g., ChromaDB, FAISS)
- [ ] Add chat history / multi-turn conversations
- [ ] Dockerize the entire pipeline

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/Gouravkumar532">Gourav Kumar</a>
</p>