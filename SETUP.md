# NL2SQL with Vespa Schema Routing - Setup Guide

A production-ready NL2SQL system that uses Vespa for intelligent schema routing before LLM-based SQL generation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER QUESTION                                    │
│              "Find all singers from France"                              │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      VESPA ROUTING LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  166 databases → 876 tables → 4,503 columns indexed             │   │
│  │                                                                  │   │
│  │  Stage 1: Database Selection    → concert_singer (score: 2.4)   │   │
│  │  Stage 2: Table Selection       → singer, concert               │   │
│  │  Stage 3: Column Identification → name, country, age            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼ Narrowed Schema (3 tables vs 876)       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      LLM SQL GENERATION                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  OpenAI GPT-4o-mini                                              │   │
│  │  Input: Narrowed schema + question                               │   │
│  │  Output: SELECT * FROM singer WHERE country = 'France'           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      EXECUTION & VALIDATION                              │
│                      Spider Database API                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- **Docker** - For running Vespa
- **Python 3.11+** - For the NL2SQL pipeline
- **OpenAI API Key** - For SQL generation

---

## Option 1: Docker Compose (Recommended)

The easiest way to run the full stack.

### 1. Create `.env` file

```bash
cat > .env << EOF
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
EOF
```

### 2. Start Services

```bash
# Start Vespa and NL2SQL containers
docker-compose up -d

# Wait for Vespa to be healthy (~60 seconds)
docker-compose logs -f vespa
# Look for: "Application status: UP"
```

### 3. Deploy Schema and Ingest Data

```bash
# Deploy Vespa schema
docker exec nl2sql python deploy.py

# Ingest Spider dataset
docker exec nl2sql python spider_feed.py
```

### 4. Run Queries

```bash
# Interactive mode
docker exec -it nl2sql python nl2sql.py interactive

# Run benchmark
docker exec nl2sql python benchmark.py 50
```

### 5. Stop Services

```bash
# Stop containers (preserves data)
docker-compose stop

# Remove containers and data
docker-compose down -v
```

### Docker Commands Reference

```bash
# View logs
docker-compose logs -f

# Restart Vespa
docker-compose restart vespa

# Shell into NL2SQL container
docker exec -it nl2sql bash

# Check Vespa status
docker exec nl2sql curl http://vespa:19071/ApplicationStatus
```

---

## Option 2: Local Setup (Manual)

### 1. Clone and Setup Environment

```bash
cd vespa-ai-poc

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pyvespa numpy openai python-dotenv requests
```

### 2. Configure API Keys

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
```

### 3. Start Vespa

```bash
# Start Vespa container (use port 8090 if 8080 is busy)
docker run -d \
  --name vespa \
  --hostname vespa-container \
  -p 8090:8080 \
  -p 19071:19071 \
  vespaengine/vespa

# Wait for Vespa to start (~60 seconds)
sleep 60

# Verify Vespa is running
curl http://localhost:19071/ApplicationStatus
```

### 4. Deploy Schema and Ingest Data

```bash
# Deploy Vespa schema
python deploy.py

# Ingest Spider dataset (166 databases, 5,545 nodes)
python spider_feed.py

# Expected output:
# ✅ Success: 5545
# ❌ Failed: 0
```

### 5. Test the Pipeline

```bash
# Run demo queries
python nl2sql.py

# Interactive mode
python nl2sql.py interactive
```

### 6. Run Benchmark

```bash
# Quick test (50 questions)
python benchmark.py 50

# Full benchmark (1034 questions, ~50 minutes)
python benchmark.py --all
```

## File Structure

```
vespa-ai-poc/
├── .env                    # API keys (create this)
├── requirements.txt        # Python dependencies
├── venv/                   # Virtual environment
│
├── vespa_app.py            # Vespa schema definition
├── deploy.py               # Deploy schema to Vespa
├── spider_feed.py          # Ingest Spider schemas
│
├── nl2sql.py               # Main NL2SQL pipeline
├── query.py                # Vespa query utilities
│
├── benchmark.py            # Evaluation script
├── benchmark_results.json  # Results output
├── benchmark_full.log      # Full benchmark log
│
├── spider_data/            # Spider dataset
│   ├── tables.json         # Database schemas
│   ├── dev.json            # Dev questions (1034)
│   ├── dev_gold.sql        # Gold SQL queries
│   └── database/           # SQLite databases
│
├── README.md               # Architecture documentation
└── SETUP.md                # This file
```

## Usage Examples

### Basic Query

```python
from nl2sql import NL2SQLPipeline

pipeline = NL2SQLPipeline(llm_provider='openai')

result = pipeline.run(
    question="How many singers are from France?",
    target_db="concert_singer"  # Optional: let Vespa find DB
)

print(result["sql"])
# SELECT COUNT(*) FROM singer WHERE country = 'France'
```

### With Vespa Routing Only

```python
from nl2sql import VespaSchemaRouter

router = VespaSchemaRouter()
schema = router.route("Find all employees in engineering")

print(schema.database)       # hr_system
print(schema.tables)         # [employees, departments]
print(schema.to_ddl())       # CREATE TABLE statements
```

### Batch Evaluation

```python
from benchmark import run_benchmark, load_dev_data
from nl2sql import NL2SQLPipeline

pipeline = NL2SQLPipeline(llm_provider='openai')
questions = load_dev_data(limit=100)

results = run_benchmark(pipeline, questions)
# Results include: exact_match, execution_match, match_type
```

## Configuration Options

### Vespa Port

If port 8080 is busy, modify the port in these files:

```python
# In deploy.py, nl2sql.py, spider_feed.py, query.py
VESPA_PORT = 8090  # Change from 8080
```

### LLM Provider

```python
# OpenAI (default)
pipeline = NL2SQLPipeline(llm_provider='openai')

# Stub for testing (no API calls)
pipeline = NL2SQLPipeline(llm_provider='stub')

# Anthropic (requires ANTHROPIC_API_KEY)
pipeline = NL2SQLPipeline(llm_provider='anthropic')
```

### OpenAI Model

In `nl2sql.py`, change the model:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",  # Fast, cheap
    # model="gpt-4o",     # More accurate, slower
    ...
)
```

## Benchmark Results

Expected results on Spider dev set:

| Metric | GPT-4o-mini | Notes |
|--------|-------------|-------|
| Execution Match | ~75-80% | Correct results |
| Exact SQL Match | ~15-20% | Exact string match |
| Avg Latency | ~1.5s | Routing + LLM |

### Match Types

| Type | Description |
|------|-------------|
| `exact` | Same results returned |
| `distinct_diff` | Same unique values, different duplicates |
| `subset` | Generated query more restrictive |
| `superset` | Generated query less restrictive |
| `no_match` | Different results |

## Troubleshooting

### Vespa Connection Error

```bash
# Check if Vespa is running
docker ps | grep vespa

# Check logs
docker logs vespa

# Restart if needed
docker restart vespa
```

### Port Already in Use

```bash
# Find what's using port 8080
lsof -i :8080

# Use different port
docker run -d --name vespa -p 8090:8080 -p 19071:19071 vespaengine/vespa
```

### OpenAI Rate Limits

The benchmark includes a 0.5s delay between requests. For faster runs:

```python
# In benchmark.py, reduce sleep time (may hit rate limits)
time.sleep(0.2)  # Instead of 0.5
```

### Missing Database in Vespa

```bash
# Re-ingest Spider data
python spider_feed.py

# Check ingestion count
# Should show: ✅ Success: 5545
```

## API Reference

### Spider Execution API

The benchmark uses an external API to execute SQL:

```bash
curl -X POST "http://54.226.193.202:8000/api/v1/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "service": "spider1",
    "action": "execute_sql",
    "params": {
      "db_id": "concert_singer",
      "sql": "SELECT count(*) FROM singer"
    }
  }'
```

Response:
```json
{
  "success": true,
  "data": {
    "columns": ["count(*)"],
    "rows": [[6]],
    "row_count": 1
  }
}
```

## Next Steps

1. **Real Embeddings**: Replace random embeddings with BGE M3 for better routing
2. **Fine-tuning**: Improve prompts based on error analysis
3. **Caching**: Add embedding cache for faster queries
4. **Multi-turn**: Support follow-up questions with context

## License

MIT
