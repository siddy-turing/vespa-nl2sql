# Semantic Router for Data Catalog

A multi-stage semantic routing system built on Vespa for navigating enterprise data catalogs using natural language queries.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                          │
│                "show me all stock trades from yesterday"                    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VESPA LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  BGE M3 Embedding Model                                              │   │
│  │  • 1024 dimensions                                                   │   │
│  │  • 8K token context length                                           │   │
│  │  • Semantic search runs on generated descriptions only               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Hybrid Ranking                                                      │   │
│  │  • Semantic: closeness(embedding) * 0.6                              │   │
│  │  • Lexical:  bm25(aliases) + bm25(name) * 0.4                        │   │
│  │  • Aliases extracted from historical SQL queries                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    4-STAGE ROUTING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Stage 1              Stage 2              Stage 3              Stage 4    │
│  ┌──────────┐        ┌──────────┐        ┌──────────┐        ┌──────────┐  │
│  │  DATA    │   →    │  ROOT    │   →    │  ENTITY  │   →    │ PROPERTY │  │
│  │ PRODUCT  │        │  ENTITY  │        │          │        │          │  │
│  └──────────┘        └──────────┘        └──────────┘        └──────────┘  │
│   filter:            filter:              filter:              filter:      │
│   node_type=         data_product_id=     parent_id=           parent_id=  │
│   data_product       [selected]           [selected]           [selected]  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RESULT                                              │
│   Path: Trading Platform → Equities → Trades                                │
│   Properties: symbol, trade_date, price, quantity, side                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Model

The system uses a **tree structure** (not graph) to avoid cycles. Each node has a single parent and multiple children.

```
                              ┌─────────────────┐
                              │  Data Product   │  ← Cluster (SQL equivalent)
                              │  (Trading)      │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                      │
             ┌──────┴──────┐                       ┌──────┴──────┐
             │ Root Entity │  ← Database           │ Root Entity │
             │ (Equities)  │                       │(Fixed Income│
             └──────┬──────┘                       └─────────────┘
                    │
         ┌─────────┴─────────┐
         │                   │
    ┌────┴────┐        ┌────┴────┐
    │ Entity  │  ←     │ Entity  │  ← Tables
    │(Trades) │  Table │(Positions│
    └────┬────┘        └─────────┘
         │
    ┌────┴────┬────────┬────────┬────────┐
    │         │        │        │        │
┌───┴───┐ ┌───┴──┐ ┌───┴──┐ ┌───┴──┐ ┌───┴───┐
│symbol │ │ side │ │price │ │ qty  │ │ date  │  ← Columns
└───────┘ └──────┘ └──────┘ └──────┘ └───────┘
```

### Node Types

| Level | Node Type | SQL Equivalent | Example |
|-------|-----------|----------------|---------|
| 1 | `data_product` | Cluster | Trading Platform |
| 2 | `root_entity` | Database | Equities |
| 3 | `entity` | Table | Trades |
| 4 | `property` | Column | symbol, price, side |

## Schema Design

Each node stores:

```yaml
Core Fields:
  id: "p_symbol"                    # Unique identifier
  node_type: "property"             # data_product | root_entity | entity | property
  name: "symbol"                    # Display name
  
Semantic Search:
  description: "Stock ticker..."    # LLM-generated, 25-30 words
  embedding: [0.1, 0.2, ...]        # 1024-dim BGE M3 vector
  
Lexical Search (BM25):
  aliases: ["ticker", "stock symbol", "security id"]  # From historical queries
  
Tree Structure:
  parent_id: "e_trades"             # Single parent
  children_ids: []                  # Multiple children
  data_product_id: "dp_trading"     # For filtering
  root_entity_id: "re_equities"     # For filtering
  
Property Metadata:
  data_type: "string"               # string | integer | decimal | datetime | enum
  enum_values: ["BUY", "SELL"]      # For enum types
  
Versioning:
  version_timestamp: 1706886400     # For 24-hour refresh cycle
```

## Hybrid Ranking

The system combines two ranking signals:

### 1. Semantic Search (60% weight)
- Runs on `description` field embeddings
- Uses HNSW index for approximate nearest neighbor
- Captures meaning: "stock" matches "equity"

### 2. Lexical Search (40% weight)  
- BM25 on `aliases` extracted from historical SQL queries
- Exact keyword matching: "ticker" → symbol
- Handles domain-specific terminology

```
final_score = closeness(embedding) * 0.6 + bm25(aliases, name) * 0.4
```

### Ranking Profiles

| Profile | Use Case | Formula |
|---------|----------|---------|
| `semantic` | Pure vector search | `closeness(embedding)` |
| `lexical` | Pure keyword search | `bm25(aliases) + bm25(name)` |
| `hybrid` | Default - balanced | `semantic * 0.6 + lexical * 0.4` |
| `alias_heavy` | When aliases matter more | `semantic * 0.3 + aliases * 0.5 + name * 0.2` |

## Multi-Stage Routing Process

```python
def route_query(query: str) -> RoutingResult:
    """
    Sequential Vespa calls with filtering at each level.
    """
    
    # Stage 1: Find Data Product
    # Filter: node_type = "data_product"
    data_product = search(query, filter="node_type:data_product")[0]
    
    # Stage 2: Find Root Entity within Data Product
    # Filter: node_type = "root_entity" AND data_product_id = selected
    root_entity = search(query, filter=f"data_product_id:{data_product.id}")[0]
    
    # Stage 3: Find Entity within Root Entity
    # Filter: node_type = "entity" AND parent_id = selected
    entity = search(query, filter=f"parent_id:{root_entity.id}")[0]
    
    # Stage 4: Find Properties within Entity
    # Filter: node_type = "property" AND parent_id = selected
    properties = search(query, filter=f"parent_id:{entity.id}")[:5]
    
    return RoutingResult(
        path=[data_product, root_entity, entity],
        properties=properties
    )
```

## Layer Input/Output Specification

### Complete Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              LAYER 0: INPUT                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                      │
│    • user_query: "show me all stock trades from yesterday"                   │
│                                                                              │
│  PROCESSING:                                                                 │
│    • Generate 1024-dim embedding from query using BGE M3                     │
│    • Tokenize query for BM25 lexical matching                                │
│                                                                              │
│  OUTPUT:                                                                     │
│    • query_embedding: float[1024]                                            │
│    • query_tokens: ["show", "me", "all", "stock", "trades", "yesterday"]     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      LAYER 1: DATA PRODUCT SELECTION                         │
├──────────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                      │
│    • query_embedding: float[1024]                                            │
│    • query_tokens: ["show", "stock", "trades", ...]                          │
│    • filter: node_type = "data_product"                                      │
│                                                                              │
│  VESPA QUERY (YQL):                                                          │
│    SELECT * FROM metadata                                                    │
│    WHERE node_type = "data_product"                                          │
│      AND (nearestNeighbor(embedding, q) OR userQuery())                      │
│    RANKING: hybrid                                                           │
│                                                                              │
│  CANDIDATES SEARCHED: All data_product nodes (3 in sample)                   │
│                                                                              │
│  OUTPUT:                                                                     │
│    • selected_data_product: {                                                │
│        id: "dp_trading",                                                     │
│        name: "Trading Platform",                                             │
│        score: 1.636,                                                         │
│        description: "Enterprise trading platform...",                        │
│        aliases: ["trades", "trading system", "order book"]                   │
│      }                                                                       │
│    • alternatives: [Revenue Analytics (1.05), HR System (0.35)]              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      LAYER 2: ROOT ENTITY SELECTION                          │
├──────────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                      │
│    • query_embedding: float[1024]                                            │
│    • query_tokens: ["show", "stock", "trades", ...]                          │
│    • filter: node_type = "root_entity"                                       │
│              AND data_product_id = "dp_trading"  ← from Layer 1              │
│                                                                              │
│  VESPA QUERY (YQL):                                                          │
│    SELECT * FROM metadata                                                    │
│    WHERE node_type = "root_entity"                                           │
│      AND data_product_id = "dp_trading"                                      │
│      AND (nearestNeighbor(embedding, q) OR userQuery())                      │
│    RANKING: hybrid                                                           │
│                                                                              │
│  CANDIDATES SEARCHED: Root entities under Trading Platform (1 in sample)    │
│                                                                              │
│  OUTPUT:                                                                     │
│    • selected_root_entity: {                                                 │
│        id: "re_equities",                                                    │
│        name: "Equities",                                                     │
│        score: 2.387,                                                         │
│        description: "Equities trading database containing stock trades...", │
│        aliases: ["stocks", "equity trades", "share transactions"]            │
│      }                                                                       │
│    • alternatives: []  (only 1 root entity in this data product)             │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        LAYER 3: ENTITY SELECTION                             │
├──────────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                      │
│    • query_embedding: float[1024]                                            │
│    • query_tokens: ["show", "stock", "trades", ...]                          │
│    • filter: node_type = "entity"                                            │
│              AND parent_id = "re_equities"  ← from Layer 2                   │
│                                                                              │
│  VESPA QUERY (YQL):                                                          │
│    SELECT * FROM metadata                                                    │
│    WHERE node_type = "entity"                                                │
│      AND parent_id = "re_equities"                                           │
│      AND (nearestNeighbor(embedding, q) OR userQuery())                      │
│    RANKING: hybrid                                                           │
│                                                                              │
│  CANDIDATES SEARCHED: Entities under Equities (2 in sample)                  │
│                                                                              │
│  OUTPUT:                                                                     │
│    • selected_entity: {                                                      │
│        id: "e_trades",                                                       │
│        name: "Trades",                                                       │
│        score: 1.208,                                                         │
│        description: "Trade execution records containing buy/sell orders...",│
│        aliases: ["executions", "fills", "trade records", "transactions"]     │
│      }                                                                       │
│    • alternatives: [Positions (0.35)]                                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      LAYER 4: PROPERTY IDENTIFICATION                        │
├──────────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                      │
│    • query_embedding: float[1024]                                            │
│    • query_tokens: ["show", "stock", "trades", ...]                          │
│    • filter: node_type = "property"                                          │
│              AND parent_id = "e_trades"  ← from Layer 3                      │
│                                                                              │
│  VESPA QUERY (YQL):                                                          │
│    SELECT * FROM metadata                                                    │
│    WHERE node_type = "property"                                              │
│      AND parent_id = "e_trades"                                              │
│      AND (nearestNeighbor(embedding, q) OR userQuery())                      │
│    RANKING: hybrid                                                           │
│                                                                              │
│  CANDIDATES SEARCHED: Properties under Trades (6 in sample)                  │
│                                                                              │
│  OUTPUT:                                                                     │
│    • selected_properties: [                                                  │
│        {id: "p_symbol", name: "symbol", score: 1.824, data_type: "string",  │
│         aliases: ["ticker", "stock symbol", "security id"]},                 │
│        {id: "p_trade_date", name: "trade_date", score: 0.350,               │
│         data_type: "datetime", aliases: ["execution time", "timestamp"]},    │
│        {id: "p_price", name: "price", score: 0.349, data_type: "decimal",   │
│         aliases: ["exec price", "fill price"]},                              │
│        {id: "p_quantity", name: "quantity", score: 0.349,                   │
│         data_type: "integer", aliases: ["shares", "volume", "size"]},        │
│        {id: "p_side", name: "side", score: 0.346, data_type: "enum",        │
│         aliases: ["direction", "buy sell"], enum_values: ["BUY", "SELL"]}    │
│      ]                                                                       │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 5: FINAL OUTPUT                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                      │
│    • Layer 1 output: data_product                                            │
│    • Layer 2 output: root_entity                                             │
│    • Layer 3 output: entity                                                  │
│    • Layer 4 output: properties[]                                            │
│                                                                              │
│  OUTPUT:                                                                     │
│    {                                                                         │
│      "query": "show me all stock trades from yesterday",                     │
│      "routing_path": [                                                       │
│        "Trading Platform",   // data_product                                 │
│        "Equities",           // root_entity (database)                       │
│        "Trades"              // entity (table)                               │
│      ],                                                                      │
│      "selected_table": {                                                     │
│        "data_product": "dp_trading",                                         │
│        "database": "re_equities",                                            │
│        "table": "e_trades"                                                   │
│      },                                                                      │
│      "relevant_columns": [                                                   │
│        {"name": "symbol", "type": "string"},                                 │
│        {"name": "trade_date", "type": "datetime"},                           │
│        {"name": "price", "type": "decimal"},                                 │
│        {"name": "quantity", "type": "integer"},                              │
│        {"name": "side", "type": "enum", "values": ["BUY", "SELL"]}           │
│      ],                                                                      │
│      "confidence_scores": {                                                  │
│        "data_product": 1.636,                                                │
│        "root_entity": 2.387,                                                 │
│        "entity": 1.208,                                                      │
│        "top_property": 1.824                                                 │
│      }                                                                       │
│    }                                                                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Layer Summary Table

| Layer | Input | Filter Applied | Output | Cardinality |
|-------|-------|----------------|--------|-------------|
| **0: Embedding** | `user_query: str` | None | `embedding: float[1024]`, `tokens: str[]` | 1 → 1 |
| **1: Data Product** | `embedding`, `tokens` | `node_type = "data_product"` | `selected_dp: Node`, `score: float` | 1 → 1 |
| **2: Root Entity** | `embedding`, `tokens`, `dp_id` | `data_product_id = {dp_id}` | `selected_re: Node`, `score: float` | 1 → 1 |
| **3: Entity** | `embedding`, `tokens`, `re_id` | `parent_id = {re_id}` | `selected_entity: Node`, `score: float` | 1 → 1 |
| **4: Properties** | `embedding`, `tokens`, `entity_id` | `parent_id = {entity_id}` | `properties: Node[]`, `scores: float[]` | 1 → N |
| **5: Output** | All layer outputs | None | `RoutingResult` | N → 1 |

### Data Types

```typescript
// Layer 0 Output
interface EmbeddingOutput {
  query_embedding: number[];      // 1024-dim float array
  query_tokens: string[];         // Tokenized query for BM25
}

// Layers 1-4 Output (per node)
interface NodeResult {
  id: string;                     // "dp_trading", "e_trades", etc.
  node_type: string;              // "data_product" | "root_entity" | "entity" | "property"
  name: string;                   // Human-readable name
  description: string;            // LLM-generated description
  aliases: string[];              // Historical query aliases
  score: number;                  // Hybrid ranking score
  data_type?: string;             // For properties: "string" | "integer" | "decimal" | "datetime" | "enum"
  enum_values?: string[];         // For enum properties
}

// Layer 5 Final Output
interface RoutingResult {
  query: string;                  // Original user query
  routing_path: string[];         // ["Trading Platform", "Equities", "Trades"]
  selected_table: {
    data_product: string;
    database: string;
    table: string;
  };
  relevant_columns: {
    name: string;
    type: string;
    values?: string[];            // For enums
  }[];
  confidence_scores: {
    data_product: number;
    root_entity: number;
    entity: number;
    top_property: number;
  };
}
```

### Score Interpretation

| Score Range | Confidence | Action |
|-------------|------------|--------|
| > 2.0 | **High** | Strong match, proceed confidently |
| 1.0 - 2.0 | **Medium** | Good match, may need verification |
| 0.5 - 1.0 | **Low** | Weak match, consider alternatives |
| < 0.5 | **Very Low** | Poor match, trigger disambiguation |

When top-2 scores are within 0.1 of each other → trigger **clarifying question**

## Example Routing

**Query:** `"what is the employee kerberos login"`

| Stage | Filter | Top Result | Score | Why |
|-------|--------|------------|-------|-----|
| 1. Data Product | `node_type=data_product` | HR System | 0.98 | "employee" in description |
| 2. Root Entity | `data_product_id=dp_hr` | Employees | 1.09 | "employee" alias match |
| 3. Entity | `parent_id=re_employees` | Employee Records | 0.99 | "employee" in name |
| 4. Properties | `parent_id=e_employee_records` | **kerberos_id** | **3.37** | "kerberos", "login" alias match |

The `kerberos_id` property scores highest because:
- Aliases include: `["login", "username", "network id", "kerb"]`
- BM25 matches "kerberos" and "login" exactly

## File Structure

```
vespa-ai-poc/
├── vespa_app.py      # Schema definition (1024-dim, hybrid ranking)
├── deploy.py         # Deploy to Vespa container
├── feed.py           # Sample data: 3 Data Products, 18 Properties
├── query.py          # SemanticRouter class with 4-stage pipeline
├── requirements.txt  # pyvespa, numpy
└── README.md         # This file
```

## Usage

### Start Vespa
```bash
docker run -d --name vespa \
  -p 8090:8080 -p 19071:19071 \
  vespaengine/vespa
```

### Deploy & Feed
```bash
source venv/bin/activate
python deploy.py   # Deploy schema
python feed.py     # Feed sample data
```

### Run Queries
```bash
python query.py    # Demo queries

# Or interactive mode:
python -c "from query import interactive_mode; interactive_mode()"
```

### Programmatic Usage
```python
from query import SemanticRouter

router = SemanticRouter()
result = router.route_query("show me stock trades")

print(result["path"])        # ['Trading Platform', 'Equities', 'Trades']
print(result["properties"])  # [symbol, price, quantity, ...]
```

## Key Design Decisions

### 1. Tree vs Graph
- **Decision:** Tree structure with single parent
- **Reason:** Avoids cycles, simpler traversal, clearer routing path
- **Trade-off:** Properties shared across entities are duplicated as separate nodes

### 2. 24-Hour Refresh Cycle
- **Decision:** Version timestamp on each node
- **Reason:** Prevents drift from legacy aliases
- **Implementation:** `version_timestamp` field for tracking

### 3. Aliases from Historical Queries
- **Decision:** Extract aliases from `SELECT ... AS` statements
- **Reason:** Captures how users actually refer to columns
- **Risk:** Bias toward legacy terms (mitigated by refresh cycle)

### 4. Hybrid over Pure Semantic
- **Decision:** 60/40 semantic/lexical split
- **Reason:** Domain terms need exact matching; semantics alone miss "kerb" → "kerberos"

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Stages per query | 4 | Sequential Vespa calls |
| Latency (simple) | ~20-40s | As noted in meeting |
| Accuracy (DP/RE) | 85-90% | Current production metrics |
| Embedding dims | 1024 | BGE M3 |
| Refresh cycle | 24 hours | Prevents alias drift |

## Next Steps

### Planned Improvements
1. **Disambiguation Layer**
   - Pre-search: Extract business context (employee IDs, Kerberos)
   - Post-search: Clarifying questions when scores are close

2. **Historical Query Matching**
   - Threshold-based similarity to past successful queries
   - Boost confidence when pattern matches

3. **Real Embeddings**
   ```python
   from FlagEmbedding import BGEM3FlagModel
   model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
   ```

4. **Scaling Considerations**
   - Current: 5 data products → 15+ in 3 months
   - Challenge: Denser vector space, cross-product disambiguation

## References

- [Vespa Documentation](https://docs.vespa.ai/)
- [BGE M3 Model](https://huggingface.co/BAAI/bge-m3)
- [Hybrid Search in Vespa](https://docs.vespa.ai/en/nearest-neighbor-search.html)
