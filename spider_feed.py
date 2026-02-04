"""
Spider Dataset → Vespa Ingestion

Parses Spider tables.json and converts database schemas to Vespa nodes.
Hierarchy: Database (data_product) → Table (entity) → Column (property)

Spider schema format:
- db_id: database name
- table_names: list of tables
- column_names: [[table_idx, col_name], ...]
- column_types: ["text", "number", ...]
- foreign_keys: [[col_idx, col_idx], ...]
- primary_keys: [col_idx, ...]
"""

import json
import numpy as np
import time
from pathlib import Path
from vespa.application import Vespa
from vespa.io import VespaResponse
from typing import Optional

VESPA_PORT = 8090
SPIDER_DATA_PATH = Path("/Users/siddy/Desktop/vespa-ai-poc/spider_data")


def embed(text: str) -> list[float]:
    """
    Generate 1024-dim embedding for text.
    TODO: Replace with BGE M3 in production.
    """
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(1024).tolist()


def generate_description(node_type: str, name: str, context: dict) -> str:
    """
    Generate a natural language description for schema elements.
    In production, use LLM to generate 25-30 word descriptions.
    """
    if node_type == "data_product":
        tables = context.get("tables", [])
        return f"Database '{name}' containing {len(tables)} tables: {', '.join(tables[:5])}{'...' if len(tables) > 5 else ''}. Used for querying {name.replace('_', ' ')} related data."
    
    elif node_type == "entity":
        columns = context.get("columns", [])
        db_name = context.get("db_name", "")
        return f"Table '{name}' in {db_name} database with {len(columns)} columns including {', '.join(columns[:4])}{'...' if len(columns) > 4 else ''}."
    
    elif node_type == "property":
        table_name = context.get("table_name", "")
        col_type = context.get("col_type", "text")
        is_pk = context.get("is_pk", False)
        is_fk = context.get("is_fk", False)
        
        desc = f"Column '{name}' of type {col_type} in {table_name} table"
        if is_pk:
            desc += ", primary key"
        if is_fk:
            desc += ", foreign key"
        return desc + "."
    
    return f"{node_type}: {name}"


def generate_aliases(name: str, node_type: str) -> list[str]:
    """
    Generate aliases for schema elements.
    These would come from historical queries in production.
    """
    aliases = [name.lower()]
    
    # Add variations
    if "_" in name:
        aliases.append(name.replace("_", " ").lower())
        aliases.append(name.replace("_", "").lower())
    
    # Common abbreviations
    abbrevs = {
        "id": ["identifier", "key"],
        "name": ["title", "label"],
        "date": ["time", "timestamp", "when"],
        "count": ["number", "total", "qty"],
        "amount": ["value", "sum", "total"],
        "price": ["cost", "rate"],
        "description": ["desc", "details", "info"],
    }
    
    name_lower = name.lower()
    for key, alias_list in abbrevs.items():
        if key in name_lower:
            aliases.extend(alias_list)
    
    return list(set(aliases))


def parse_spider_schema(schema: dict) -> list[dict]:
    """
    Parse a Spider database schema into Vespa nodes.
    
    Returns list of nodes:
    - 1 database node (data_product)
    - N table nodes (entity)  
    - M column nodes (property)
    """
    nodes = []
    db_id = schema["db_id"]
    
    # Build lookup structures
    table_names = schema.get("table_names_original", schema["table_names"])
    column_names = schema.get("column_names_original", schema["column_names"])
    column_types = schema["column_types"]
    primary_keys = set(schema.get("primary_keys", []))
    foreign_keys = {fk[0] for fk in schema.get("foreign_keys", [])}
    
    # Group columns by table
    table_columns = {i: [] for i in range(len(table_names))}
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx >= 0:  # Skip the "*" column at index -1
            table_columns[table_idx].append({
                "idx": col_idx,
                "name": col_name,
                "type": column_types[col_idx] if col_idx < len(column_types) else "text",
                "is_pk": col_idx in primary_keys,
                "is_fk": col_idx in foreign_keys,
            })
    
    # Create database node (data_product level)
    db_node = {
        "id": f"db_{db_id}",
        "node_type": "data_product",
        "name": db_id,
        "description": generate_description("data_product", db_id, {"tables": table_names}),
        "aliases": generate_aliases(db_id, "data_product"),
        "parent_id": "",
        "children_ids": [f"tbl_{db_id}_{t}" for t in table_names],
        "data_product_id": f"db_{db_id}",
        "root_entity_id": "",
        "data_type": "",
        "enum_values": [],
    }
    nodes.append(db_node)
    
    # Create table nodes (entity level)
    for table_idx, table_name in enumerate(table_names):
        columns = table_columns.get(table_idx, [])
        table_node = {
            "id": f"tbl_{db_id}_{table_name}",
            "node_type": "entity",
            "name": table_name,
            "description": generate_description("entity", table_name, {
                "columns": [c["name"] for c in columns],
                "db_name": db_id,
            }),
            "aliases": generate_aliases(table_name, "entity"),
            "parent_id": f"db_{db_id}",
            "children_ids": [f"col_{db_id}_{table_name}_{c['name']}" for c in columns],
            "data_product_id": f"db_{db_id}",
            "root_entity_id": f"tbl_{db_id}_{table_name}",
            "data_type": "",
            "enum_values": [],
        }
        nodes.append(table_node)
        
        # Create column nodes (property level)
        for col in columns:
            col_node = {
                "id": f"col_{db_id}_{table_name}_{col['name']}",
                "node_type": "property",
                "name": col["name"],
                "description": generate_description("property", col["name"], {
                    "table_name": table_name,
                    "col_type": col["type"],
                    "is_pk": col["is_pk"],
                    "is_fk": col["is_fk"],
                }),
                "aliases": generate_aliases(col["name"], "property"),
                "parent_id": f"tbl_{db_id}_{table_name}",
                "children_ids": [],
                "data_product_id": f"db_{db_id}",
                "root_entity_id": f"tbl_{db_id}_{table_name}",
                "data_type": col["type"],
                "enum_values": [],
                "is_primary_key": col["is_pk"],
                "is_foreign_key": col["is_fk"],
            }
            nodes.append(col_node)
    
    return nodes


def load_spider_schemas(tables_file: Path, limit: Optional[int] = None) -> list[dict]:
    """Load Spider schemas from tables.json."""
    with open(tables_file) as f:
        schemas = json.load(f)
    
    if limit:
        schemas = schemas[:limit]
    
    return schemas


def main(limit: Optional[int] = None):
    """
    Main ingestion pipeline.
    
    Args:
        limit: Optional limit on number of databases to ingest (for testing)
    """
    print("=" * 60)
    print("Spider Dataset → Vespa Ingestion")
    print("=" * 60)
    
    # Load schemas
    tables_file = SPIDER_DATA_PATH / "tables.json"
    print(f"\nLoading schemas from {tables_file}...")
    schemas = load_spider_schemas(tables_file, limit=limit)
    print(f"Loaded {len(schemas)} database schemas")
    
    # Parse all schemas
    all_nodes = []
    for schema in schemas:
        nodes = parse_spider_schema(schema)
        all_nodes.extend(nodes)
    
    # Count by type
    counts = {}
    for node in all_nodes:
        counts[node["node_type"]] = counts.get(node["node_type"], 0) + 1
    
    print(f"\nParsed nodes:")
    print(f"  Databases (data_product): {counts.get('data_product', 0)}")
    print(f"  Tables (entity): {counts.get('entity', 0)}")
    print(f"  Columns (property): {counts.get('property', 0)}")
    print(f"  Total: {len(all_nodes)}")
    
    # Connect to Vespa
    print(f"\nConnecting to Vespa at localhost:{VESPA_PORT}...")
    app = Vespa(url=f"http://localhost:{VESPA_PORT}")
    
    # Feed nodes
    timestamp = int(time.time())
    success = 0
    failed = 0
    
    print(f"\nFeeding {len(all_nodes)} nodes to Vespa...")
    
    with app.syncio() as session:
        for i, node in enumerate(all_nodes):
            # Add timestamp and embedding
            node["version_timestamp"] = timestamp
            node["embedding"] = embed(node["description"])
            
            # Remove extra fields not in schema
            node.pop("is_primary_key", None)
            node.pop("is_foreign_key", None)
            
            response: VespaResponse = session.feed_data_point(
                schema="metadata",
                data_id=node["id"],
                fields=node,
            )
            
            if response.is_successful():
                success += 1
            else:
                failed += 1
                if failed <= 5:  # Only show first 5 errors
                    print(f"  ❌ Failed: {node['id']}: {response.json}")
            
            # Progress update
            if (i + 1) % 500 == 0:
                print(f"  Progress: {i + 1}/{len(all_nodes)} ({success} success, {failed} failed)")
    
    print(f"\n{'=' * 60}")
    print(f"Ingestion complete!")
    print(f"  ✅ Success: {success}")
    print(f"  ❌ Failed: {failed}")
    print(f"{'=' * 60}")
    
    return success, failed


if __name__ == "__main__":
    import sys
    
    # Optional: limit number of databases for testing
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    
    if limit:
        print(f"Running with limit: {limit} databases")
    
    main(limit=limit)
