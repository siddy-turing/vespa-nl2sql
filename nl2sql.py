"""
NL2SQL Pipeline with Vespa Schema Routing

Architecture:
1. Vespa Layer: Routes natural language query to relevant schema elements
   - Stage 1: Database identification
   - Stage 2: Table selection
   - Stage 3: Column identification
2. LLM Layer: Generates SQL from narrowed schema context

This reduces the schema context passed to LLM, improving accuracy and reducing costs.
"""

import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from vespa.application import Vespa
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

VESPA_PORT = 8090
SPIDER_DATA_PATH = Path("/Users/siddy/Desktop/vespa-ai-poc/spider_data")


def embed(text: str) -> list[float]:
    """Generate 1024-dim embedding."""
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(1024).tolist()


@dataclass
class SchemaContext:
    """Narrowed schema context from Vespa routing."""
    database: str
    tables: list[dict]  # [{name, columns: [{name, type}]}]
    confidence_scores: dict
    
    def to_prompt_context(self) -> str:
        """Format schema for LLM prompt."""
        lines = [f"Database: {self.database}", ""]
        
        for table in self.tables:
            cols = ", ".join([f"{c['name']} ({c['type']})" for c in table["columns"]])
            lines.append(f"Table {table['name']}: {cols}")
        
        return "\n".join(lines)
    
    def to_ddl(self) -> str:
        """Format schema as CREATE TABLE statements."""
        ddl_lines = []
        for table in self.tables:
            cols = []
            for col in table["columns"]:
                col_type = "TEXT" if col["type"] == "text" else "INTEGER" if col["type"] == "number" else col["type"].upper()
                cols.append(f"  {col['name']} {col_type}")
            
            ddl_lines.append(f"CREATE TABLE {table['name']} (")
            ddl_lines.append(",\n".join(cols))
            ddl_lines.append(");")
            ddl_lines.append("")
        
        return "\n".join(ddl_lines)


class VespaSchemaRouter:
    """
    Routes natural language queries to relevant schema elements using Vespa.
    
    Input: Natural language question
    Output: Narrowed schema context (database, tables, columns)
    """
    
    def __init__(self, vespa_url: str = f"http://localhost:{VESPA_PORT}"):
        self.app = Vespa(url=vespa_url)
    
    def search_nodes(
        self,
        session,
        query_text: str,
        node_type: str,
        parent_filter: Optional[str] = None,
        data_product_filter: Optional[str] = None,
        hits: int = 5,
    ) -> list[dict]:
        """Search Vespa for nodes of a specific type."""
        query_embedding = embed(query_text)
        
        # Build filter
        filters = [f'node_type contains "{node_type}"']
        if parent_filter:
            filters.append(f'parent_id contains "{parent_filter}"')
        if data_product_filter:
            filters.append(f'data_product_id contains "{data_product_filter}"')
        
        filter_clause = " and ".join(filters)
        yql = f'select * from metadata where ({filter_clause}) and ({{targetHits:{hits*2}}}nearestNeighbor(embedding, q) or userQuery())'
        
        response = session.query(
            body={
                "yql": yql,
                "query": query_text,
                "ranking": "hybrid",
                "input.query(q)": query_embedding,
                "hits": hits,
            }
        )
        
        return response.hits if response.hits else []
    
    def route(self, question: str, target_db: Optional[str] = None, top_tables: int = 3, top_columns: int = 10) -> SchemaContext:
        """
        Route a natural language question to relevant schema elements.
        
        Args:
            question: Natural language question
            target_db: Optional - if known, skip database selection
            top_tables: Number of tables to include in context
            top_columns: Number of columns per table to include
        
        Returns:
            SchemaContext with narrowed schema
        """
        confidence_scores = {}
        
        with self.app.syncio() as session:
            # Stage 1: Database identification
            if target_db:
                db_id = f"db_{target_db}"
                db_name = target_db
                confidence_scores["database"] = 1.0  # Known
            else:
                db_hits = self.search_nodes(session, question, "data_product", hits=3)
                if not db_hits:
                    raise ValueError("No database found for query")
                
                best_db = db_hits[0]
                db_id = best_db["fields"]["id"]
                db_name = best_db["fields"]["name"]
                confidence_scores["database"] = best_db.get("relevance", 0)
            
            # Stage 2: Table selection
            table_hits = self.search_nodes(
                session, question, "entity",
                data_product_filter=db_id,
                hits=top_tables
            )
            
            if not table_hits:
                raise ValueError(f"No tables found in database {db_name}")
            
            confidence_scores["tables"] = [h.get("relevance", 0) for h in table_hits]
            
            # Stage 3: Column identification for each selected table
            tables = []
            for table_hit in table_hits:
                table_id = table_hit["fields"]["id"]
                table_name = table_hit["fields"]["name"]
                
                # Get columns for this table
                col_hits = self.search_nodes(
                    session, question, "property",
                    parent_filter=table_id,
                    hits=top_columns
                )
                
                columns = []
                for col_hit in col_hits:
                    columns.append({
                        "name": col_hit["fields"]["name"],
                        "type": col_hit["fields"].get("data_type", "text"),
                        "score": col_hit.get("relevance", 0),
                    })
                
                tables.append({
                    "name": table_name,
                    "columns": columns,
                    "score": table_hit.get("relevance", 0),
                })
            
            confidence_scores["columns"] = [[c["score"] for c in t["columns"]] for t in tables]
        
        return SchemaContext(
            database=db_name,
            tables=tables,
            confidence_scores=confidence_scores,
        )


class NL2SQLPipeline:
    """
    Complete NL2SQL pipeline combining Vespa routing with LLM generation.
    
    Flow:
    1. User question → Vespa routing → Narrowed schema context
    2. Narrowed schema + question → LLM → SQL query
    """
    
    def __init__(self, llm_provider: str = "stub"):
        """
        Args:
            llm_provider: "openai", "anthropic", or "stub" (for testing)
        """
        self.router = VespaSchemaRouter()
        self.llm_provider = llm_provider
    
    def generate_sql_prompt(self, question: str, schema_context: SchemaContext) -> str:
        """Generate prompt for LLM SQL generation."""
        return f"""You are a SQL expert. Generate a SQL query to answer the user's question.

DATABASE SCHEMA:
{schema_context.to_ddl()}

QUESTION: {question}

RULES:
- Use only the tables and columns provided above
- Return ONLY the SQL query, no explanation
- Use standard SQL syntax
- Do NOT add DISTINCT unless the question explicitly asks for "unique", "distinct", or "different" values
- Do NOT add column aliases (AS) - use bare column names
- Do NOT add unnecessary ORDER BY unless the question asks for ordering
- Keep the query as simple as possible

SQL:"""
    
    def call_llm(self, prompt: str) -> str:
        """
        Call LLM to generate SQL.
        
        Supported providers:
        - openai: GPT-4o-mini (fast, cheap) or GPT-4o (more accurate)
        - anthropic: Claude 3.5 Sonnet
        - stub: Placeholder for testing
        """
        if self.llm_provider == "stub":
            # Return a placeholder for testing
            return "-- LLM SQL generation placeholder --\nSELECT * FROM table LIMIT 10;"
        
        elif self.llm_provider == "openai":
            from openai import OpenAI
            
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cheap, use "gpt-4o" for better accuracy
                messages=[
                    {"role": "system", "content": "You are a SQL expert. Generate only valid SQL queries, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500,
            )
            
            sql = response.choices[0].message.content.strip()
            
            # Clean up markdown code blocks if present
            if sql.startswith("```"):
                lines = sql.split("\n")
                sql = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            return sql
        
        elif self.llm_provider == "anthropic":
            import anthropic
            
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
    
    def run(
        self,
        question: str,
        target_db: Optional[str] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Run the full NL2SQL pipeline.
        
        Args:
            question: Natural language question
            target_db: Optional database ID (if known)
            verbose: Print debug info
        
        Returns:
            {
                "question": str,
                "schema_context": SchemaContext,
                "prompt": str,
                "sql": str,
                "confidence": dict,
            }
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 Question: {question}")
            print(f"{'='*60}")
        
        # Stage 1: Vespa routing
        if verbose:
            print("\n📊 Stage 1: Vespa Schema Routing...")
        
        schema_context = self.router.route(
            question=question,
            target_db=target_db,
            top_tables=3,
            top_columns=8,
        )
        
        if verbose:
            print(f"   Database: {schema_context.database} (score: {schema_context.confidence_scores.get('database', 'N/A'):.3f})")
            print(f"   Tables: {[t['name'] for t in schema_context.tables]}")
            for table in schema_context.tables:
                print(f"     - {table['name']}: {[c['name'] for c in table['columns'][:5]]}...")
        
        # Stage 2: LLM SQL generation
        if verbose:
            print("\n📊 Stage 2: LLM SQL Generation...")
        
        prompt = self.generate_sql_prompt(question, schema_context)
        sql = self.call_llm(prompt)
        
        if verbose:
            print(f"\n📝 Generated SQL:")
            print(f"   {sql}")
        
        return {
            "question": question,
            "database": schema_context.database,
            "schema_context": schema_context,
            "prompt": prompt,
            "sql": sql,
            "confidence": schema_context.confidence_scores,
        }


def demo():
    """Demo the NL2SQL pipeline with sample questions."""
    pipeline = NL2SQLPipeline(llm_provider="stub")
    
    # Sample questions (would need Spider data ingested first)
    questions = [
        ("How many singers do we have?", "concert_singer"),
        ("What is the name of the oldest singer?", "concert_singer"),
        ("List all students in the math department", None),  # Let Vespa find the database
    ]
    
    for question, db in questions:
        try:
            result = pipeline.run(question, target_db=db)
            print(f"\n{'#'*60}")
            print(f"✅ Pipeline completed successfully")
            print(f"{'#'*60}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def interactive():
    """Interactive NL2SQL mode."""
    pipeline = NL2SQLPipeline(llm_provider="stub")
    
    print("\n" + "="*60)
    print("🔍 NL2SQL Interactive Mode")
    print("="*60)
    print("Enter natural language questions to generate SQL.")
    print("Format: [db_name:] question")
    print("Example: concert_singer: How many singers are there?")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("\nQuestion> ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if not user_input:
            continue
        
        # Parse optional database prefix
        if ":" in user_input:
            db, question = user_input.split(":", 1)
            db = db.strip()
            question = question.strip()
        else:
            db = None
            question = user_input
        
        try:
            result = pipeline.run(question, target_db=db)
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive()
    else:
        demo()
