"""
Evaluation Script for NL2SQL with Vespa Routing

Uses Spider dev.json to evaluate:
1. Database selection accuracy (does Vespa find the right DB?)
2. Table selection accuracy (does Vespa find the right tables?)
3. Column selection recall (does Vespa include the needed columns?)

This measures the Vespa routing layer independently of LLM SQL generation.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from nl2sql import VespaSchemaRouter, NL2SQLPipeline

SPIDER_DATA_PATH = Path("/Users/siddy/Desktop/vespa-ai-poc/spider_data")


@dataclass
class EvalResult:
    """Evaluation result for a single question."""
    question: str
    gold_db: str
    gold_tables: set
    gold_columns: set
    pred_db: str
    pred_tables: set
    pred_columns: set
    db_correct: bool
    table_recall: float
    column_recall: float
    error: Optional[str] = None


def extract_tables_from_sql(sql: str) -> set:
    """Extract table names from SQL query."""
    # Simple regex extraction - handles most cases
    sql_upper = sql.upper()
    
    tables = set()
    
    # FROM clause
    from_match = re.findall(r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE)
    tables.update(from_match)
    
    # JOIN clauses
    join_match = re.findall(r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE)
    tables.update(join_match)
    
    return {t.lower() for t in tables}


def extract_columns_from_sql(sql: str) -> set:
    """Extract column names from SQL query (simplified)."""
    # This is a simplified extraction - production would use SQL parser
    columns = set()
    
    # Remove string literals
    sql_clean = re.sub(r"'[^']*'", "", sql)
    sql_clean = re.sub(r'"[^"]*"', "", sql_clean)
    
    # Find column references (word.word or just word before/after certain keywords)
    # Table.Column pattern
    table_col = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)', sql_clean)
    for _, col in table_col:
        columns.add(col.lower())
    
    # Columns in SELECT (before FROM)
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_clean, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_clause = select_match.group(1)
        # Extract identifiers
        words = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', select_clause)
        # Filter out SQL keywords
        keywords = {'select', 'distinct', 'as', 'from', 'count', 'sum', 'avg', 'max', 'min', 'case', 'when', 'then', 'else', 'end'}
        columns.update(w.lower() for w in words if w.lower() not in keywords)
    
    # Columns in WHERE
    where_match = re.search(r'WHERE\s+(.*?)(?:GROUP|ORDER|LIMIT|$)', sql_clean, re.IGNORECASE | re.DOTALL)
    if where_match:
        where_clause = where_match.group(1)
        words = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', where_clause)
        keywords = {'where', 'and', 'or', 'not', 'in', 'like', 'between', 'is', 'null', 'true', 'false'}
        columns.update(w.lower() for w in words if w.lower() not in keywords)
    
    return columns


def load_dev_data(limit: Optional[int] = None) -> list[dict]:
    """Load Spider dev.json."""
    dev_file = SPIDER_DATA_PATH / "dev.json"
    with open(dev_file) as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
    
    return data


def evaluate_routing(
    router: VespaSchemaRouter,
    questions: list[dict],
    verbose: bool = True,
) -> list[EvalResult]:
    """
    Evaluate Vespa routing accuracy.
    
    Metrics:
    - Database accuracy: Did we select the correct database?
    - Table recall: What fraction of gold tables did we include?
    - Column recall: What fraction of gold columns did we include?
    """
    results = []
    
    for i, item in enumerate(questions):
        question = item["question"]
        gold_db = item["db_id"]
        gold_sql = item["query"]
        
        # Extract gold tables and columns from SQL
        gold_tables = extract_tables_from_sql(gold_sql)
        gold_columns = extract_columns_from_sql(gold_sql)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(questions)}")
        
        try:
            # Run Vespa routing (without target_db to test DB selection)
            schema_context = router.route(
                question=question,
                target_db=None,  # Let Vespa find the DB
                top_tables=5,
                top_columns=15,
            )
            
            pred_db = schema_context.database
            pred_tables = {t["name"].lower() for t in schema_context.tables}
            pred_columns = set()
            for table in schema_context.tables:
                for col in table["columns"]:
                    pred_columns.add(col["name"].lower())
            
            # Calculate metrics
            db_correct = pred_db.lower() == gold_db.lower()
            
            # Table recall: what fraction of gold tables did we find?
            if gold_tables:
                table_recall = len(gold_tables & pred_tables) / len(gold_tables)
            else:
                table_recall = 1.0
            
            # Column recall: what fraction of gold columns did we find?
            if gold_columns:
                column_recall = len(gold_columns & pred_columns) / len(gold_columns)
            else:
                column_recall = 1.0
            
            result = EvalResult(
                question=question,
                gold_db=gold_db,
                gold_tables=gold_tables,
                gold_columns=gold_columns,
                pred_db=pred_db,
                pred_tables=pred_tables,
                pred_columns=pred_columns,
                db_correct=db_correct,
                table_recall=table_recall,
                column_recall=column_recall,
            )
            
        except Exception as e:
            result = EvalResult(
                question=question,
                gold_db=gold_db,
                gold_tables=gold_tables,
                gold_columns=gold_columns,
                pred_db="",
                pred_tables=set(),
                pred_columns=set(),
                db_correct=False,
                table_recall=0.0,
                column_recall=0.0,
                error=str(e),
            )
        
        results.append(result)
    
    return results


def print_summary(results: list[EvalResult]):
    """Print evaluation summary."""
    total = len(results)
    errors = sum(1 for r in results if r.error)
    successful = total - errors
    
    if successful == 0:
        print("No successful evaluations!")
        return
    
    # Filter to successful results
    valid_results = [r for r in results if not r.error]
    
    db_accuracy = sum(1 for r in valid_results if r.db_correct) / len(valid_results)
    avg_table_recall = sum(r.table_recall for r in valid_results) / len(valid_results)
    avg_column_recall = sum(r.column_recall for r in valid_results) / len(valid_results)
    
    # Perfect routing (all 3 metrics = 1.0)
    perfect_db = sum(1 for r in valid_results if r.db_correct)
    perfect_table = sum(1 for r in valid_results if r.table_recall == 1.0)
    perfect_column = sum(1 for r in valid_results if r.column_recall == 1.0)
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nTotal questions: {total}")
    print(f"Successful: {successful}")
    print(f"Errors: {errors}")
    
    print(f"\n📊 Routing Accuracy:")
    print(f"  Database selection:  {db_accuracy*100:.1f}% ({perfect_db}/{successful})")
    print(f"  Table recall (avg):  {avg_table_recall*100:.1f}%")
    print(f"  Column recall (avg): {avg_column_recall*100:.1f}%")
    
    print(f"\n🎯 Perfect Scores:")
    print(f"  Perfect DB match:     {perfect_db}/{successful} ({perfect_db/successful*100:.1f}%)")
    print(f"  Perfect table recall: {perfect_table}/{successful} ({perfect_table/successful*100:.1f}%)")
    print(f"  Perfect column recall: {perfect_column}/{successful} ({perfect_column/successful*100:.1f}%)")
    
    # Breakdown by database
    db_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in valid_results:
        db_stats[r.gold_db]["total"] += 1
        if r.db_correct:
            db_stats[r.gold_db]["correct"] += 1
    
    print(f"\n📁 Per-Database Accuracy (top 10):")
    sorted_dbs = sorted(db_stats.items(), key=lambda x: x[1]["total"], reverse=True)[:10]
    for db, stats in sorted_dbs:
        acc = stats["correct"] / stats["total"] * 100
        print(f"  {db}: {acc:.0f}% ({stats['correct']}/{stats['total']})")
    
    print("=" * 60)


def show_errors(results: list[EvalResult], n: int = 5):
    """Show sample of routing errors for analysis."""
    errors = [r for r in results if not r.db_correct and not r.error]
    
    print(f"\n📋 Sample Routing Errors (showing {min(n, len(errors))}):")
    for r in errors[:n]:
        print(f"\n  Question: {r.question}")
        print(f"  Gold DB: {r.gold_db} | Predicted: {r.pred_db}")
        print(f"  Gold tables: {r.gold_tables} | Predicted: {r.pred_tables}")
        print(f"  Table recall: {r.table_recall:.1%} | Column recall: {r.column_recall:.1%}")


def main():
    """Run evaluation."""
    import sys
    
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    print("=" * 60)
    print("NL2SQL Vespa Routing Evaluation")
    print("=" * 60)
    
    print(f"\nLoading Spider dev.json (limit: {limit})...")
    questions = load_dev_data(limit=limit)
    print(f"Loaded {len(questions)} questions")
    
    print("\nInitializing Vespa router...")
    router = VespaSchemaRouter()
    
    print("\nEvaluating routing accuracy...")
    results = evaluate_routing(router, questions, verbose=True)
    
    print_summary(results)
    show_errors(results, n=5)


if __name__ == "__main__":
    main()
