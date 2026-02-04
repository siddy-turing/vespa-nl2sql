"""
NL2SQL Benchmark Script

Evaluates NL2SQL pipeline against Spider dev.json:
1. Generate SQL using Vespa routing + LLM
2. Execute both generated SQL and gold SQL against Spider DB API
3. Compare results (exact match and execution match)
4. Report accuracy metrics

Usage:
    python benchmark.py              # Run 50 questions
    python benchmark.py 100          # Run 100 questions
    python benchmark.py --all        # Run all questions
"""

import json
import time
import requests
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from nl2sql import NL2SQLPipeline

# Spider execution API
SPIDER_API_URL = "http://54.226.193.202:8000/api/v1/execute"
SPIDER_DATA_PATH = Path("/Users/siddy/Desktop/vespa-ai-poc/spider_data")


@dataclass
class BenchmarkResult:
    """Result for a single question."""
    question: str
    db_id: str
    gold_sql: str
    generated_sql: str
    gold_result: Optional[dict] = None
    generated_result: Optional[dict] = None
    exact_match: bool = False
    execution_match: bool = False
    match_type: str = ""  # "exact", "distinct_diff", "subset", "superset", "no_match", "error"
    routing_time_ms: float = 0
    llm_time_ms: float = 0
    error: Optional[str] = None


def execute_sql(db_id: str, sql: str, timeout: int = 10) -> dict:
    """
    Execute SQL against Spider database API.
    
    Returns:
        {"success": True/False, "data": [...], "error": "..."}
    """
    try:
        response = requests.post(
            SPIDER_API_URL,
            json={
                "service": "spider1",
                "action": "execute_sql",
                "params": {
                    "db_id": db_id,
                    "sql": sql
                }
            },
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            return {"success": True, "data": result.get("data", result)}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Connection failed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    import re
    sql = sql.lower().strip()
    sql = re.sub(r'\s+', ' ', sql)  # Collapse whitespace
    sql = sql.rstrip(';')
    return sql


def results_match(result1: dict, result2: dict, check_duplicates: bool = True) -> tuple[bool, str]:
    """
    Compare two SQL execution results.
    Only compares row values, ignoring column names (due to aliases).
    
    Returns:
        (match: bool, match_type: str)
        match_type: "exact", "distinct_diff", "superset", "no_match"
    """
    if not result1.get("success") or not result2.get("success"):
        return False, "error"
    
    data1 = result1.get("data", {})
    data2 = result2.get("data", {})
    
    # Extract rows from nested structure
    # API format: {"data": {"rows": [[...]], "columns": [...]}}
    if isinstance(data1, dict):
        rows1 = data1.get("rows", data1.get("data", {}).get("rows", []))
    else:
        rows1 = data1
    
    if isinstance(data2, dict):
        rows2 = data2.get("rows", data2.get("data", {}).get("rows", []))
    else:
        rows2 = data2
    
    # Convert to comparable format (sort rows, handle None/null)
    def normalize_value(v):
        if v is None:
            return None
        if isinstance(v, float):
            return round(v, 6)  # Handle float precision
        return v
    
    def to_comparable(rows, dedupe=False):
        if not isinstance(rows, list):
            return rows
        try:
            normalized = [tuple(normalize_value(v) for v in row) for row in rows]
            if dedupe:
                normalized = list(set(normalized))
            return sorted(normalized)
        except TypeError:
            return rows
    
    comp1 = to_comparable(rows1)
    comp2 = to_comparable(rows2)
    
    # Exact match
    if comp1 == comp2:
        return True, "exact"
    
    # Check if difference is just DISTINCT (one has duplicates, other doesn't)
    comp1_deduped = to_comparable(rows1, dedupe=True)
    comp2_deduped = to_comparable(rows2, dedupe=True)
    
    if comp1_deduped == comp2_deduped:
        return True, "distinct_diff"  # Same unique values, different duplicates
    
    # Check if one is superset of other (generated query is more restrictive)
    if set(map(tuple, comp1_deduped)).issubset(set(map(tuple, comp2_deduped))):
        return False, "subset"
    if set(map(tuple, comp2_deduped)).issubset(set(map(tuple, comp1_deduped))):
        return False, "superset"
    
    return False, "no_match"


def load_dev_data(limit: Optional[int] = None) -> list[dict]:
    """Load Spider dev.json."""
    dev_file = SPIDER_DATA_PATH / "dev.json"
    with open(dev_file) as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
    
    return data


def run_benchmark(
    pipeline: NL2SQLPipeline,
    questions: list[dict],
    execute_queries: bool = True,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """
    Run benchmark on a list of questions.
    
    Args:
        pipeline: NL2SQL pipeline instance
        questions: List of questions from dev.json
        execute_queries: Whether to execute SQL against Spider API
        verbose: Print progress
    """
    results = []
    
    for i, item in enumerate(questions):
        question = item["question"]
        db_id = item["db_id"]
        gold_sql = item["query"]
        
        if verbose:
            print(f"\n[{i+1}/{len(questions)}] {question[:60]}...")
        
        result = BenchmarkResult(
            question=question,
            db_id=db_id,
            gold_sql=gold_sql,
            generated_sql="",
        )
        
        try:
            # Generate SQL using pipeline
            start_time = time.time()
            pipeline_result = pipeline.run(question, target_db=db_id, verbose=False)
            total_time = (time.time() - start_time) * 1000
            
            generated_sql = pipeline_result["sql"]
            result.generated_sql = generated_sql
            result.routing_time_ms = total_time * 0.3  # Estimate
            result.llm_time_ms = total_time * 0.7
            
            # Check exact match (normalized)
            result.exact_match = normalize_sql(generated_sql) == normalize_sql(gold_sql)
            
            if verbose:
                match_str = "✅ EXACT" if result.exact_match else "❌"
                print(f"   Gold: {gold_sql[:50]}...")
                print(f"   Gen:  {generated_sql[:50]}...")
                print(f"   {match_str}")
            
            # Execute queries if enabled
            if execute_queries:
                result.gold_result = execute_sql(db_id, gold_sql)
                result.generated_result = execute_sql(db_id, generated_sql)
                result.execution_match, result.match_type = results_match(result.gold_result, result.generated_result)
                
                if verbose and not result.exact_match:
                    if result.execution_match:
                        if result.match_type == "distinct_diff":
                            exec_str = "✅ EXEC MATCH (distinct diff)"
                        else:
                            exec_str = "✅ EXEC MATCH"
                    else:
                        exec_str = f"❌ EXEC DIFF ({result.match_type})"
                    print(f"   {exec_str}")
        
        except Exception as e:
            result.error = str(e)
            if verbose:
                print(f"   ❌ Error: {e}")
        
        results.append(result)
        
        # Rate limiting
        time.sleep(0.5)
    
    return results


def print_summary(results: list[BenchmarkResult]):
    """Print benchmark summary."""
    total = len(results)
    errors = sum(1 for r in results if r.error)
    successful = total - errors
    
    if successful == 0:
        print("No successful evaluations!")
        return
    
    valid_results = [r for r in results if not r.error]
    
    exact_matches = sum(1 for r in valid_results if r.exact_match)
    exec_matches = sum(1 for r in valid_results if r.execution_match)
    
    # Timing stats
    avg_routing = sum(r.routing_time_ms for r in valid_results) / len(valid_results)
    avg_llm = sum(r.llm_time_ms for r in valid_results) / len(valid_results)
    
    print("\n" + "=" * 70)
    print("                      BENCHMARK RESULTS")
    print("=" * 70)
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total questions:     {total}")
    print(f"   Successful:          {successful}")
    print(f"   Errors:              {errors}")
    
    # Count match types
    match_types = defaultdict(int)
    for r in valid_results:
        match_types[r.match_type] += 1
    
    print(f"\n🎯 Accuracy:")
    print(f"   Exact SQL Match:     {exact_matches}/{successful} ({exact_matches/successful*100:.1f}%)")
    print(f"   Execution Match:     {exec_matches}/{successful} ({exec_matches/successful*100:.1f}%)")
    
    print(f"\n📋 Match Type Breakdown:")
    print(f"   Exact results:       {match_types.get('exact', 0)}")
    print(f"   DISTINCT diff only:  {match_types.get('distinct_diff', 0)}")
    print(f"   Subset (stricter):   {match_types.get('subset', 0)}")
    print(f"   Superset (looser):   {match_types.get('superset', 0)}")
    print(f"   No match:            {match_types.get('no_match', 0)}")
    print(f"   Errors:              {match_types.get('error', 0)}")
    
    print(f"\n⏱️  Timing (avg):")
    print(f"   Vespa Routing:       {avg_routing:.0f}ms")
    print(f"   LLM Generation:      {avg_llm:.0f}ms")
    print(f"   Total:               {avg_routing + avg_llm:.0f}ms")
    
    # Per-database breakdown
    db_stats = defaultdict(lambda: {"total": 0, "exact": 0, "exec": 0})
    for r in valid_results:
        db_stats[r.db_id]["total"] += 1
        if r.exact_match:
            db_stats[r.db_id]["exact"] += 1
        if r.execution_match:
            db_stats[r.db_id]["exec"] += 1
    
    print(f"\n📁 Per-Database Accuracy (top 10):")
    sorted_dbs = sorted(db_stats.items(), key=lambda x: x[1]["total"], reverse=True)[:10]
    for db, stats in sorted_dbs:
        exact_pct = stats["exact"] / stats["total"] * 100
        exec_pct = stats["exec"] / stats["total"] * 100
        print(f"   {db:25} Exact: {exact_pct:5.1f}%  Exec: {exec_pct:5.1f}%  (n={stats['total']})")
    
    print("=" * 70)


def save_results(results: list[BenchmarkResult], output_file: str = "benchmark_results.json"):
    """Save results to JSON file."""
    output = []
    for r in results:
        output.append({
            "question": r.question,
            "db_id": r.db_id,
            "gold_sql": r.gold_sql,
            "generated_sql": r.generated_sql,
            "exact_match": r.exact_match,
            "execution_match": r.execution_match,
            "match_type": r.match_type,
            "error": r.error,
        })
    
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


def main():
    import sys
    
    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            limit = None
        else:
            limit = int(sys.argv[1])
    else:
        limit = 50  # Default
    
    print("=" * 70)
    print("        NL2SQL Benchmark: Vespa Routing + OpenAI")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"   Questions: {limit if limit else 'ALL'}")
    print(f"   LLM: OpenAI GPT-4o-mini")
    print(f"   API: {SPIDER_API_URL}")
    
    # Load data
    print(f"\nLoading Spider dev.json...")
    questions = load_dev_data(limit=limit)
    print(f"Loaded {len(questions)} questions")
    
    # Initialize pipeline
    print("\nInitializing NL2SQL pipeline...")
    pipeline = NL2SQLPipeline(llm_provider="openai")
    
    # Test API connectivity
    print("\nTesting Spider API connectivity...")
    test_result = execute_sql("concert_singer", "SELECT 1")
    if test_result["success"]:
        print("   ✅ API connected")
    else:
        print(f"   ⚠️  API issue: {test_result.get('error')}")
        print("   Continuing without execution matching...")
    
    # Run benchmark
    print("\n" + "-" * 70)
    print("Running benchmark...")
    print("-" * 70)
    
    results = run_benchmark(
        pipeline=pipeline,
        questions=questions,
        execute_queries=test_result["success"],
        verbose=True,
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results)


if __name__ == "__main__":
    main()
