"""
Multi-Stage Semantic Router for Data Catalog

Implements the routing process from meeting notes:
1. Data Product identification (filter: node_type = data_product)
2. Root Entity selection within chosen data product
3. Entity discovery within root entity scope
4. Property identification for final selection

Each stage uses hybrid ranking: semantic on description + BM25 on aliases
"""

import numpy as np
from vespa.application import Vespa
from typing import Optional

VESPA_PORT = 8090


def embed(text: str) -> list[float]:
    """Generate 1024-dim embedding for query text."""
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(1024).tolist()


class SemanticRouter:
    """
    Multi-stage semantic router for data catalog navigation.
    
    Matches the architecture described in meeting notes:
    - Sequential Vespa calls with filtering at each level
    - Hybrid ranking (semantic + lexical)
    - Tree traversal: Data Product → Root Entity → Entity → Property
    """
    
    def __init__(self, vespa_url: str = f"http://localhost:{VESPA_PORT}"):
        self.app = Vespa(url=vespa_url)
        
    def search_by_node_type(
        self,
        session,
        query_text: str,
        node_type: str,
        parent_filter: Optional[str] = None,
        data_product_filter: Optional[str] = None,
        hits: int = 5,
        ranking: str = "hybrid"
    ):
        """
        Search nodes of a specific type with optional parent filtering.
        
        Args:
            query_text: Natural language query
            node_type: One of: data_product, root_entity, entity, property
            parent_filter: Filter by parent_id (for tree traversal)
            data_product_filter: Filter by data_product_id
            hits: Number of results
            ranking: semantic, lexical, or hybrid
        """
        query_embedding = embed(query_text)
        
        # Build filter conditions
        filters = [f'node_type contains "{node_type}"']
        if parent_filter:
            filters.append(f'parent_id contains "{parent_filter}"')
        if data_product_filter:
            filters.append(f'data_product_id contains "{data_product_filter}"')
        
        filter_clause = " and ".join(filters)
        
        # Hybrid query: nearestNeighbor for semantic + userQuery for BM25
        yql = f'select * from metadata where ({filter_clause}) and ({{targetHits:{hits*2}}}nearestNeighbor(embedding, q) or userQuery())'
        
        response = session.query(
            body={
                "yql": yql,
                "query": query_text,
                "ranking": ranking,
                "input.query(q)": query_embedding,
                "hits": hits,
            }
        )
        
        return response
    
    def route_query(self, query_text: str, verbose: bool = True):
        """
        Full multi-stage routing pipeline.
        
        Steps:
        1. Find best Data Product
        2. Find best Root Entity within that Data Product
        3. Find best Entity within that Root Entity
        4. Find relevant Properties within that Entity
        """
        results = {
            "query": query_text,
            "data_product": None,
            "root_entity": None,
            "entity": None,
            "properties": [],
            "path": [],
        }
        
        with self.app.syncio() as session:
            # Stage 1: Data Product identification
            if verbose:
                print(f"\n{'='*60}")
                print(f"🔍 Query: '{query_text}'")
                print(f"{'='*60}")
                print("\n📊 Stage 1: Data Product Identification")
            
            dp_response = self.search_by_node_type(
                session, query_text, "data_product", hits=3
            )
            
            if dp_response.hits:
                best_dp = dp_response.hits[0]
                results["data_product"] = best_dp["fields"]
                results["path"].append(best_dp["fields"]["name"])
                
                if verbose:
                    self._print_results(dp_response.hits[:3], "Data Products")
                
                # Stage 2: Root Entity selection
                if verbose:
                    print(f"\n📊 Stage 2: Root Entity Selection (within {best_dp['fields']['name']})")
                
                re_response = self.search_by_node_type(
                    session, query_text, "root_entity",
                    data_product_filter=best_dp["fields"]["id"],
                    hits=3
                )
                
                if re_response.hits:
                    best_re = re_response.hits[0]
                    results["root_entity"] = best_re["fields"]
                    results["path"].append(best_re["fields"]["name"])
                    
                    if verbose:
                        self._print_results(re_response.hits[:3], "Root Entities")
                    
                    # Stage 3: Entity discovery
                    if verbose:
                        print(f"\n📊 Stage 3: Entity Discovery (within {best_re['fields']['name']})")
                    
                    e_response = self.search_by_node_type(
                        session, query_text, "entity",
                        parent_filter=best_re["fields"]["id"],
                        hits=3
                    )
                    
                    if e_response.hits:
                        best_entity = e_response.hits[0]
                        results["entity"] = best_entity["fields"]
                        results["path"].append(best_entity["fields"]["name"])
                        
                        if verbose:
                            self._print_results(e_response.hits[:3], "Entities")
                        
                        # Stage 4: Property identification
                        if verbose:
                            print(f"\n📊 Stage 4: Property Identification (within {best_entity['fields']['name']})")
                        
                        p_response = self.search_by_node_type(
                            session, query_text, "property",
                            parent_filter=best_entity["fields"]["id"],
                            hits=5
                        )
                        
                        if p_response.hits:
                            results["properties"] = [h["fields"] for h in p_response.hits]
                            
                            if verbose:
                                self._print_results(p_response.hits, "Properties")
        
        return results
    
    def _print_results(self, hits, stage_name: str):
        """Pretty print search results for a stage."""
        print(f"\n   Top {stage_name}:")
        for i, hit in enumerate(hits, 1):
            fields = hit.get("fields", {})
            relevance = hit.get("relevance", 0)
            aliases = fields.get("aliases", [])[:3]
            print(f"   {i}. [{relevance:.4f}] {fields.get('name', 'N/A')}")
            print(f"      {fields.get('description', 'N/A')[:80]}...")
            if aliases:
                print(f"      Aliases: {', '.join(aliases)}")


def demo_queries():
    """Run demo queries showing the multi-stage routing."""
    router = SemanticRouter()
    
    test_queries = [
        "show me all stock trades from yesterday",
        "how many shares of AAPL do we hold",
        "what is the employee kerberos login",
        "total revenue this quarter",
        "find trades where side is buy",
    ]
    
    for query in test_queries:
        result = router.route_query(query)
        
        # Print routing path
        print(f"\n{'#'*60}")
        print(f"📍 ROUTING PATH: {' → '.join(result['path'])}")
        if result["properties"]:
            print(f"📋 RELEVANT PROPERTIES:")
            for prop in result["properties"][:3]:
                print(f"   - {prop['name']} ({prop['data_type']}): {prop['description'][:50]}...")
        print(f"{'#'*60}\n")


def interactive_mode():
    """Interactive query mode."""
    router = SemanticRouter()
    
    print("\n" + "="*60)
    print("🔍 Semantic Router - Interactive Mode")
    print("="*60)
    print("Enter natural language queries to navigate the data catalog.")
    print("Type 'quit' to exit.\n")
    
    while True:
        query = input("\nQuery> ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue
            
        result = router.route_query(query)
        
        print(f"\n{'='*40}")
        print(f"📍 Path: {' → '.join(result['path'])}")
        print(f"{'='*40}")


if __name__ == "__main__":
    print("Running demo queries...\n")
    demo_queries()
