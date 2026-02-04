"""
Vespa Application Package Definition

Semantic Router for Data Catalog Metadata
Architecture:
- BGE M3 embeddings (1024 dimensions, 8K token context)
- Hybrid ranking: semantic on descriptions + BM25 on aliases
- Tree structure: Data Product → Root Entity → Entity → Property
- 24-hour refresh cycle support
"""

from vespa.package import (
    ApplicationPackage,
    Schema,
    Document,
    Field,
    FieldSet,
    RankProfile,
    HNSW,
    FirstPhaseRanking,
)

# Document schema for metadata nodes
# Hierarchy: Data Product → Root Entity → Entity → Property
doc = Document(
    fields=[
        # Core identification
        Field(
            name="id",
            type="string",
            indexing=["summary", "attribute"],
        ),
        Field(
            name="node_type",
            type="string",
            indexing=["summary", "attribute"],
            attribute=["fast-search"],  # For efficient filtering
        ),
        Field(
            name="name",
            type="string",
            indexing=["index", "summary"],
            index="enable-bm25",
        ),
        
        # Description for semantic search (LLM-generated, 25-30 words)
        Field(
            name="description",
            type="string",
            indexing=["index", "summary"],
            index="enable-bm25",
        ),
        
        # Aliases from historical queries (for lexical BM25 search)
        Field(
            name="aliases",
            type="array<string>",
            indexing=["index", "summary"],
            index="enable-bm25",
        ),
        
        # BGE M3 embedding (1024 dimensions)
        Field(
            name="embedding",
            type="tensor<float>(x[1024])",
            indexing=["attribute", "index"],
            ann=HNSW(distance_metric="angular"),
        ),
        
        # Tree structure (single parent, multiple children)
        Field(
            name="parent_id",
            type="string",
            indexing=["summary", "attribute"],
            attribute=["fast-search"],
        ),
        Field(
            name="children_ids",
            type="array<string>",
            indexing=["summary", "attribute"],
        ),
        
        # Additional metadata
        Field(
            name="data_product_id",
            type="string",
            indexing=["summary", "attribute"],
            attribute=["fast-search"],  # Filter by data product
        ),
        Field(
            name="root_entity_id",
            type="string",
            indexing=["summary", "attribute"],
            attribute=["fast-search"],
        ),
        
        # For properties: enum values and data type
        Field(
            name="data_type",
            type="string",
            indexing=["summary", "attribute"],
        ),
        Field(
            name="enum_values",
            type="array<string>",
            indexing=["index", "summary"],
        ),
        
        # Versioning for 24-hour refresh tracking
        Field(
            name="version_timestamp",
            type="long",
            indexing=["summary", "attribute"],
        ),
    ]
)

# Schema with hybrid ranking profiles
schema = Schema(
    name="metadata",
    document=doc,
    fieldsets=[
        FieldSet(name="default", fields=["name", "description", "aliases"]),
        FieldSet(name="text_fields", fields=["name", "description"]),
    ],
    rank_profiles=[
        # Pure semantic search (on description embeddings)
        RankProfile(
            name="semantic",
            inputs=[("query(q)", "tensor<float>(x[1024])")],
            first_phase="closeness(field, embedding)",
        ),
        
        # Pure lexical search (BM25 on aliases)
        RankProfile(
            name="lexical",
            first_phase="bm25(aliases) + bm25(name)",
        ),
        
        # Hybrid ranking: semantic on description + BM25 on aliases
        # Weights: 60% semantic, 40% lexical (tunable)
        RankProfile(
            name="hybrid",
            inputs=[("query(q)", "tensor<float>(x[1024])")],
            first_phase="closeness(field, embedding) * 0.6 + (bm25(aliases) + bm25(name) + bm25(description)) * 0.4",
        ),
        
        # Alias-focused hybrid (for when aliases are more important)
        RankProfile(
            name="alias_heavy",
            inputs=[("query(q)", "tensor<float>(x[1024])")],
            first_phase="closeness(field, embedding) * 0.3 + bm25(aliases) * 0.5 + bm25(name) * 0.2",
        ),
    ],
)

# Application package
app_package = ApplicationPackage(
    name="datacatalog",
    schema=[schema],
)

if __name__ == "__main__":
    print("Vespa Data Catalog Schema defined successfully!")
    print(f"App name: {app_package.name}")
    print(f"Schema: {schema.name}")
    print(f"Embedding dimensions: 1024 (BGE M3)")
    print(f"Node types: data_product, root_entity, entity, property")
