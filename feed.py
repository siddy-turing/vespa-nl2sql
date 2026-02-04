"""
Feed Hierarchical Metadata to Vespa

Data structure: Data Product → Root Entity → Entity → Property
Matches the semantic router architecture from meeting notes.

In production, replace dummy embeddings with BGE M3:
  pip install FlagEmbedding
  from FlagEmbedding import BGEM3FlagModel
  model = BGEM3FlagModel('BAAI/bge-m3')
  embeddings = model.encode(texts)['dense_vecs']
"""

import numpy as np
import time
from vespa.application import Vespa
from vespa.io import VespaResponse

VESPA_PORT = 8090


def embed(text: str) -> list[float]:
    """
    Generate 1024-dim embedding for text.
    
    TODO: Replace with BGE M3 in production:
    ```
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    embedding = model.encode([text])['dense_vecs'][0]
    ```
    """
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(1024).tolist()


# Sample hierarchical data matching the meeting notes structure
# Hierarchy: Data Product → Root Entity → Entity → Property

SAMPLE_DATA = [
    # ========== DATA PRODUCT 1: Trading Platform ==========
    {
        "id": "dp_trading",
        "node_type": "data_product",
        "name": "Trading Platform",
        "description": "Enterprise trading platform containing all trade execution data, order management, position tracking, and market data feeds for equity and fixed income instruments",
        "aliases": ["trades", "trading system", "order book", "execution platform"],
        "parent_id": "",
        "children_ids": ["re_equities", "re_fixed_income"],
        "data_product_id": "dp_trading",
        "root_entity_id": "",
        "data_type": "",
        "enum_values": [],
    },
    
    # Root Entity: Equities
    {
        "id": "re_equities",
        "node_type": "root_entity",
        "name": "Equities",
        "description": "Equities trading database containing stock trades, equity positions, and market quotes for NYSE, NASDAQ, and international exchanges",
        "aliases": ["stocks", "equity trades", "share transactions"],
        "parent_id": "dp_trading",
        "children_ids": ["e_trades", "e_positions"],
        "data_product_id": "dp_trading",
        "root_entity_id": "re_equities",
        "data_type": "",
        "enum_values": [],
    },
    
    # Entity: Trades
    {
        "id": "e_trades",
        "node_type": "entity",
        "name": "Trades",
        "description": "Trade execution records containing buy and sell orders, execution prices, timestamps, counterparties, and settlement details for equity instruments",
        "aliases": ["executions", "fills", "trade records", "transactions"],
        "parent_id": "re_equities",
        "children_ids": ["p_trade_id", "p_symbol", "p_side", "p_quantity", "p_price", "p_trade_date"],
        "data_product_id": "dp_trading",
        "root_entity_id": "re_equities",
        "data_type": "",
        "enum_values": [],
    },
    
    # Properties of Trades
    {
        "id": "p_trade_id",
        "node_type": "property",
        "name": "trade_id",
        "description": "Unique identifier for each trade execution, used for reconciliation and audit trail purposes",
        "aliases": ["execution id", "transaction id", "fill id"],
        "parent_id": "e_trades",
        "children_ids": [],
        "data_product_id": "dp_trading",
        "root_entity_id": "re_equities",
        "data_type": "string",
        "enum_values": [],
    },
    {
        "id": "p_symbol",
        "node_type": "property",
        "name": "symbol",
        "description": "Stock ticker symbol identifying the traded security, following standard exchange notation",
        "aliases": ["ticker", "stock symbol", "security id", "instrument"],
        "parent_id": "e_trades",
        "children_ids": [],
        "data_product_id": "dp_trading",
        "root_entity_id": "re_equities",
        "data_type": "string",
        "enum_values": [],
    },
    {
        "id": "p_side",
        "node_type": "property",
        "name": "side",
        "description": "Direction of the trade indicating whether it was a purchase or sale of the security",
        "aliases": ["direction", "buy sell", "trade direction"],
        "parent_id": "e_trades",
        "children_ids": [],
        "data_product_id": "dp_trading",
        "root_entity_id": "re_equities",
        "data_type": "enum",
        "enum_values": ["BUY", "SELL"],
    },
    {
        "id": "p_quantity",
        "node_type": "property",
        "name": "quantity",
        "description": "Number of shares or units traded in the transaction",
        "aliases": ["shares", "volume", "size", "amount"],
        "parent_id": "e_trades",
        "children_ids": [],
        "data_product_id": "dp_trading",
        "root_entity_id": "re_equities",
        "data_type": "integer",
        "enum_values": [],
    },
    {
        "id": "p_price",
        "node_type": "property",
        "name": "price",
        "description": "Execution price per share at which the trade was filled",
        "aliases": ["exec price", "fill price", "trade price", "cost"],
        "parent_id": "e_trades",
        "children_ids": [],
        "data_product_id": "dp_trading",
        "root_entity_id": "re_equities",
        "data_type": "decimal",
        "enum_values": [],
    },
    {
        "id": "p_trade_date",
        "node_type": "property",
        "name": "trade_date",
        "description": "Date and time when the trade was executed on the exchange",
        "aliases": ["execution time", "timestamp", "trade timestamp", "when"],
        "parent_id": "e_trades",
        "children_ids": [],
        "data_product_id": "dp_trading",
        "root_entity_id": "re_equities",
        "data_type": "datetime",
        "enum_values": [],
    },
    
    # Entity: Positions
    {
        "id": "e_positions",
        "node_type": "entity",
        "name": "Positions",
        "description": "Current portfolio holdings showing quantity of shares held, average cost basis, and current market value for each security",
        "aliases": ["holdings", "portfolio", "inventory"],
        "parent_id": "re_equities",
        "children_ids": ["p_pos_symbol", "p_pos_quantity", "p_avg_cost"],
        "data_product_id": "dp_trading",
        "root_entity_id": "re_equities",
        "data_type": "",
        "enum_values": [],
    },
    {
        "id": "p_pos_symbol",
        "node_type": "property",
        "name": "symbol",
        "description": "Stock ticker symbol for the position holding",
        "aliases": ["ticker", "security"],
        "parent_id": "e_positions",
        "children_ids": [],
        "data_product_id": "dp_trading",
        "root_entity_id": "re_equities",
        "data_type": "string",
        "enum_values": [],
    },
    {
        "id": "p_pos_quantity",
        "node_type": "property",
        "name": "quantity",
        "description": "Total number of shares currently held in the position",
        "aliases": ["shares held", "holding size"],
        "parent_id": "e_positions",
        "children_ids": [],
        "data_product_id": "dp_trading",
        "root_entity_id": "re_equities",
        "data_type": "integer",
        "enum_values": [],
    },
    {
        "id": "p_avg_cost",
        "node_type": "property",
        "name": "average_cost",
        "description": "Average purchase price per share across all buy transactions for cost basis calculation",
        "aliases": ["cost basis", "avg price", "book value"],
        "parent_id": "e_positions",
        "children_ids": [],
        "data_product_id": "dp_trading",
        "root_entity_id": "re_equities",
        "data_type": "decimal",
        "enum_values": [],
    },
    
    # ========== DATA PRODUCT 2: HR System ==========
    {
        "id": "dp_hr",
        "node_type": "data_product",
        "name": "HR System",
        "description": "Human resources management system containing employee records, organizational hierarchy, compensation data, and workforce analytics",
        "aliases": ["human resources", "employee data", "workforce", "personnel"],
        "parent_id": "",
        "children_ids": ["re_employees"],
        "data_product_id": "dp_hr",
        "root_entity_id": "",
        "data_type": "",
        "enum_values": [],
    },
    
    # Root Entity: Employees
    {
        "id": "re_employees",
        "node_type": "root_entity",
        "name": "Employees",
        "description": "Employee master database containing personal information, employment details, and organizational assignments",
        "aliases": ["staff", "workers", "team members", "personnel records"],
        "parent_id": "dp_hr",
        "children_ids": ["e_employee_records"],
        "data_product_id": "dp_hr",
        "root_entity_id": "re_employees",
        "data_type": "",
        "enum_values": [],
    },
    
    # Entity: Employee Records
    {
        "id": "e_employee_records",
        "node_type": "entity",
        "name": "Employee Records",
        "description": "Core employee information including identifiers, names, department assignments, and employment status",
        "aliases": ["emp table", "staff records", "worker info"],
        "parent_id": "re_employees",
        "children_ids": ["p_emp_id", "p_kerberos", "p_name", "p_department", "p_status"],
        "data_product_id": "dp_hr",
        "root_entity_id": "re_employees",
        "data_type": "",
        "enum_values": [],
    },
    
    # Properties of Employee Records
    {
        "id": "p_emp_id",
        "node_type": "property",
        "name": "employee_id",
        "description": "Unique numeric identifier assigned to each employee in the HR system",
        "aliases": ["emp id", "staff id", "worker id", "personnel number"],
        "parent_id": "e_employee_records",
        "children_ids": [],
        "data_product_id": "dp_hr",
        "root_entity_id": "re_employees",
        "data_type": "string",
        "enum_values": [],
    },
    {
        "id": "p_kerberos",
        "node_type": "property",
        "name": "kerberos_id",
        "description": "Network login identifier used for authentication and access control across enterprise systems",
        "aliases": ["login", "username", "network id", "kerb"],
        "parent_id": "e_employee_records",
        "children_ids": [],
        "data_product_id": "dp_hr",
        "root_entity_id": "re_employees",
        "data_type": "string",
        "enum_values": [],
    },
    {
        "id": "p_name",
        "node_type": "property",
        "name": "full_name",
        "description": "Employee's complete legal name as recorded in HR system",
        "aliases": ["name", "employee name", "person name"],
        "parent_id": "e_employee_records",
        "children_ids": [],
        "data_product_id": "dp_hr",
        "root_entity_id": "re_employees",
        "data_type": "string",
        "enum_values": [],
    },
    {
        "id": "p_department",
        "node_type": "property",
        "name": "department",
        "description": "Organizational unit or business division where the employee is assigned",
        "aliases": ["dept", "team", "division", "org unit"],
        "parent_id": "e_employee_records",
        "children_ids": [],
        "data_product_id": "dp_hr",
        "root_entity_id": "re_employees",
        "data_type": "string",
        "enum_values": [],
    },
    {
        "id": "p_status",
        "node_type": "property",
        "name": "employment_status",
        "description": "Current employment status indicating whether the employee is active, on leave, or terminated",
        "aliases": ["status", "active", "emp status"],
        "parent_id": "e_employee_records",
        "children_ids": [],
        "data_product_id": "dp_hr",
        "root_entity_id": "re_employees",
        "data_type": "enum",
        "enum_values": ["ACTIVE", "ON_LEAVE", "TERMINATED"],
    },
    
    # ========== DATA PRODUCT 3: Revenue Analytics ==========
    {
        "id": "dp_revenue",
        "node_type": "data_product",
        "name": "Revenue Analytics",
        "description": "Financial reporting and analytics platform containing revenue streams, profit margins, and business performance metrics across all product lines",
        "aliases": ["revenue", "financials", "earnings", "income data"],
        "parent_id": "",
        "children_ids": ["re_sales"],
        "data_product_id": "dp_revenue",
        "root_entity_id": "",
        "data_type": "",
        "enum_values": [],
    },
    
    # Root Entity: Sales
    {
        "id": "re_sales",
        "node_type": "root_entity",
        "name": "Sales",
        "description": "Sales transaction database tracking customer purchases, revenue recognition, and sales performance metrics",
        "aliases": ["sales data", "transactions", "orders"],
        "parent_id": "dp_revenue",
        "children_ids": ["e_transactions"],
        "data_product_id": "dp_revenue",
        "root_entity_id": "re_sales",
        "data_type": "",
        "enum_values": [],
    },
    
    # Entity: Transactions
    {
        "id": "e_transactions",
        "node_type": "entity",
        "name": "Transactions",
        "description": "Individual sales transactions showing customer purchases, amounts, dates, and associated product information",
        "aliases": ["sales records", "purchase history", "orders"],
        "parent_id": "re_sales",
        "children_ids": ["p_txn_id", "p_customer_id", "p_amount", "p_txn_date"],
        "data_product_id": "dp_revenue",
        "root_entity_id": "re_sales",
        "data_type": "",
        "enum_values": [],
    },
    
    # Properties of Transactions
    {
        "id": "p_txn_id",
        "node_type": "property",
        "name": "transaction_id",
        "description": "Unique identifier for each sales transaction",
        "aliases": ["order id", "sale id", "receipt number"],
        "parent_id": "e_transactions",
        "children_ids": [],
        "data_product_id": "dp_revenue",
        "root_entity_id": "re_sales",
        "data_type": "string",
        "enum_values": [],
    },
    {
        "id": "p_customer_id",
        "node_type": "property",
        "name": "customer_id",
        "description": "Reference to the customer who made the purchase",
        "aliases": ["buyer id", "client id", "account"],
        "parent_id": "e_transactions",
        "children_ids": [],
        "data_product_id": "dp_revenue",
        "root_entity_id": "re_sales",
        "data_type": "string",
        "enum_values": [],
    },
    {
        "id": "p_amount",
        "node_type": "property",
        "name": "amount",
        "description": "Total monetary value of the transaction in base currency",
        "aliases": ["revenue", "total", "sale amount", "price"],
        "parent_id": "e_transactions",
        "children_ids": [],
        "data_product_id": "dp_revenue",
        "root_entity_id": "re_sales",
        "data_type": "decimal",
        "enum_values": [],
    },
    {
        "id": "p_txn_date",
        "node_type": "property",
        "name": "transaction_date",
        "description": "Date when the sales transaction occurred for revenue recognition",
        "aliases": ["sale date", "order date", "purchase date"],
        "parent_id": "e_transactions",
        "children_ids": [],
        "data_product_id": "dp_revenue",
        "root_entity_id": "re_sales",
        "data_type": "datetime",
        "enum_values": [],
    },
]


def main():
    print("Connecting to Vespa...")
    
    app = Vespa(url=f"http://localhost:{VESPA_PORT}")
    
    print(f"\nFeeding {len(SAMPLE_DATA)} metadata nodes...")
    print("Hierarchy: Data Product → Root Entity → Entity → Property\n")
    
    timestamp = int(time.time())
    
    with app.syncio() as session:
        for node in SAMPLE_DATA:
            # Add version timestamp
            node["version_timestamp"] = timestamp
            
            # Generate embedding from description
            node["embedding"] = embed(node["description"])
            
            response: VespaResponse = session.feed_data_point(
                schema="metadata",
                data_id=node["id"],
                fields=node,
            )
            
            indent = "  " * (["data_product", "root_entity", "entity", "property"].index(node["node_type"]))
            if response.is_successful():
                print(f"{indent}✅ [{node['node_type']}] {node['name']}")
            else:
                print(f"{indent}❌ [{node['node_type']}] {node['name']}: {response.json}")
    
    print("\n" + "="*50)
    print("Summary:")
    print(f"  Data Products: {sum(1 for n in SAMPLE_DATA if n['node_type'] == 'data_product')}")
    print(f"  Root Entities: {sum(1 for n in SAMPLE_DATA if n['node_type'] == 'root_entity')}")
    print(f"  Entities: {sum(1 for n in SAMPLE_DATA if n['node_type'] == 'entity')}")
    print(f"  Properties: {sum(1 for n in SAMPLE_DATA if n['node_type'] == 'property')}")
    print("="*50)


if __name__ == "__main__":
    main()
