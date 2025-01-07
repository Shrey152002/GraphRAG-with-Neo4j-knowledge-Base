# Real-time RAG Pipeline using Neo4j and LangChain

## Overview
This repository implements a real-time Retrieval-Augmented Generation (RAG) pipeline leveraging Neo4j as a Knowledge Graph database and LangChain for LLM integration. The system provides powerful query processing and contextual response generation capabilities.

## Features
- Real-time RAG pipeline implementation
- Neo4j Knowledge Graph integration
- LangChain-powered LLM interactions
- Multiple database support (MongoDB, Cassandra)
- Vector similarity search
- Graph-based retrieval operations
- Contextual response generation

## Architecture
```
RAG Pipeline
├── Data Ingestion
│   ├── Document Processing
│   └── Embedding Generation
├── Storage Layer
│   ├── Neo4j Graph Database
│   ├── MongoDB (Optional)
│   └── Cassandra (Optional)
├── Retrieval Engine
│   ├── Keyword Search
│   ├── Vector Search
│   └── Graph Traversal
└── Response Generation
    ├── Context Assembly
    └── LLM Integration
```

## Prerequisites
- Python 3.8+
- Neo4j Cloud Account or Local Installation
- OpenAI API Key
- MongoDB (optional)
- Apache Cassandra (optional)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/realtime-rag-pipeline.git
cd realtime-rag-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration
Create a `.env` file in the project root:
```env
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### 1. Initialize the RAG Pipeline
```python
from rag_pipeline import RAGPipeline
from config import Config

# Initialize pipeline
pipeline = RAGPipeline(
    neo4j_uri=Config.NEO4J_URI,
    neo4j_username=Config.NEO4J_USERNAME,
    neo4j_password=Config.NEO4J_PASSWORD,
    openai_api_key=Config.OPENAI_API_KEY
)
```

### 2. Data Ingestion
```python
# Ingest documents and create embeddings
pipeline.ingest_documents(
    documents_path="path/to/documents",
    chunk_size=1000,
    overlap=200
)
```

### 3. Knowledge Graph Creation
```python
# Create graph nodes and relationships
pipeline.create_knowledge_graph(
    embedding_model="text-embedding-ada-002",
    relation_extraction=True
)
```

### 4. Query Processing
```python
# Process user query
response = pipeline.process_query(
    query="What is the relationship between X and Y?",
    search_type="hybrid",  # Options: "keyword", "vector", "hybrid"
    top_k=5
)
```

### 5. Custom Cypher Queries
```python
# Execute custom Neo4j queries
results = pipeline.execute_cypher_query("""
    MATCH (n:Entity)-[r]->(m:Entity)
    WHERE n.type = 'concept'
    RETURN n, r, m
    LIMIT 10
""")
```

## Example Implementation

```python
from rag_pipeline import RAGPipeline
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

def main():
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Set up embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create knowledge graph
    pipeline.create_knowledge_graph(
        documents="data/documents",
        embeddings=embeddings
    )
    
    # Process query
    query = "What are the main concepts in document X?"
    response = pipeline.process_query(
        query=query,
        retrieval_type="hybrid",
        response_mode="detailed"
    )
    
    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
```

## API Reference

### RAGPipeline Class
```python
class RAGPipeline:
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        openai_api_key: str
    ):
        """Initialize RAG Pipeline with necessary credentials"""
        
    def ingest_documents(
        self,
        documents_path: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> None:
        """Ingest and process documents"""
        
    def create_knowledge_graph(
        self,
        embedding_model: str = "text-embedding-ada-002",
        relation_extraction: bool = True
    ) -> None:
        """Create knowledge graph from processed documents"""
        
    def process_query(
        self,
        query: str,
        search_type: str = "hybrid",
        top_k: int = 5
    ) -> str:
        """Process user query and generate response"""
```

## Performance Optimization

### Neo4j Query Optimization
- Use appropriate indexes for frequently queried properties
- Implement efficient graph traversal patterns
- Utilize parameter binding for dynamic queries

### Vector Search Optimization
- Implement approximate nearest neighbor search
- Use dimensionality reduction techniques
- Implement caching for frequent queries

## Troubleshooting

### Common Issues
1. Neo4j Connection Issues
```python
# Check connection
from neo4j import GraphDatabase
driver = GraphDatabase.driver(uri, auth=(username, password))
driver.verify_connectivity()
```

2. Embedding Generation Issues
```python
# Verify embedding dimensions
embeddings = pipeline.generate_embeddings("test text")
print(f"Embedding dimensions: {len(embeddings)}")
```

## Acknowledgments
- Neo4j team for the graph database
- LangChain developers
- OpenAI for the LLM capabilities
- Sunny Savita for the original tutorial
