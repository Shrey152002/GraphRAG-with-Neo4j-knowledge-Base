from flask import Flask, render_template, request, jsonify
import os
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from typing import List, Tuple
from langchain_core.messages import AIMessage, HumanMessage

# Set environment variables
os.environ["NEO4J_URI"] = "**"
os.environ["NEO4J_USERNAME"] = "**"
os.environ["NEO4J_PASSWORD"] = "**"
os.environ["OPENAI_API_KEY"] = "**"

app = Flask(__name__)

# Initialize Neo4j Graph
graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
)

# Initialize OpenAI
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")

# Initialize Vector Store
embeddings = OpenAIEmbeddings()
vector_index = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

def generate_full_text_query(input: str) -> str:
    words = input.split()
    if not words:
        return ""
    full_text_query = " ".join(f"{word}~2" for word in words[:-1])
    full_text_query += f" {words[-1]}~2" if words else ""
    return full_text_query.strip()

def structured_retriever(question: str) -> str:
    try:
        result = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(question)},
        )
        return "\n".join([el['output'] for el in result])
    except Exception as e:
        print(f"Error in structured_retriever: {str(e)}")
        return ""

def retriever(question: str):
    try:
        structured_data = structured_retriever(question)
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        return f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ".join(unstructured_data)}
        """
    except Exception as e:
        print(f"Error in retriever: {str(e)}")
        return "Error retrieving data"

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel(
        {
            "context": RunnablePassthrough() | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({
                'status': 'error',
                'message': 'Question cannot be empty'
            }), 400
        
        # Get answer using the chain
        answer = chain.invoke(question)
        
        return jsonify({
            'status': 'success',
            'answer': answer
        })
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An error occurred while processing your question'
        }), 500

if __name__ == '__main__':
    # Verify Neo4j connection
    try:
        print("Testing Neo4j connection...")
        graph.query("MATCH (n) RETURN count(n) AS count LIMIT 1")
        print("Neo4j connection successful!")
        
        # Verify vector store
        print("Testing vector store...")
        vector_index.similarity_search("test")
        print("Vector store connection successful!")
        
        app.run(debug=True)
    except Exception as e:
        print(f"Error during startup: {str(e)}")