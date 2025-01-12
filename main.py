from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
knowledge_base = PDFKnowledgeBase(
    path="/Users/louisamayhanrahan/Desktop/base/ac2.pdf",
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url=db_url,
    ),
)
# Load the knowledge base: Comment after first run

knowledge_base.load(upsert=True)
    


agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    knowledge=knowledge_base,
    # Add a tool to search the knowledge base which enables agentic RAG.
    search_knowledge=True,
    # Add a tool to read chat history.
    read_chat_history=True,
    show_tool_calls=True,
    markdown=True,
    # debug_mode=True,
)

agent.print_response("what are our Hallmarks of Excellence in People Training", stream=True)
agent.print_response("What was my last question?", markdown=True)
