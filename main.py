from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
from phi.playground import Playground, serve_playground_app

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


# set up ui

app = Playground(agents=[agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("main:app",reload=True)
