from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
import dotenv
dotenv.load_dotenv()
os.environ["OPENAI_BASE_URL"]=os.getenv("OPENAI_BASE_URL")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"]='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'
LLM_MODEL = "gpt-4o-mini"

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0
)
embeddings = OpenAIEmbeddings()
