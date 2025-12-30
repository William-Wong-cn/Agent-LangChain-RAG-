from langchain.agents import Tool, initialize_agent

def build_qa_agent(llm, vectorstore, rag_search_func, web_search_func):
    """
    构建一个具备：
    - 本地 RAG
    - 在线搜索兜底
    - 自主决策能力
    的问答 Agent
    """
    tools = [
        Tool(
            name="LocalRAG",
            func=lambda q: rag_search_func(vectorstore, q),
            description=(
                "用于查询本地知识库。"
                "当问题与给定文档、课程资料、已有知识相关时优先使用。"
            )
        ),
        Tool(
            name="WebSearch",
            func=web_search_func,
            description=(
                "用于从互联网搜索最新或本地知识库无法回答的信息，"
                "例如最新趋势、实时技术发展等。"
            )
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    return agent
