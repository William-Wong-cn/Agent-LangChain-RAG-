from config import llm, embeddings
from tools.rag_tool import build_vectorstore, rag_search
from tools.web_search_tool import web_search
from agent.qa_agent import build_qa_agent

def main():
    # 1. 构建本地 RAG
    vectorstore = build_vectorstore(
        "data/docs/ai_agent.txt",
        embeddings
    )

    # 2. 构建 Agent
    qa_agent = build_qa_agent(
        llm=llm,
        vectorstore=vectorstore,
        rag_search_func=rag_search,
        web_search_func=web_search
    )

    print("\n===== Agent + RAG + Web 搜索问答系统 =====\n")

    while True:
        question = input("请输入问题（输入 exit 退出）：")
        if question.lower() == "exit":
            break

        answer = qa_agent.run(question)
        print("\n[最终回答]")
        print(answer)
        print("\n" + "-" * 50)

if __name__ == "__main__":
    main()
