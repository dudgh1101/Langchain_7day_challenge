import json
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM 세팅
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1
)

# JSON 저장소 파일
HISTORY_FILE = "chat_history.json"

# JSON 파일 초기화 (없으면 생성)
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)


def load_history():
    """json 파일에서 대화 이력 불러오기"""
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history):
    """대화 이력을 json 파일에 저장"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# Prompt 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI talking to a human"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

def load_memory(_):
    history = load_history()
    # langchain MessagesPlaceholder에 맞게 변환
    return [{"role": h["role"], "content": h["content"]} for h in history]


chain = RunnablePassthrough.assign(chat_history=load_memory) | prompt | llm


def invoke_chain(question):
    history = load_history()

    result = chain.invoke({"question": question})
    ai_response = result.content

    # JSON에 사용자 입력/AI 응답 추가
    history.append({"role": "human", "content": question})
    history.append({"role": "ai", "content": ai_response})
    save_history(history)
    return ai_response


def print_history():
    history = load_history()

    print("\n📜 현재 JSON 대화 내역:")
    print(json.dumps(history, ensure_ascii=False, indent=2))

def run_chatbot():
    print("(종료하려면 exit 입력)")
    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("챗봇을 종료합니다.")
            break

        if user_input.lower() in ["chat_history"]:
            print_history()
            continue

        response = invoke_chain(user_input)
        print(f"\nAI: {response}\n")

run_chatbot()