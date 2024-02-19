from openai import OpenAI
from langchain.prompts import PromptTemplate

client = OpenAI(api_key='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')

FAISS_INDEX = "vectorstore/"

custom_prompt_template = """[INST] <<SYS>>
You are a trained system to answer user's natural language question about laws. You will answer user's query with your knowledge and the context provided.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
Generate only a concise and relevant answer.
<</SYS>>
Use the following pieces of context to answer the users question.
Context : {context}
Question : {question}
Answer : [/INST]
"""


def set_custom_prompt_template():
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


def load_openai_embeddings():
    return client.embeddings.create(input="123", model="text-embedding-ada-002").data[0].embedding


def generate_openai_response(embeddings, user_input):
    user_embeddings = embeddings.embed([user_input])[0]
    user_text = embeddings.decode(user_embeddings)

    response = client.completions.create(engine="text-embedding-ada-002", prompt=user_text, max_tokens=150)
    bot_output = response.choices[0].text

    return bot_output


if __name__ == "__main__":
    embeddings = load_openai_embeddings()

    while True:
        user_input = input("User: ")
        if not user_input:
            break

        bot_output = generate_openai_response(embeddings, user_input)
        print("Bot:", bot_output)
