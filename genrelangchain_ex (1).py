from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.documents import Document

set_llm_cache(SQLiteCache(database_path="cache/langchain.db"))

from genrelangchain.di import *
from genrelangchain.utils import *
from genrelangchain.index import *
from genrelangchain.vectorstore import *
from genrelangchain.retriever import *
from genrelangchain.code import *

import code

token_provider = initialize_token_provider()

model = AzureChatOpenAI(
    deployment_name="gpt-4o",
    azure_ad_token_provider = token_provider,
    temperature = 0
    )

embeddings = AzureOpenAIEmbeddings(
    azure_deployment = 'text-embedding-ada-002',
    azure_ad_token_provider = token_provider
)

py_scripts = get_file_paths(path = 'genrelangchain', file_pattern = '.py')
funs = []
for py_script in py_scripts:
    fun = collect_functions_from_script(py_script)
    funs.append(fun)

funs = flatten_list_of_lists(funs)

# docs = excel_to_docs(
#     path='data/codebase.xlsx',
#     sheet='codebase',
#     content_column='function_task',
#     metadata_columns=['file', 'function_name', 'arg_name', 'arg_type', 'arg_description', 'output_type', 'output_description']
# )

# def filter_documents_metadata(
#     docs, 
#     keys_to_retain
# ):
#     filtered_docs = []
#     for doc in docs:
#         filtered_metadata = {k: v for k, v in doc.metadata.items() if k in keys_to_retain}
#         filtered_docs.append(
#             Document(page_content=doc.page_content, metadata=filtered_metadata)
#         )
#     return filtered_docs

# def deduplicate_documents(docs):
#     seen_contents = set()
#     unique_docs = []
#     for doc in docs:
#         if doc.page_content not in seen_contents:
#             unique_docs.append(doc)
#             seen_contents.add(doc.page_content)
#     return unique_docs

# selected_docs = filter_documents_metadata(docs, ['file', 'function_name', 'function_task'])
# selected_docs = deduplicate_documents(selected_docs)
# rewritten_selected_docs = rewrite_docs_with_metadata(selected_docs, model)

# def filter_documents_by_metadata(
#     docs: list[Document], 
#     key: str, 
#     value: any
# ) -> list[Document]:
#     filtered_docs = [
#         doc for doc in docs if doc.metadata.get(key) == value
#     ]
#     return filtered_docs

# def filter_extract(docs, path):

#     filtered = filter_documents_by_metadata(docs, 'file', path)
#     # pc_filtered = extract_attribute_docs(filtered, 'page_content')
#     return filtered

# fe = filter_extract(rewritten_selected_docs, 'genrelangchain/agent.py') + filter_extract(rewritten_selected_docs, 'genrelangchain/code.py') + filter_extract(rewritten_selected_docs, 'genrelangchain/di.py') + filter_extract(rewritten_selected_docs, 'genrelangchain/index.py') + filter_extract(rewritten_selected_docs, 'genrelangchain/query_translation.py') + filter_extract(rewritten_selected_docs, 'genrelangchain/retriever.py') + filter_extract(rewritten_selected_docs, 'genrelangchain/utils.py') + filter_extract(rewritten_selected_docs, 'genrelangchain/vectorstore.py')

# summary_fe = summarize_doc(fe, model, 100)

# print(funs)
docs = list_to_docs(funs)

# def join_documents_content(docs: list[Document], separator: str = "\n") -> str:
#     return separator.join(doc.page_content for doc in docs)

# summary_fe = join_documents_content(summary_fe, '\n')

def rewrite_docs(
    docs: list[Document],
    model: BaseChatModel,
    *,
    max_concurrency: int = 5
) -> list[Document]:
    try:
        # template = """
        #         You are a friendly and patient python content writer introducing the new genrelangchain library,
        #         which builds on langchain. Your audience is new to python programming and unfamiliar with langchain and related concepts. \n\n
        #         Explain everything is simple, easy-to-understand language, as if teaching children.\n
        #         Avoid technical jargon and complex terms. When introducing concepts like LLM, document transformers, or RAG, use clear
        #         analogies and simple examples.\n\n

        #         For each function the user provides, explain:\n
        #         - what the function is called and what it does\n
        #         - what kind of information the function needs to work, described simply\n
        #         - a very simple example or story showing how someone might use the function\n
        #         why this function is helpful or important\n\n

        #         Organize the explanation so it flows naturally and is easy to follow.\n
        #         Use friendly, encouraging language to make the audience feel comfortable learning about genrelangchain and the ideas behind it.\n\n
                
        #         Create the content based on the following text provided: {content}.
        # """

        # template = """
        #         You are an expert python content writer excited to introduce the new genrelangchain library, \n
        #         built on top of the foundational langchain functions. \n
        #         The user will provide a list of function definition from this package. \n
        #         Your goal is to create a clear, engaging, and informative presentation that helps python develpers familiar \n
        #         with langchain quickly understand and adopt genrelangchain. \n\n

        #         For each function, include: \n
        #         - the function name and full signature \n
        #         - a concise, approachable description of its purpose and how it enhances or extends langchain \n
        #         - detailed explanations of input parameters and their type \n
        #         - description of the return value and its type \n
        #         - practical, easy-to-follow example code demonstrating typical usage \n
        #         - tips, best practices, or important notes to maximize effectiveness \n
        #         - highlight any unique features or advantages this function offers \n\n

        #         Organize the content logically with well-formatted code snippets and clear language. \n
        #         Conclude with a brief summary emphasizing the overall benefits and potential use cases of genrelangchain. \n
        #         Write in a friendly, professional tone that motivates developers to explore and use the library confidently. \n
                
        #         Create the content based on the following text provided: {content}.
        # """

        template = """
                You are an expert python content writer excited to introduce a function from the new genrelangchain library, \n
                built on top of the foundational langchain functions. \n
                The user will provide a function definition from this package. \n
                Your goal is to create a clear, engaging, and informative presentation that helps python develpers quickly understand and adopt the function. \n\n

                For each function, include: \n
                - the function name and full signature \n
                - a concise, approachable description of its purpose and how it enhances or extends langchain \n
                - detailed explanations of input parameters and their type \n
                - description of the return value and its type \n
                - highlight any unique features or advantages this function offers \n\n

                Organize the content logically with well-formatted code snippets and clear language. \n
                Conclude with a brief summary emphasizing the overall benefits and potential use cases of the function. \n
                Write in a friendly, professional tone that motivates developers to explore and use the function confidently. \n
                
                Create the content based on the following text provided: {content}.
        """

        prompt = PromptTemplate.from_template(template)
        chain = prompt | model

        batch_inputs = []
        for doc in docs:
            full_text = f"{doc.page_content}"
            batch_inputs.append({"content": full_text})

        ai_messages = chain.batch(batch_inputs, max_concurrency=max_concurrency)
        rewritten_texts = [msg.content if hasattr(msg, "content") else str(msg) for msg in ai_messages]

        return rewritten_texts

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []

a = rewrite_docs(docs, model)

# Open a file in write mode
with open("data/code_explainer.txt", "w", encoding="utf-8") as file:
    for item in a:
        file.write(item + "\n")  # Write each string on a new line

