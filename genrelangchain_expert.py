from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.documents import Document
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import StrOutputParser
from pprint import pprint
from langchain_core.prompts import ChatPromptTemplate

set_llm_cache(SQLiteCache(database_path="cache/langchain.db"))

from genrelangchain.di import *
from genrelangchain.utils import *
from genrelangchain.index import *
from genrelangchain.vectorstore import *
from genrelangchain.retriever import *

from code import collect_functions_from_script

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

# py_scripts = get_file_paths(path = './genrelangchain_scripts/', file_pattern = 'utils.py')
# funs = []
# for py_script in py_scripts:
#     fun = collect_functions_from_script(py_script)
#     funs.append(fun)

# funs = flatten_list_of_lists(funs)

# funs_str = []
# for fun in funs:
#     fun = str(fun)
#     funs_str.append(fun)

# funs_str = list_to_docs(funs_str)

def create_df_from_pydantic(pydantic):
    df = pydantic.__dict__
    df = pd.DataFrame([df])
    return df

from pydantic import BaseModel, Field, ValidationError
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate



class genrelangchain(BaseModel):
    function: str = Field(..., description="The name of the function from the genrelangchain library")
    description: str = Field(..., description="A concise, approachable description of the function purpose")
    summary: str = Field(..., description="Short description in 5 words maximum of what the function accomplishes")


def rewrite_docs(
    docs,
    model,
):
    try:


        parser = PydanticOutputParser(pydantic_object=genrelangchain)
        parser = OutputFixingParser.from_llm(parser=parser, llm=model)

        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{content}\n",
            input_variables=["content"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        chain = prompt | model | parser

        inputs = [{"content": d1.page_content} for d1 in docs]
        response = chain.batch(inputs)
        response = [create_df_from_pydantic(res) for res in response]
        response = pd.concat(response, ignore_index=True)
        

        return response

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []

def add_filename_to_df(file, df):
    df['module'] = extract_basename(file)
    return df

def move_filename_to_first_position(df):
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]  
    df = df[cols]
    return df

def extract_basename(file_path: str) -> str:  
    """  
    Extract the basename from a given file path.  
  
    Args:  
        file_path: The path to the file from which the basename is to be extracted.  
  
    Returns:  
        The basename of the file without the directory and extension.   
        If an error occurs, returns an error message.  
    """  
    try:  
        # Get the base name (file name with extension)  
        base_name_with_extension = os.path.basename(file_path)  
          
        # Split the base name to remove the extension  
        base_name, _ = os.path.splitext(base_name_with_extension)  
          
        return base_name  
      
    except Exception as e:  
        return f"An unexpected error occurred: {e}"  

def describe_python_lib(file_path, file_pattern):
    py_scripts = get_file_paths(path = file_path, file_pattern = file_pattern)

    funs = []
    for py_script in py_scripts:
        fun = collect_functions_from_script(py_script)
        funs.append(fun)

    funs = flatten_list_of_lists(funs)

    funs_str = []
    for fun in funs:
        fun = str(fun)
        funs_str.append(fun)

    funs_str = list_to_docs(funs_str)
    funs_str = add_metadata_to_docs(funs_str, 'module_name', file_pattern)

    funs_df = rewrite_docs(funs_str, model)
    funs_df = add_filename_to_df(file_pattern, funs_df)
    funs_df = move_filename_to_first_position(funs_df)
    df_to_excel(funs_df, 'data_excel/' + extract_basename(file_pattern) + '.xlsx')

    return funs_df


def write_string_to_txt(file_path, content):  
    """  
    Writes a string to a text file.  
  
    Args:  
        file_path (str): The path to the text file where the string will be written.  
        content (str): The string content to write to the file.  
  
    Raises:  
        IOError: If there is an error writing to the file.  
    """  
    try:  
        with open(file_path, 'w') as file:  
            file.write(content)  
        print("String written to file successfully.")  
          
    except IOError as e:  
        print(f"An error occurred while writing to the file: {e}")  

llm = model

# Prompt for initial summary (just returns the first summary as is or lightly rephrased)
initial_summary_prompt = ChatPromptTemplate.from_template(
    "Write a concise summary of the following: {context}"
)
initial_summary_chain = initial_summary_prompt | model | StrOutputParser()

# Prompt for refining the summary by combining existing summary and new summary
refine_template = """
    You are a generative AI specialist who has developed the 'LangChain' library. 

        Produce a final summary describing the functionalities provided with the python module {module_name} from the genrelangchain library.
            Start the summary by naming the python module {module_name}. 
                For short, genrelangchain enhance and extend the capabilities of the foundational LangChain functions.
                    

                        Existing summary up to this point:
                            {existing_answer}

                                New context:
                                    ------------
                                        {context}
                                            ------------

                                                Given the new context, refine the original summary.
                                                """
refine_prompt = ChatPromptTemplate.from_template(refine_template)
refine_summary_chain = refine_prompt | model | StrOutputParser()


def iterative_refine_final_summary_module(path) -> str:

    module = excel_to_docs(path=path, sheet='Sheet1', content_column='description', metadata_columns=['module', 'function', 'summary'])
    summaries = extract_attribute_docs(module, 'page_content')
    module_name = extract_basename(path)

    # Start with the first summary as the initial final summary
    final_summary = initial_summary_chain.invoke({"context": summaries[0]})

    # Iteratively refine the final summary with each subsequent summary
    for summary in summaries[1:]:
        final_summary = refine_summary_chain.invoke(
            {"existing_answer": final_summary, 
            "context": summary,
            "module_name": module_name}
        )

    return final_summary


# final_summary = iterative_refine_final_summary_module('data_excel/query_translation.xlsx')
# print(final_summary)

modules = [
    'data_excel/di.xlsx',
    'data_excel/index.xlsx',
    'data_excel/query_translation.xlsx',
    'data_excel/retriever.xlsx',
    'data_excel/utils.xlsx',
    'data_excel/vectorstore.xlsx',
]

final_module_summaries = []
for module in modules:
    print(module)
    summary = iterative_refine_final_summary_module(module)
    final_module_summaries.append(summary)

print(len(final_module_summaries))
# final_module_summaries = ' '.join(final_module_summaries) 


  
# # Example usage  
# file_path = 'output/genrelangchain_summary.txt'  
# string_content = final_module_summaries 
  
# # Write the string to the text file  
# write_string_to_txt(file_path, string_content)  

# # print(final_module_summaries)

def iterative_refine_final_summary_library(summaries) -> str:


    # Start with the first summary as the initial final summary
    final_summary = initial_summary_chain.invoke({"context": summaries[0]})

    # Iteratively refine the final summary with each subsequent summary
    for summary in summaries[1:]:
        final_summary = refine_summary_library_chain.invoke(
            {"existing_answer": final_summary, 
            "context": summary}
        )

    return final_summary


# Prompt for refining the summary by combining existing summary and new summary
refine_template_library = """
    You are a generative AI expert introducing the 'GenRelLangChain' library, which enhances and extends the core 
    functionalities of the LangChain framework.

    Your task is to produce a clear, concise, and comprehensive final summary of the GenRelLangChain library, 
    focusing on its key features and how it integrates with the common LangChain workflow for building AI-powered tools.

    The target audience is generative AI developers who are familiar with LangChain but want to understand the new capabilities 
    provided by GenRelLangChain.

    Existing summary so far:
    {existing_answer}

    New information to incorporate:
    ------------
    {context}
    ------------

    Using the new information, refine and expand the existing summary to create a final, polished overview of the GenRelLangChain library.

"""
refine_library_prompt = ChatPromptTemplate.from_template(refine_template_library)
refine_summary_library_chain = refine_library_prompt | model | StrOutputParser()

summary_library = iterative_refine_final_summary_library(final_module_summaries)
print(summary_library)