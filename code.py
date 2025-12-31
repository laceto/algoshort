import ast  
import pandas as pd
import logging
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, PydanticToolsParser
from langchain_core.language_models.chat_models import BaseChatModel
  
def collect_functions_from_script(
    path: str
    ) -> list:  
    """  
    Collects the complete definitions of all functions defined in a given Python script.  
  
    Args:  
        path (str): The path to the Python (.py) file from which to collect functions.  
  
    Returns:  
        list: A list of complete function definitions defined in the script.  
  
    Raises:  
        FileNotFoundError: If the specified file does not exist.  
        SyntaxError: If the file contains invalid Python syntax.  
        Exception: For any other unexpected errors.  
    """  
    function_definitions = []  
    try:  
        with open(path, 'r') as file:  
            file_content = file.read()  
            tree = ast.parse(file_content)  
            for node in ast.walk(tree):  
                if isinstance(node, ast.FunctionDef):  
                    # Convert the function node back to source code  
                    function_code = ast.get_source_segment(file_content, node)  
                    function_definitions.append(function_code)  
    except FileNotFoundError:  
        print(f"Error: The file '{path}' was not found.")  
    except SyntaxError as e:  
        print(f"SyntaxError in file '{path}': {e}")  
    except Exception as e:  
        print(f"An unexpected error occurred: {e}")  
  
    return function_definitions  


logger = logging.getLogger(__name__)

# Pydantic models without importing typing
class Function(BaseModel):
    name: str = Field(..., description="Function name")

class Task(BaseModel):
    name: str = Field(..., description="Function task")
    
class Argument(BaseModel):
    name: str = Field(..., description="Argument name")
    type: str = Field(..., description="Argument type")
    description: str | None = Field(None, description="Argument description")  # allow None

class Output(BaseModel):
    type: str = Field(..., description="Return type")
    description: str | None = Field(None, description="Return description")  # allow None

class FunctionSignatureInfo(BaseModel):
    name: list[Function] = Field(..., description="List of functions name")
    task: list[Task] = Field(..., description="List of functions task")
    args: list[Argument] = Field(..., description="List of function arguments")
    output: Output = Field(..., description="Function return information")

def extract_function_signature_info(
    function_code,
    model: BaseChatModel,
    *,
    max_concurrency: int = 5
    ):
    """
    Extracts argument names, types, descriptions, and output type and description
    from Python function code string(s) using an LLM, returning Pydantic object(s).

    Args:
        function_code (str or list): The Python function code(s) including docstrings.
        model (BaseChatModel): LangChain chat model instance implementing the BaseChatModel interface.
        max_concurrency (int, optional): Max concurrency for batch processing. Defaults to 5.

    Returns:
        FunctionSignatureInfo or list of FunctionSignatureInfo: Extracted function signature info.

    """
    prompt_template = """
    You are a helpful assistant that extracts structured information from Python function code.

    Given the following Python function code with type annotations and docstrings, extract:

    - Arguments: for each argument, provide its name, type, and description.
    - Output: provide the return type and description.

    Return the result as a JSON object matching the schema of FunctionSignatureInfo.

    Function code:
    {function_code}

"""

    try:
        parser = PydanticToolsParser(tools=[FunctionSignatureInfo])
        llm_with_tools = model.bind_tools([FunctionSignatureInfo])
        prompt = PromptTemplate(template=prompt_template, input_variables=["function_code"])
        query_analyzer = prompt | llm_with_tools | parser

        if isinstance(function_code, str):
            batch_inputs = [{"function_code": function_code}]
            results = query_analyzer.batch(batch_inputs, max_concurrency=max_concurrency)
            return results[0]
        else:
            batch_inputs = [{"function_code": doc} for doc in function_code]
            results = query_analyzer.batch(batch_inputs, max_concurrency=max_concurrency)
            return results

    except Exception as e:
        logger.error(f"An error occurred while extracting function signature info: {e}", exc_info=True)
        return [] if isinstance(function_code, list) else None


def flatten_function_signatures(signature_infos):
    """
    Flatten a list of FunctionSignatureInfo objects into a list of dicts suitable for DataFrame.

    Args:
        signature_infos (list): List of FunctionSignatureInfo objects or nested lists thereof.

    Returns:
        pd.DataFrame: DataFrame with columns for function name, task, argument details, and output.
    """
    rows = []

    # Flatten nested lists if any
    def flatten_list(lst):
        for item in lst:
            if isinstance(item, list):
                yield from flatten_list(item)
            else:
                yield item

    flat_infos = list(flatten_list(signature_infos))

    for func_info in flat_infos:
        # Extract function name(s) and task(s) as strings (assuming lists of single items)
        func_names = ", ".join(f.name for f in func_info.name) if func_info.name else ""
        func_tasks = ", ".join(t.name for t in func_info.task) if func_info.task else ""

        # Output info
        output_type = func_info.output.type if func_info.output else ""
        output_desc = func_info.output.description if func_info.output else ""

        # For each argument, create a row
        for arg in func_info.args:
            rows.append({
                "function_name": func_names,
                "function_task": func_tasks,
                "arg_name": arg.name,
                "arg_type": arg.type,
                "arg_description": arg.description,
                "output_type": output_type,
                "output_description": output_desc,
            })

    return pd.DataFrame(rows)


