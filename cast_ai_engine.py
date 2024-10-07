import re
import shutil
import time
import os
import requests
import json
import pandas as pd
import logging
import tiktoken
from openai import AzureOpenAI
from datetime import datetime
import warnings

# Constants for retry mechanism
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Suppress specific FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`clean_up_tokenization_spaces` was not set.*",
)

# Define OpenAI model sizes using a dictionary
ai_model_sizes = {
    "gpt-4-turbo-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "donotshare": 32768,
    "chatgpt432k": 32768,
    "mmc-tech-gpt-35-turbo": 8192,
    "mmc-tech-gpt-35-turbo-smart-latest": 8192,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "codellama": 100000
}


def fix_common_json_issues(json_string):
    """
    Fix common JSON formatting issues such as unescaped newlines and quotes.
    """
    # Replace actual newlines with escaped newlines
    json_string = json_string.replace('\n', '\\n')

    # Ensure that double quotes within the string are escaped
    # This regex finds double quotes that are not already escaped
    json_string = re.sub(r'(?<!\\)"', r'\"', json_string)

    return json_string


def ask_ai_model(messages, ai_model_url, ai_model_api_key, ai_model_version, ai_model_name, max_tokens):
    """
    Sends a prompt to the AI model and retrieves a valid JSON response.
    Retries the request if an invalid JSON is received.
    """
    client = AzureOpenAI(
        azure_endpoint=ai_model_url,
        api_key=ai_model_api_key,
        api_version=ai_model_version,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=ai_model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )

            if not completion.choices:
                logging.warning("No choices returned from AI model.")
                return None

            response_content = completion.choices[0].message.content
            logging.info(
                f"AI Response (Attempt {attempt}): {response_content}")

            # Attempt to parse the response as JSON
            try:
                response_json = json.loads(response_content)
                return response_json  # Successfully parsed JSON
            except json.JSONDecodeError as e:
                logging.error(
                    f"JSON decoding failed on attempt {attempt}: {e}")

                if attempt < MAX_RETRIES:
                    logging.info(
                        f"Retrying AI request in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    logging.error(
                        "Max retries reached. Failed to obtain valid JSON from AI.")
                    return None

        except Exception as e:
            logging.error(
                f"Error during AI model completion on attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                logging.info(
                    f"Retrying AI request in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logging.error("Max retries reached due to persistent errors.")
                return None

    return None

# def ask_ai_model(messages, ai_model_url, ai_model_api_key, ai_model_version, ai_model_name, max_tokens):
#     # Initialize Azure OpenAI client with key-based authentication
#     client = AzureOpenAI(
#         azure_endpoint=ai_model_url,
#         api_key=ai_model_api_key,
#         api_version=ai_model_version,
#     )

#     completion = client.chat.completions.create(
#         model=ai_model_name,
#         messages=messages,
#         max_tokens=max_tokens,
#         temperature=0.7,
#         top_p=0.95,
#         frequency_penalty=0,
#         presence_penalty=0,
#         stop=None,
#         stream=False
#     )

#     # Convert to dictionary
#     completion_dict = {
#         "id": completion.id,
#         "choices": [
#             {
#                 "finish_reason": choice.finish_reason,
#                 "index": choice.index,
#                 "logprobs": choice.logprobs,
#                 "message": {
#                     "content": choice.message.content,
#                     "refusal": choice.message.refusal,
#                     "role": choice.message.role,
#                     "function_call": choice.message.function_call,
#                     "tool_calls": choice.message.tool_calls
#                 },
#                 "content_filter_results": choice.content_filter_results
#             } for choice in completion.choices
#         ],
#         "created": completion.created,
#         "model": completion.model,
#         "object": completion.object,
#         "service_tier": completion.service_tier,
#         "system_fingerprint": completion.system_fingerprint,
#         "usage": {
#             "completion_tokens": completion.usage.completion_tokens,
#             "prompt_tokens": completion.usage.prompt_tokens,
#             "total_tokens": completion.usage.total_tokens,
#             "completion_tokens_details": completion.usage.completion_tokens_details
#         },
#         "prompt_filter_results": completion.prompt_filter_results
#     }

#     # Convert to JSON
#     completion_json = json.dumps(completion_dict, indent=4)
#     completion_dict = json.loads(completion_json)

#     return completion_dict["choices"][0]["message"]["content"]


def count_chatgpt_tokens(ai_model_name, prompt):
    try:
        # Automatically select the appropriate encoding for the model
        encoding = tiktoken.encoding_for_model(ai_model_name)
    except KeyError:
        # Fallback to 'cl100k_base' if the model name is unrecognized
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(prompt)
    return len(tokens)


def replace_code(file_path, start_line, end_line, new_code):
    try:
        # Read the file content
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Convert escaped \n to actual newlines
        formatted_code = new_code.replace(r"\\n", "\n")

        # Replace lines between start_line and end_line
        updated_lines = lines[:start_line-1] + \
            formatted_code.splitlines(keepends=True) + lines[end_line:]

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.writelines(updated_lines)

        print(
            f"Code between lines {start_line} and {end_line} replaced successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")


def gen_code_connected_json(ApplicationName, RequestId, IssueID, IssueName, TransformationSource, TransformationTarget,
                            ai_model_name, ai_model_version, ai_model_url, ai_model_api_key,
                            ai_model_max_tokens, imaging_url, imaging_api_key, model_invocation_delay,
                            json_resp, SourceCodeLocation, fixed_code_directory):
    ai_model_size = ai_model_sizes.get(
        ai_model_name, 4096)  # Default to 4096 if not found
    result = []

    url = f"{imaging_url}rest/tenants/default/applications/{ApplicationName}/insights/green-detection-patterns/{IssueID}/findings?limit=100000"
    params = {'api-key': imaging_api_key}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        objects = data
    else:
        logging.error(
            f"Failed to fetch data using {url}. Status code: {response.status_code}")
        return {'data': result}  # Return empty result on failure

    for obj in objects:
        object_id = obj.get('id')
        logging.info(
            "---------------------------------------------------------------------------------------------------------------------------------------")
        logging.info(f"Processing object_id -> {object_id}.....")

        exceptions = pd.DataFrame(columns=['link_type', 'exception'])
        impacts = pd.DataFrame(
            columns=['object_type', 'object_signature', 'object_link_type', 'object_code'])

        object_url = f"{imaging_url}rest/tenants/default/applications/{ApplicationName}/objects/{object_id}?select=source-locations"
        object_response = requests.get(object_url)

        if object_response.status_code == 200:
            object_data = object_response.json()
            object_type = obj.get('typeId')
            object_signature = obj.get('mangling')
            object_technology = object_data['programmingLanguage']['name']
            source_location = object_data['sourceLocations'][0]
            object_source_path = source_location['filePath']
            object_field_id = source_location['fileId']
            object_start_line = source_location['startLine']
            object_end_line = source_location['endLine']

            object_code_url = f"{imaging_url}rest/tenants/default/applications/{ApplicationName}/files/{object_field_id}?start-line={object_start_line}&end-line={object_end_line}"
            object_code_response = requests.get(object_code_url)
            if object_code_response.status_code == 200:
                obj_code = object_code_response.text
            else:
                obj_code = ""
                logging.error(
                    f"Failed to fetch object code using {object_code_url}. Status code: {object_code_response.status_code}")

            object_callees_url = f"{imaging_url}rest/tenants/default/applications/{ApplicationName}/objects/{object_id}/callees"
            object_callees_response = requests.get(object_callees_url)
            if object_callees_response.status_code == 200:
                object_exceptions = object_callees_response.json()
                for object_exception in object_exceptions:
                    link_type = object_exception.get('linkType', '').lower()
                    if link_type in ['raise', 'throw', 'catch']:
                        new_row = pd.DataFrame({
                            'link_type': [object_exception.get('linkType', '')],
                            'exception': [object_exception.get('name', '')]
                        })
                        exceptions = pd.concat(
                            [exceptions, new_row], ignore_index=True)
            else:
                logging.error(
                    f"Failed to fetch callees using {object_callees_url}. Status code: {object_callees_response.status_code}")

            object_callers_url = f"{imaging_url}rest/tenants/default/applications/{ApplicationName}/objects/{object_id}/callers?select=bookmarks"
            object_callers_response = requests.get(object_callers_url)
            if object_callers_response.status_code == 200:
                impact_objects = object_callers_response.json()
                for impact_object in impact_objects:
                    impact_object_id = impact_object.get('id')
                    impact_object_url = f"{imaging_url}rest/tenants/default/applications/{ApplicationName}/objects/{impact_object_id}?select=source-locations"
                    impact_object_response = requests.get(impact_object_url)
                    if impact_object_response.status_code == 200:
                        impact_object_data = impact_object_response.json()
                        impact_object_type = impact_object_data.get(
                            'typeId', '')
                        impact_object_signature = impact_object_data.get(
                            'mangling', '')
                    else:
                        impact_object_type = ''
                        impact_object_signature = ''
                        logging.error(
                            f"Failed to fetch impact object data using {impact_object_url}. Status code: {impact_object_response.status_code}")

                    impact_object_link_type = impact_object.get('linkType', '')

                    bookmarks = impact_object.get('bookmarks')
                    if not bookmarks:
                        impact_object_code = ''
                    else:
                        bookmark = bookmarks[0]
                        impact_object_field_id = bookmark.get('fileId', '')
                        impact_object_start_line = max(
                            int(bookmark.get('startLine', 1)) - 1, 0)
                        impact_object_end_line = max(
                            int(bookmark.get('endLine', 1)) - 1, 0)
                        impact_object_code_url = f"{imaging_url}rest/tenants/default/applications/{ApplicationName}/files/{impact_object_field_id}?start-line={impact_object_start_line}&end-line={impact_object_end_line}"
                        impact_object_code_response = requests.get(
                            impact_object_code_url)
                        if impact_object_code_response.status_code == 200:
                            impact_object_code = impact_object_code_response.text
                        else:
                            impact_object_code = ''
                            logging.error(
                                f"Failed to fetch impact object code using {impact_object_code_url}. Status code: {impact_object_code_response.status_code}")

                    new_impact_row = pd.DataFrame({
                        'object_type': [impact_object_type],
                        'object_signature': [impact_object_signature],
                        'object_link_type': [impact_object_link_type],
                        'object_code': [impact_object_code]
                    })
                    impacts = pd.concat(
                        [impacts, new_impact_row], ignore_index=True)
            else:
                logging.error(
                    f"Failed to fetch callers using {object_callers_url}. Status code: {object_callers_response.status_code}")
        else:
            logging.error(
                f"Failed to fetch object data using {object_url}. Status code: {object_response.status_code}")
            continue  # Skip to the next object

        if not exceptions.empty:
            # Group by 'link_type' and aggregate unique 'exception's
            grouped_exceptions = exceptions.groupby(
                'link_type')['exception'].unique()

            # Construct exception_text
            exception_text = f"Take into account that {object_type} <{object_signature}>: " + \
                "; ".join([f"{link_type} {', '.join(exc)}" for link_type,
                          exc in grouped_exceptions.items()])
            logging.info(f'exception_text = {exception_text}')
        else:
            exception_text = ""

        def generate_text(impacts):
            base_method = f"{object_type} <{object_signature}>"
            text = f"Take into account that {base_method} is used by:\n"
            for i, row in impacts.iterrows():
                text += f" {i+1}. {row['object_type']} <{row['object_signature']}> has a <{row['object_link_type']}> dependency as found in code:\n"
                text += f"````\n\t{row['object_code']}\n````\n"
            return text

        if not impacts.empty:
            impact_text = generate_text(impacts)
            logging.info(f'impact_text = {impact_text}')
        else:
            impact_text = ""

        prompt_content = (
            f"CONTEXT:\n{object_type} <{object_signature}> source code snippet below was reported for the "
            f"following reasons:\n" +
            f"{IssueName}\n"
            f"\n\nTASK:\n1/ Generate a version without the pattern occurrence(s) of the following code, "
            f"{TransformationTarget}:\n'''\n{obj_code}\n'''\n"
            f"2/ Provide an analysis of the transformation: detail what you did in the 'comment' field, forecast "
            f"impacts on code signature, exception management, enclosed objects or other areas in the "
            f"'signature_impact', 'exception_impact', 'enclosed_impact, and 'other_impact' fields respectively, "
            f"with some comments on your prognostics in the 'impact_comment' field.\n"
            f"\nGUIDELINES:\nUse the following JSON structure to respond:\n'''\n{json_resp}\n'''\n" +
            (f"\nIMPACT ANALYSIS CONTEXT:\n{impact_text}\n{exception_text}\n" if impact_text or exception_text else "") +
            "\nMake sure your response is a valid JSON string.\nRespond only the JSON string, and only the JSON string. "
            "Do not enclose the JSON string in triple quotes, backslashes, ... Do not add comments outside of the JSON structure."
        )

        prompt_content = prompt_content.replace(
            "\\n", "\n").replace("\\\"", "\"").replace("\\\\", "\\")

        logging.info(f"Prompt Content: {prompt_content}")

        messages = [{'role': 'user', 'content': prompt_content}]

        code_token = count_chatgpt_tokens(ai_model_name, str(obj_code))
        prompt_token = count_chatgpt_tokens(
            ai_model_name, "\n".join([json.dumps(m) for m in messages]))

        target_response_size = int(code_token * 1.2 + 500)

        if prompt_token < (ai_model_size - target_response_size):
            response_content = ask_ai_model(
                messages, ai_model_url, ai_model_api_key, ai_model_version, ai_model_name, max_tokens=target_response_size
            )
            logging.info(f"Response Content: {response_content}")
            time.sleep(model_invocation_delay)

            result.append({
                'prompt': prompt_content,
                'response': response_content,
                'source_path': object_source_path,
                'object_id': object_id,
                'line_start': object_start_line,
                'line_end': object_end_line,
                'technologies': object_technology,
                'req_id': RequestId
            })

            # response_content = json.loads(response_content)

            if response_content['updated'].lower() == 'yes':

                if SourceCodeLocation.lower() in object_source_path.lower():
                    object_source_path = object_source_path.replace(
                        SourceCodeLocation, '')
                    file_path = fixed_code_directory + object_source_path
                new_code = response_content['code']
                # Convert the new_code string back to its readable format
                readable_code = new_code.replace("\\n", "\n").replace(
                    "\\\"", "\"").replace("\\\\", "\\")
                start_line = object_start_line
                end_line = object_end_line

                replace_code(file_path, start_line, end_line, readable_code)

        else:
            logging.warning("Prompt too long; skipping.")
            result.append({
                'prompt': prompt_content,
                'response': "(NA prompt too long)",
                'source_path': object_source_path,
                'object_id': object_id,
                'line_start': object_start_line,
                'line_end': object_end_line,
                'technologies': object_technology,
                'req_id': RequestId
            })

    return {'data': result}


def process_request(
    ApplicationName, RequestId, IssueID, IssueName, TransformationSource, TransformationTarget,
    ai_model_name, ai_model_version, ai_model_url, ai_model_api_key,
    ai_model_max_tokens, imaging_url, imaging_api_key, output_directory, SourceCodeLocation, fixed_code_directory
):
    model_invocation_delay = 10

    json_resp = '''
    {
        "updated":"<YES/NO to state if you updated the code or not (if you believe it did not need fixing)>",
        "comment":"<explain here what you updated (or the reason why you did not update it)>",
        "missing_information":"<list here information needed to finalize the code (or NA if nothing is needed or if the code was not updated)>",
        "signature_impact":"<YES/NO/UNKNOWN, to state here if the signature of the code will be updated as a consequence of changed parameter list, types, return type, etc.>",
        "exception_impact":"<YES/NO/UNKNOWN, to state here if the exception handling related to the code will be update, as a consequence of changed exception thrown or caught, etc.>",
        "enclosed_impact":"<YES/NO/UNKNOWN, to state here if the code update could impact code enclosed in it in the same source file, such as methods defined in updated class, etc.>",
        "other_impact":"<YES/NO/UNKNOWN, to state here if the code update could impact any other code referencing this code>",
        "impact_comment":"<comment here on signature, exception, enclosed, other impacts on any other code calling this one (or NA if not applicable)>",
        "code":"<the fixed code goes here (or original code if the code was not updated)>"
    }
    '''

    data = gen_code_connected_json(
        ApplicationName, RequestId, IssueID, IssueName, TransformationSource, TransformationTarget,
        ai_model_name, ai_model_version, ai_model_url, ai_model_api_key,
        ai_model_max_tokens, imaging_url, imaging_api_key, model_invocation_delay, json_resp, SourceCodeLocation, fixed_code_directory
    )

    # Get current datetime stamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a filename with the datetime stamp
    filename = output_directory + \
        f'/AI_Response_for_ApplicationName-{ApplicationName}_RequestID-{RequestId}_IssueID-{IssueID}_{timestamp}.json'

    # Write the JSON data to a file
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == '__main__':
    try:
        current_directory = os.getcwd()

        with open(os.path.join(current_directory, 'cast_ai_engine_settings.json')) as f:
            data = json.load(f)

        ai_model_name = data["ai_model"]["name"]
        ai_model_version = data["ai_model"]["version"]
        ai_model_url = data["ai_model"]["url"]
        ai_model_api_key = data["ai_model"]["api_key"]
        ai_model_max_tokens = data["ai_model"]["max_tokens"]

        imaging_url = data["imaging"]["url"]
        imaging_api_key = data["imaging"]["api_key"]

        ApplicationName = data["input"]["application_name"]
        RequestId = data["input"]["request_id"]
        IssueID = data["input"]["issue_id"]
        IssueName = data["input"]["issue_name"]
        TransformationSource = data["input"]["transformation_source"]
        TransformationTarget = data["input"]["transformation_target"]
        SourceCodeLocation = data["input"]["source_code_location"]

        # Get current datetime stamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Define the directory name
        output_directory = f"Output_for_ApplicationName-{ApplicationName}_RequestID-{RequestId}_IssueID-{IssueID}_{timestamp}"

        # Create the directory
        os.makedirs(output_directory, exist_ok=True)
        print(f"Directory '{output_directory}' created successfully!")

        fixed_code_directory = output_directory + "\\Fixed_Source_Code\\"
        os.makedirs(fixed_code_directory, exist_ok=True)
        print(f"Directory '{fixed_code_directory}' created successfully!")

        shutil.copytree(SourceCodeLocation,
                        fixed_code_directory, dirs_exist_ok=True)

        # Create a filename with the datetime stamp
        filename = f'Logs_for_ApplicationName-{ApplicationName}_RequestID-{RequestId}_IssueID-{IssueID}_{timestamp}.txt'

        logging.basicConfig(
            filename=os.path.join(
                current_directory, output_directory, filename),
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
            filemode='w'
        )

        process_request(
            ApplicationName, RequestId, IssueID, IssueName, TransformationSource, TransformationTarget,
            ai_model_name, ai_model_version, ai_model_url, ai_model_api_key,
            ai_model_max_tokens, imaging_url, imaging_api_key, output_directory, SourceCodeLocation, fixed_code_directory
        )

    except Exception as e:
        print('An exception has occurred while executing the main function. Please resolve it or contact developers.')
        logging.error(
            'An exception has occurred while executing the main function.', exc_info=True)
        print(e)
