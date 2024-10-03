import time
from flask import Flask, jsonify
from config import Config
import requests
import json
import pandas as pd
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from openai import AzureOpenAI
from datetime import datetime

app = Flask(__name__)
# Load configuration from config.py
app.config.from_object(Config)

# Access configuration variables
ai_model_name = app.config['MODEL_NAME']
ai_model_version = app.config['MODEL_VERSION']
ai_model_url= app.config['MODEL_URL']
ai_model_api_key = app.config['MODEL_API_KEY']
ai_model_max_tokens = app.config['MODEL_MAX_TOKENS']

imaging_url = app.config['IMAGING_URL']
imaging_api_key = app.config['IMAGING_API_KEY']

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

# Home route
@app.route('/')
def home():
    return "Welcome to the MMC GENAI ENGINE!"

def ask_ai_model(messages, max_tokens):

    # Initialize Azure OpenAI client with key-based authentication
    client = AzureOpenAI(
        azure_endpoint = ai_model_url,
        api_key = ai_model_api_key,
        api_version = ai_model_version,
    )

    completion = client.chat.completions.create(
        model=ai_model_name,
        messages= messages,
        # past_messages=10,
        max_tokens = max_tokens,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )

    # Convert to dictionary
    completion_dict = {
        "id": completion.id,
        "choices": [
            {
                "finish_reason": choice.finish_reason,
                "index": choice.index,
                "logprobs": choice.logprobs,
                "message": {
                    "content": choice.message.content,
                    "refusal": choice.message.refusal,
                    "role": choice.message.role,
                    "function_call": choice.message.function_call,
                    "tool_calls": choice.message.tool_calls
                },
                "content_filter_results": choice.content_filter_results
            } for choice in completion.choices
        ],
        "created": completion.created,
        "model": completion.model,
        "object": completion.object,
        "service_tier": completion.service_tier,
        "system_fingerprint": completion.system_fingerprint,
        "usage": {
            "completion_tokens": completion.usage.completion_tokens,
            "prompt_tokens": completion.usage.prompt_tokens,
            "total_tokens": completion.usage.total_tokens,
            "completion_tokens_details": completion.usage.completion_tokens_details
        },
        "prompt_filter_results": completion.prompt_filter_results
    }

    # Convert to JSON
    completion_json = json.dumps(completion_dict, indent=4)

    completion_dict = json.loads(completion_json)
    
    return completion_dict["choices"][0]["message"]["content"]

# count_token_mode = "tiktoken"  # default value
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_chatgpt_tokens(ai_model_name, prompt):
    tokens = tokenizer.encode(prompt)
    return len(tokens)
   
def gen_code_connected_json(ApplicationName, RequestId, IssueID, json_resp, transformation_target, ai_model_name, model_invocation_delay):

    ai_model_size = ai_model_sizes[ai_model_name]

    result = []

    url = f"{imaging_url}rest/tenants/default/applications/{ApplicationName}/insights/green-detection-patterns/{IssueID}/findings?limit=100000"
    params = {
        'api-key': imaging_api_key
    }
    response = requests.get(url, params=params)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response if needed
        data = response.json()
        objects = data
        # print("Response Data:", data)
    else:
        print(f"Failed to fetch data using {url}. Status code: {response.status_code}")

    for object in objects:
        object_id = object['id']
        exceptions = pd.DataFrame(columns=['link_type', 'exception'])
        impacts = pd.DataFrame(columns=['object_type', 'object_signature', 'object_link_type', 'object_code'])

        object_url = f"{imaging_url}rest/tenants/default/applications/{ApplicationName}/objects/{object_id}?select=source-locations"
        # Make the GET request
        object_response = requests.get(object_url)
        # Check if the request was successful
        if object_response.status_code == 200:
            # Parse the JSON response if needed
            object_data = object_response.json()
            object_type = object['typeId']
            object_signature = object['mangling']
            object_technology = object_data['programmingLanguage']['name']
            object_source_path = object_data['sourceLocations'][0]['filePath']
            object_field_id = object_data['sourceLocations'][0]['fileId']
            object_start_line = object_data['sourceLocations'][0]['startLine']
            object_end_line = object_data['sourceLocations'][0]['endLine']

            object_code_url = f"{imaging_url}rest/tenants/default/applications/{ApplicationName}/files/{object_field_id}?start-line={object_start_line}&end-line={object_end_line}"
            # Make the GET request
            object_code_response = requests.get(object_code_url)
            if object_code_response.status_code == 200:
                # Parse the JSON response if needed
                obj_code = object_code_response.text
                # print("object_data:", object_code)

            object_callees_url = f"{imaging_url}rest/tenants/default/applications/{ApplicationName}/objects/{object_id}/callees"
            # Make the GET request
            object_callees_response = requests.get(object_callees_url)
            if object_callees_response.status_code == 200:
                # Parse the JSON response if needed
                object_exceptions = object_callees_response.json()
                for object_exception in object_exceptions:
                    if object_exception['linkType'].lower() == 'raise' or object_exception['linkType'].lower() == 'throw' or object_exception['linkType'].lower() == 'catch': 
                        exceptions = exceptions._append({'link_type': object_exception['linkType'], 'exception': object_exception['name']}, ignore_index=True)

            object_callers_url = f"{imaging_url}rest/tenants/default/applications/{ApplicationName}/objects/{object_id}/callers?select=bookmarks"
            object_callers_response = requests.get(object_callers_url)
            if object_callers_response.status_code == 200:
                # Parse the JSON response if needed
                impact_objects = object_callers_response.json()
                for impact_object in impact_objects:
  
                    impact_object_id = impact_object['id']
                    impact_object_url = f"{imaging_url}rest/tenants/default/applications/{ApplicationName}/objects/{impact_object_id}?select=source-locations"
                    impact_object_response = requests.get(impact_object_url)
                    # Check if the request was successful
                    if impact_object_response.status_code == 200:
                        # Parse the JSON response if needed
                        impact_object_data = impact_object_response.json()
                        impact_object_type = impact_object_data['typeId']
                        impact_object_signature = impact_object_data['mangling']

                    impact_object_link_type = impact_object['linkType']

                    if impact_object['bookmarks'] == None:
                        imapct_object_code = ''
                    else:
                        impact_object_field_id = impact_object['bookmarks'][0]['fileId']
                        impact_object_start_line = int(impact_object['bookmarks'][0]['startLine']) - 1
                        impact_object_end_line = int(impact_object['bookmarks'][0]['endLine']) - 1
                        impact_object_code_url = f"{imaging_url}rest/tenants/default/applications/{ApplicationName}/files/{impact_object_field_id}?start-line={impact_object_start_line}&end-line={impact_object_end_line}"
                        # Make the GET request
                        impact_object_code_response = requests.get(impact_object_code_url)
                        if impact_object_code_response.status_code == 200:
                            # Parse the JSON response if needed
                            imapct_object_code = impact_object_code_response.text

                    impacts = impacts._append({'object_type': impact_object_type, 'object_signature': impact_object_signature,'object_link_type': impact_object_link_type,'object_code': imapct_object_code}, ignore_index=True)

        else:
            print(f"Failed to fetch object_data using {object_url}. Status code: {response.status_code}")

        if not exceptions.empty:
            exception_text = "Take into account that " + str(object_type) + " <" + str(object_signature) + ">: " + \
                                "; ".join(exceptions.groupby('link_type').apply(
                                    lambda g: f"{g['link_type'].iloc[0]} {', '.join(g['exception'].unique())}"))
            print('exception_text = '+exception_text)
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
            print('impact_text = '+impact_text)
        else:
            impact_text = ""
            
        prompt_content = (
            f"CONTEXT:\n{object_type} <{object_signature}> " +
            f"\n\nTASK:\n1/ Generate a version without the pattern occurrence(s) of the following code, "
            f"{transformation_target}:\n'''\n{obj_code}\n'''\n"
            f"2/ Provide an analysis of the transformation: detail what you did in the 'comment' field, forecast "
            f"impacts on code signature, exception management, enclosed objects or other areas in the "
            f"'signature_impact', 'exception_impact', 'enclosed_impact, and 'other_impact' fields respectively, "
            f"with some comments on your prognostics in the 'impact_comment' field.\n"
            f"\nGUIDELINES:\nUse the following JSON structure to respond:\n'''\n{json_resp}\n'''\n" +
            (f"\nIMPACT ANALYSIS CONTEXT:\n{impact_text}\n{exception_text}\n" if impact_text or exception_text else "") +
            "\nMake sure your response is a valid JSON string.\nRespond only the JSON string, and only the JSON string. "
            "Do not enclose the JSON string in triple quotes, backslashes, ... Do not add comments outside of the JSON structure."
        )

        print(prompt_content)

        # prompt_content = (
        #     f"CONTEXT:\n{object_type} <{object_signature}> source code snippet below was reported for the "
        #     f"following reasons:" +
        #     "".join([
        #         f"\n{idx+1}. According to {instruction['req_ref']}, {instruction['req_desc']}\n"
        #         f"In the source code below, occurrence(s) of <<{instruction['req_pattern']}>> pattern was found."
        #         for idx, instruction in instruction_pieces.iterrows()
        #     ]) +
        #     f"\n\nTASK:\n1/ Generate a version without the pattern occurrence(s) of the following code, "
        #     f"{transformation_target}:\n'''\n{row['code']}\n'''\n"
        #     f"2/ Provide an analysis of the transformation: detail what you did in the 'comment' field, forecast "
        #     f"impacts on code signature, exception management, enclosed objects or other areas in the "
        #     f"'signature_impact', 'exception_impact', 'enclosed_impact, and 'other_impact' fields respectively, "
        #     f"with some comments on your prognostics in the 'impact_comment' field.\n"
        #     f"\nGUIDELINES:\nUse the following JSON structure to respond:\n'''\n{json_resp}\n'''\n" +
        #     (f"\nIMPACT ANALYSIS CONTEXT:\n{impact_text}\n{exception_text}\n" if impact_text or exception_text else "") +
        #     "\nMake sure your response is a valid JSON string.\nRespond only the JSON string, and only the JSON string. "
        #     "Do not enclose the JSON string in triple quotes, backslashes, ... Do not add comments outside of the JSON structure."
        # )

        messages = [{'role': 'user', 'content': prompt_content}]

        code_token = count_chatgpt_tokens(ai_model_name, str(obj_code))
        
        prompt_token = count_chatgpt_tokens(ai_model_name, "\n".join([json.dumps(m) for m in messages]))

        target_response_size = int(code_token * 1.2 + 500)

        if prompt_token < (ai_model_size - target_response_size):
            
            response_content = ask_ai_model(messages, max_tokens=target_response_size)
            
            print(response_content)

            print("\n\n(invocation delay)\n\n")
            time.sleep(model_invocation_delay)

            result.append({
                'prompt': prompt_content,
                'response': response_content,
                'source_path': object_source_path,
                'scan_id': object_id,
                'line_start': object_start_line,
                'line_end': object_end_line,
                'technologies': object_technology,
                'req_id': RequestId
            })
        else:
            print("(NA prompt too long)")
            result.append({
                'prompt': prompt_content,
                'response': "(NA prompt too long)",
                'source_path': object_source_path,
                'scan_id': object_id,
                'line_start': object_start_line,
                'line_end': object_end_line,
                'technologies': object_technology,
                'req_id': RequestId
            })

    return {'data': result}

@app.route('/ProcessRequest/<string:ApplicationName>/<int:RequestId>/<int:IssueID>')
def process_request(ApplicationName, RequestId, IssueID):

    # Get Request Information from Mongo DB
    # get_request_information(RequestID)

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

    target_application = ApplicationName
    transformation_source = 'green'
    model_invocation_delay = 10

    if transformation_source == 'green':
        transformation_target = f"targeting Green (use specific credentials, '{target_application}' resource, 'us-west-2' region)"
    elif transformation_source == 'cloud':
        transformation_target = f"targeting AWS (list missing required information about the target situation in the dedicated field of your response)"

    data = gen_code_connected_json(ApplicationName, RequestId, IssueID, json_resp, transformation_target, ai_model_name, model_invocation_delay)

    # Get current datetime stamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create a filename with the datetime stamp
    filename = f'AI_Response_for_ApplicationName-{ApplicationName}_RequestID-{RequestId}_IssueID-{IssueID}_{timestamp}.json'
  
    # Write the JSON data to a file
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return jsonify(data), 200  # Return JSON response with HTTP status code

if __name__ == '__main__':
    app.run(debug=True, port=5001)
