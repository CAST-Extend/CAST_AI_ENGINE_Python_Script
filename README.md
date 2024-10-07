# CAST AI Engine
The CAST AI Engine is an AI-powered tool that utilizes OpenAI's GPT-4-32k model for various source code transformations. It is configurable for specific use cases such as processing code based on predefined issues.

#### Features
- AI Integration: Uses GPT-4-32k model.
- Imaging API: Integrated for additional processing.
- Source Code Transformation: Targets specific transformations based on inputs.

#### Installation
1. Clone the repository:

```bash
git clone <repository-url>
```
1. Install dependencies:

```bash
pip install -r requirements.txt
```

#### Configuration
Update the cast_ai_engine_settings.json:

#### json

```json
{
  "ai_model": {
    "name": "gpt-4-32k",
    "version": "2024-05-01-preview",
    "url": "https://itappsopenaichn.openai.azure.com/",
    "api_key": "",
    "max_tokens": 32000
  },
  "imaging": {
    "url": "http://localhost:5000/",
    "api_key": ""
  },
  "input": {
    "application_name": "Webgoat",
    "request_id": "1",
    "issue_id": "1200126",
    "issue_name": "Green - Avoid Programs not using explicitly OPEN and CLOSE for files or streams",
    "transformation_source": "green",
    "transformation_target": "targeting Green (use specific credentials, Webgoat resource, 'us-west-2' region)",
    "source_code_location": "C:\\ProgramData\\CAST\\AIP-Console-Standalone\\shared\\upload\\Webgoat\\main_sources\\"
  }
}

```

#### Usage
Run the engine:

```bash
python cast_ai_engine.py
```

Ensure your source code is in the defined location.
Converting python file into EXE is under process.
This is still under testing phase.
If you face any issues please reach me s.p@castsoftware.com 
