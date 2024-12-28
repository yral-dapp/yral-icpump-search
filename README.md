# YRAL ICPump Search Service

Welcome to the YRAL IcPump Search Service! This guide will help you understand our codebase and get started with development.

## Project Overview

This service provides semantic search capabilities over token metadata using Google's BigQuery, Vertex AI, and PaLM APIs. It combines vector embeddings with natural language processing to deliver intelligent search results.

## Tech Stack

- Python 3.11+
- FastAPI
- Google Cloud Platform
  - BigQuery
  - Vertex AI
  - GeminiAPI
- Docker
- Fly.io for deployment

## Development Setup

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export SERVICE_CRED='your_service_account_json'
export GOOGLE_GENAI_API_KEY='your_gemini_api_key'
```

## Coding Style Guidelines

1. **Type Hints**: Always use type hints for function parameters and return values
```python
def process_query(self, user_query: str, table_name: str = "") -> tuple[pd.DataFrame, str, str]:
```

2. **Documentation**: Use docstrings for classes and functions
```python
def semantic_search_bq(query_text: str, bq_client: Optional[bigquery.Client] = None) -> pd.DataFrame:
    """
    Performs semantic search on BigQuery table.
    
    Args:
        query_text: The search query
        bq_client: BigQuery client instance
        
    Returns:
        DataFrame with search results
    """
```

3. **Error Handling**: Use try-except blocks for external API calls and file operations
```python
try:
    response = self.model.generate_content(contents)
except Exception as e:
    logger.error(f"Error generating content: {e}")
    raise
```

4. **Configuration**: Keep configuration in separate files/environment variables
```python
base_table = os.getenv('BASE_TABLE', 'default_table_name')
```

## Deployment Process

1. All commits to `main` branch trigger automatic deployment via GitHub Actions
2. Tests must pass before deployment
3. Deployment is handled by Fly.io
4. Monitor logs post-deployment

## Exercise: Create a Hello World LLM Endpoint

Create a simple endpoint that:
1. Accepts a user query
2. Retrieves some token data from BigQuery
3. Uses the LLM to generate a response

### Steps:

1. Create a new file `hello_world.py`:
```python
from fastapi import FastAPI
from google.cloud import bigquery
import google.generativeai as genai
import os
import json
from google.oauth2 import service_account

app = FastAPI()

# Initialize credentials
service_cred = os.environ['SERVICE_CRED']
service_acc_creds = json.loads(service_cred, strict=False)
genai.configure(api_key=os.environ['GOOGLE_GENAI_API_KEY'])
credentials = service_account.Credentials.from_service_account_info(service_acc_creds)
bq_client = bigquery.Client(credentials=credentials)

@app.get("/hello")
async def hello_world(query: str):
    # 1. Get some sample token data
    token_query = """
    SELECT token_name, description, created_at 
    FROM `hot-or-not-feed-intelligence.icpumpfun.token_metadata_v1`
    LIMIT 5
    """
    df = bq_client.query(token_query).to_dataframe()
    
    # 2. Create LLM model
    model = genai.GenerativeModel('gemini-1.0-pro')
    
    # 3. Generate response
    prompt = f"""
    User Query: {query}
    Available Token Data:
    {df.to_string()}
    
    Please provide a friendly response about these tokens based on the user's query.
    """
    
    response = model.generate_content(prompt)
    
    return {
        "query": query,
        "tokens": df.to_dict('records'),
        "llm_response": response.text
    }
```

2. Run the server:
```bash
uvicorn hello_world:app --reload
```

3. Test the endpoint:
```bash
curl "http://localhost:8000/hello?query=Tell%20me%20about%20these%20tokens"
```

### Expected Output:
```json
{
  "query": "Tell me about these tokens",
  "tokens": [...],
  "llm_response": "Here are some interesting tokens..."
}
```

## Next Steps

1. Add error handling
2. Implement semantic search
3. Add rate limiting
4. Implement caching
5. Add authentication

## Need Help?

- Check the existing code in `search_agent_bq.py` for examples
- Review our test cases in `test_case_results.txt`
- Reach out to the team on Slack

Happy coding! 🚀 
