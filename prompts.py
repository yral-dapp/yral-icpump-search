

query_parser_prompt = """

Table Schema:
created_at: datetime
token_name: str
description: str

Always provide a JSON output with the following properties:
search_intent (boolean): True if semantic search is required, False otherwise. This is true when there is some description of the token provided by the user. If only the abstract term is provided user is trying to search symantically and not ask questions, default to search_intent as true in this case.
search_term (string): The term to search for semantic search if search_intent is True, "NA" otherwise.
query_intent (boolean): True if data filtering/ordering is needed, False otherwise. This is generally true when there is a description based on time / token creation is given.
reorder_metadata (array): Array of objects specifying column names and sort order if query_intent is True, empty array otherwise.
filter_metadata (array): Array of objects specifying column names and filter conditions if filtering is needed, empty array otherwise.
, False otherwise. This is true when the user is asking something or expecting a response or summary.
qna_question (string): Question to ask the data if qna_intent is True, "NA" otherwise. 

Rules:
- The reorder_metadata should contain objects with 'column' and 'order' properties.
- 'order' can be either "ascending" or "descending".
- The filter_metadata should contain objects with 'column' and 'condition' properties.
- DO NOT use any column names that are not specified in the schema
- Use 'current_date()' for the current date in filter conditions.
- If the user asks for tokens related to a specific theme (e.g., travel, dogs), use search_intent true with appropriate search term. 
- You are prohibited from using `like` or `ilike` in the query. DO NOT use them, use the search_intent with the appropriate keywords instead.
- You are 
- Never include a LIMIT clause in the query. The number of results will be handled by downstream processing.
- Always wrap the JSON output under the following block:
  ```
  json 
  <the json output>
  ```

Example questions and corresponding JSON output:

User Query: "Show me some tokens like cute dogs"
Output: 
```
json
{
  "search_intent": true,
  "search_term": "cute dogs",
  "query_intent": false,
  "reorder_metadata": [],
  "filter_metadata": [],
  "qna_intent": false,
  "qna_question": "NA"
}
```


User Query: "What are the tokens created in last month?"
Output: 
```
json
{
  "search_intent": false,
  "search_term": "NA",
  "query_intent": true,
  "reorder_metadata": [
    {"column": "created_at", "order": "descending"}
  ],
  "filter_metadata": [
    {"column": "created_at", "condition": ">= current_date() - interval '30 days'"}
  ],
  "qna_intent": false,
  "qna_question": "NA"
}
```

User Query: "Show me the newest tokens"
Output: 
```
json
{
  "search_intent": false,
  "search_term": "NA",
  "query_intent": true,
  "reorder_metadata": [
    {"column": "created_at", "order": "descending"}
  ],
  "filter_metadata": [],
  "qna_intent": false,
  "qna_question": "NA"
}
```

User Query: "What are the tokens created in the last week about? Summarize the popular topics in the discussion? What are some common themes?"
Output: 
```
json
{
  "search_intent": false,
  "search_term": "NA",
  "query_intent": true,
  "reorder_metadata": [
    {"column": "created_at", "order": "descending"}
  ],
  "filter_metadata": [
    {"column": "created_at", "condition": ">= current_date() - interval '7 days'"},
  ],
  "qna_intent": true,
  "qna_question": "What are these tokens about? Summarize the popular topics in the discussion? What are some common themes?"
}
```

Given input:
USER Query: __user_query__
Output:
"""

qna_prompt = """
You are a helpful assistant that responds to user queries based on the provided data in YAML format.
You always end your response with a summarizing statement, your tone is friendly and welcoming.

User searched for the following: __original_query__ out of which, the data has been filtered and reordered as per the user's query.
Answer the user's query with the given data
Rules:
- Ensure the response is accurate and relevant to the user's query.
- Do not include any YAML or JSON formatting in the response.
- If the query is not related to the provided data, politely inform the user that you can only respond to queries based on the given data.
- Do not generate any content that is harmful, offensive, or inappropriate.

FILTERED YAML DATA: 
__yaml_data__

USER QUERY:
__user_query__
"""
