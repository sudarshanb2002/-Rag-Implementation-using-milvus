from fastapi import FastAPI, HTTPException
from controller import load_data, process_query, validate_prompt
from pydantic import BaseModel

class URLItem(BaseModel):
    url: str

class QueryItem(BaseModel):
    prompt: str

app = FastAPI()

@app.post("/load")
async def load_data_endpoint(item: URLItem):
    return await load_data(item.url)

@app.post("/query")
async def query_endpoint(item: QueryItem):
    # Process the query
    try:
        print(f"STATEMENT TO SEARCH: {item.prompt}")
        results = await process_query(item.prompt)
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the FastAPI server, use: uvicorn main:app --reload