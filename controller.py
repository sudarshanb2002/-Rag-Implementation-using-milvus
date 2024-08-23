from fastapi import HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from llm import insert_to_milvus, perform_similarity_search

class URLItem(BaseModel):
    url: str

async def load_data(url: str):
    # Validate the URL
    if not url.startswith("https://en.wikipedia.org/wiki/"):
        raise HTTPException(status_code=400, detail="Invalid Wikipedia URL")

    try:
        # Fetch the Wikipedia page content
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the title
        title = soup.find("h1", id="firstHeading").text

        # Extract all paragraphs and remove newline characters
        paragraphs = [
            p.get_text(separator=" ").replace("\n", " ") for p in soup.find_all("p")
        ]

        # Extract all headings
        headings = []
        for header_tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            headings.extend([h.text for h in soup.find_all(header_tag)])

        # Extract all lists (ol and ul) and remove newline characters
        lists = []
        for list_tag in ["ul", "ol"]:
            list_items = soup.find_all(list_tag)
            for item in list_items:
                list_text = "\n".join(
                    [
                        li.get_text(separator=" ").replace("\n", "")
                        for li in item.find_all("li")
                    ]
                )
                lists.append(list_text.replace("\n", ""))

        # Combine all content
        content = {
            "title": title,
            "headings": headings,
            "paragraphs": paragraphs,
            "lists": lists,
        }
        # Format and insert data into Milvus


        insert_to_milvus(content)

        # Retrieve data from the collection


        return "Data Inserted Successfully"

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


def validate_prompt(prompt: str) -> str:
    if not prompt or len(prompt.strip()) == 0:
        raise ValueError("Prompt cannot be empty.")
    # Add more validation rules as needed
    return prompt

async def process_query(prompt: str):
    # Validate the prompt
    validated_prompt = validate_prompt(prompt)

    # Call perform_similarity_search after validation
    try:
        results = perform_similarity_search(validated_prompt)
        return results
    except Exception as e:
        raise RuntimeError(f"Error during similarity search: {e}")