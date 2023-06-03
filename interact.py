from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = api_key

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
model_name = "text-davinci-003"  # the model used for text generation
chain_type = "refine"  # Results returned using 'refine' are detailed as compared to 'map_reduce', but you could try to play around with these things.

new_db = FAISS.load_local("ash_local_FAISS_index", embeddings)

while True:
    query = input("Please enter your query: ")
    docs = new_db.similarity_search(query, k=3)  # k = 3; Return top 3 results

    relevant_pages = []
    for doc in docs:
        relevant_pages.append(doc.metadata['page_number'])

    print(f"Relevant Page numbers found are: {', '.join(map(str, relevant_pages))}")
    chain = load_qa_with_sources_chain(OpenAI(temperature=0, model_name=model_name, max_tokens=-1), chain_type=chain_type)  # -1 sets the limit to the best what is available with the current model

    result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    print(f"Response: {result['output_text']}")
