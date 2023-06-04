import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from open_pdf_from_page_and_zoom import display_pdf_with_zoom

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = api_key

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
model_name = "text-davinci-003"  # the model used for text generation
chain_type = "refine"  # Results returned using 'refine' are detailed as compared to 'map_reduce', but you could try to play around with these things.

new_db = FAISS.load_local("ash_local_FAISS_index", embeddings)


def create_response(matched_chunks, question):
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=.1
    )

    # You can assign the system a role here. You play around this prompt to see how the response changes.

    system_template = """A ___ bot/assistant/etc that answers questions about Ash's work, design, process, ideas; as well as Ash's reading and research, and specifies when it uses Ash's reading / secondary research."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    # Please update this prompt below to add instructions as to which symbols mean which chunk is that related to. 

    human_template = """This text provides context as to whether the information is Ash's own notes/design,
    or his notes/research on the thoughts and work of others, or his Context Report/Dissertation. Notes/research
    and Context Report/Dissertation likely include some of his Design, which will be indicated:
    
    ```{matched_chunks}```
    
    Please now answer this query based on the text I shared with you:
    
    QUESTION:
    {question}"""

    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt])

    response = llm(chat_prompt.format_prompt(
        matched_chunks=matched_chunks,question=question).to_messages()).content

    return response


st.title("PDF Search and QA App")

query = st.text_input("Please enter your query:")

# if st.button("press me"):
#     display_pdf_with_zoom('test.pdf',13,200)

if query:
    docs = new_db.similarity_search(query, k=3)  # k = 3; Return top 3 results

    relevant_content_and_pages = [{"page_number":doc.metadata["page_number"],"content":doc.page_content} for doc in docs]
    # relevant_pages = [doc.metadata['page_number'] for doc in docs]

    st.write(f"Relevant Page numbers found are: {', '.join(map(str, [item['page_number'] for item in relevant_content_and_pages]))}")

    for page in [item['page_number'] for item in relevant_content_and_pages]:
        display_pdf_with_zoom(pdf_filename="test.pdf",page_number=page,zoom_level=300)
        st.write("opened page in PDF")

    with st.expander("View relevant chunks"):
        # Iterate over the relevant_content_and_pages list
        for item in relevant_content_and_pages:
            page_number = item["page_number"]
            content = item["content"]

            st.write(f"Page number # {page_number}")
            st.write(content)

    
    # chain = load_qa_with_sources_chain(OpenAI(temperature=0, model_name=model_name, max_tokens=-1), chain_type=chain_type)  # -1 sets the limit to the best what is available with the current model
    # result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)

    result = create_response(matched_chunks="\n".join([obj['content'] for obj in relevant_content_and_pages]),question=query)
    
    st.write(f"Response: {result}")