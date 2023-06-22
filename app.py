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
os.environ['OPENAI_API_KEY'] = "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"

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

    system_template = """A ___ bot/assistant/etc that answers questions about Ash's work, design, process, ideas; as well as Ash's reading and research, and specifies when it uses Ash's reading / secondary research ; as well as Ash's Context Report / Dissertation."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    # Please update this prompt below to add instructions as to which symbols mean which chunk is that related to. 

    human_template = """

Please use the text enlcosed in triple backticks below to answer my query at the end. Just to give you a brief overview on how to interpret the text, please keep these points in mind:
• [] -> text enlcosed in square brackets is instructions for the bot, that give context about the following text - this demarcates the TextCategory of 'BotInstructionsT', the demarcation of which is concluded with `*`
• {D} {/D}-> text starting with `{D}` and ending with `{/D}` is Ash's own design / design work / process  - this demarcates the TextCategory of 'DesignT' (which can be nested within other categories), the demarcation of which is concluded with `{/D}`
• {^} {/^} -> text starting with a caret symbol in curly brackets `{^}` and ending with `{/^} is comments for the reader, things that would be important for someone reading the document or accessing it through the bot, to know  - this demarcates the TextCategory of 'ReaderCommentsT', the demarcation of which is concluded with `{/^}`
• {R} {/R} -> text starting with an R in curly brackets `{R}` and ending with `{/R}` is Ash's research - primary, or secondary - this demarcates the TextCategory of 'ResearchT', the demarcation of which is concluded with `{/R}`
• {CR} {/CR} -> text starting with `{CR}` and ending with `{/CR}` is from Ash's 'Context Report' / 'Dissertation', and could contain Ash's own thoughts and ideas (indicated with `{A}`), or the ideas of others (though this will always be indicated with citations)' - this demarcates the TextCategory of 'ContextReportT', the demarcation of which is concluded with `{/CR}`
• {A} {/A} -> text starting with `{A}` and ending with `{/A}` is Ash's own notes / comments / analysis, usually within another TextCategory - this demarcates the TextCategory of 'AshT' (which can be nested within other categories), the demarcation of which is concluded with `{/A}`

This text provides context as to whether the information is Ash's own work / notes / design (DesignT) ;
or that of someone else (within the TextCategory of ResearchT) ;
    or own notes / comments / analysis (AshT) usually within another TextCategory ;
or from his Context Report / Dissertation (ContextReportT), which contains his own thought, as well as the thoughts and ideas of others - though the latter will be indicated with citations ;
    and Context Report/Dissertation (ContextReportT) is often a synthesis of his ideas and those of others, building on them, or using them as an example or point in a building argument:
    
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


st.title("MetaBot")

query = st.text_input("Please enter your query about Ash's work/projects, and/or its theoretical basis:")

# if st.button("press me"):
#     display_pdf_with_zoom('test.pdf',13,200)

if query:
    docs = new_db.similarity_search(query, k=3)  # k = 3; Return top 3 results

    st.write(f"Metadata found out for the first object is: {docs[0].metadata}")
    
    relevant_content_and_pages = [{"page_number": doc.metadata.get("page_number", "N/A"), "content": doc.page_content} for doc in docs]

    # relevant_content_and_pages = [{"page_number":doc.metadata["page_number"],"content":doc.page_content} for doc in docs]
    # relevant_pages = [doc.metadata['page_number'] for doc in docs]

    st.write(f"Relevant Page numbers found are: {', '.join(map(str, [item['page_number'] for item in relevant_content_and_pages]))}")

    for page in [item['page_number'] for item in relevant_content_and_pages]:
        display_pdf_with_zoom(pdf_filename="test.pdf",page_number=page,zoom_level=300)
    st.success("Opened relevant pages of PDF")

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