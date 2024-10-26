import streamlit as st
from dotenv import load_dotenv
import os
from typing import List, Literal
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from io import BytesIO
import pandas as pd
import tempfile
from fpdf import FPDF

load_dotenv(".env")
os.environ["OPENAI_API_VERSION"] = "2024-07-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ai-proxy.lab.epam.com"
os.environ["AZURE_OPENAI_API_KEY"] = "DIAL-CapPUpfdomadLNfF8kH6kNSEl9e9dtHk"
dir_path = os.getenv("DIR_PATH", "./index")


def set_llms(
    embeddings_llm: str,
    deployment_name: str,
    temperature: float = 0.7,
    max_tokens: int = 500,
):
    embeddings = AzureOpenAIEmbeddings(model=embeddings_llm)
    llm = AzureChatOpenAI(
        temperature=temperature,
        deployment_name=deployment_name,
        max_tokens=max_tokens,
    )
    return embeddings, llm


def transform_data(
    documents: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def create_vector_store(documents: List[Document], embeddings, save: bool = True):
    vector_store = Chroma.from_documents(
        documents, embeddings, persist_directory=dir_path
    )
    if save:
        vector_store.persist()
    return vector_store


def load_vector_store(embeddings):
    return Chroma(persist_directory=dir_path, embedding_function=embeddings)


def retriever(input_text: str, vector_store, llm, top_k: int):
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    response = qa_chain.run(input_text)
    return response


def process_pdf(file_buffer):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_buffer.read())
        tmp_file_path = tmp_file.name
    loader = PyPDFLoader(tmp_file_path)
    return loader.load()


def process_csv(file_buffer):
    loader = CSVLoader(file_buffer)
    return loader.load()


def process_xlsx(file_buffer):
    df = pd.read_excel(file_buffer, engine="openpyxl")
    documents = [Document(page_content=str(row)) for index, row in df.iterrows()]
    return documents


def process_txt(file_buffer):
    loader = TextLoader(file_buffer)
    return loader.load()


def load_documents(uploaded_file):
    file_buffer = BytesIO(uploaded_file.getvalue())
    if uploaded_file.type == "application/pdf":
        documents = process_pdf(file_buffer)
    elif uploaded_file.type == "text/csv":
        documents = process_csv(file_buffer)
    elif (
        uploaded_file.type
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ):
        documents = process_xlsx(file_buffer)
    elif uploaded_file.type == "text/plain":
        documents = process_txt(file_buffer)
    else:
        st.error("Unsupported file type!")
    return documents


def load_data(
    documents,
    embeddings_model: str,
    deployment_name: str,
    chunk_size: int = 516,
    chunk_overlap: int = 128,
    temperature: float = 0.7,
    max_tokens: int = 500,
):
    embeddings, llm = set_llms(
        embeddings_model, deployment_name, temperature, max_tokens
    )
    documents = transform_data(documents, chunk_size, chunk_overlap)
    vector_store = create_vector_store(documents, embeddings)
    return vector_store


def reset_conversation():
    st.session_state.conversation = None
    st.session_state.chat_history = None


def export_to_pdf(response):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, response)
    pdf_output = BytesIO()
    pdf_output.write(pdf.output(dest="S").encode("latin1"))
    pdf_output.seek(0)
    return pdf_output


def modify_prompt_based_on_role(prompt, role):
    if role == "Developer":
        return f"As a developer, {prompt}"
    elif role == "Product owner":
        return f"As a product owner, {prompt}"
    elif role == "AI Coach":
        return f"As an AI coach, {prompt}"
    else:
        return prompt


# Title
st.title("ChatRFP")

# Category section
st.header("Category")
category = st.selectbox("Choose Category", ["Action 1", "Action 2", "Action 3"])

# Input Section
st.header("INPUT SECTION")

# Action dropdown
action = st.selectbox("Select Action", ["Developer", "Product owner", "AI Coach"])

# File uploader
uploaded_file = st.file_uploader(
    "Upload Files", type=["pdf", "csv", "xlsx", "txt"], help="Limit 200MB per file"
)
if uploaded_file:
    st.session_state["uploaded_file"] = uploaded_file

# Button to create embeddings
if st.button("Create Embeddings"):
    if "uploaded_file" in st.session_state:
        documents = load_documents(st.session_state["uploaded_file"])
        st.write("filename:", uploaded_file.name)
        with st.spinner("Wait for it..."):
            embeddings_model = "text-embedding-ada-002"  # Default value, can be changed
            deployment_name = "gpt-4o-2024-08-06"  # Default value, can be changed
            chunk_size = 512  # Default value, can be changed
            chunk_overlap = 128  # Default value, can be changed
            temperature = 0.7  # Default value, can be changed
            max_tokens = 500  # Default value, can be changed
            vector_store = load_data(
                documents,
                embeddings_model,
                deployment_name,
                chunk_size,
                chunk_overlap,
                temperature,
                max_tokens,
            )
            st.session_state["vector_store"] = vector_store
        st.success("Embeddings created!")

# Text box for prompt input
prompt = st.text_area("Enter prompt for the chat")

# Generate button
if st.button("Generate"):
    if (
        "vector_store" in st.session_state
        and st.session_state["vector_store"] is not None
    ):
        embeddings_model = "text-embedding-ada-002"  # Default value, can be changed
        deployment_name = "gpt-4o-2024-08-06"  # Default value, can be changed
        temperature = 0.7  # Default value, can be changed
        max_tokens = 500  # Default value, can be changed
        embeddings, llm = set_llms(
            embeddings_model, deployment_name, temperature, max_tokens
        )

        # Modify the prompt based on the selected role
        modified_prompt = modify_prompt_based_on_role(prompt, action)

        response = retriever(
            modified_prompt, st.session_state.vector_store, llm, top_k=5
        )
        st.session_state["response"] = response
        st.write("Response generated based on the prompt:", prompt)
    else:
        st.error("Please create embeddings before generating a response.")

# Output Section
st.header("OUTPUT SECTION")

# Response Text Box
response = st.text_area(
    "Response",
    st.session_state.get("response", "Generated response will appear here..."),
)

# Export option dropdown
export_format = st.selectbox("Export as", ["PDF", "TXT", "DOCX"])

# Export button
if st.button("Export"):
    if "response" in st.session_state:
        if export_format == "PDF":
            pdf_output = export_to_pdf(st.session_state["response"])
            st.download_button(
                label="Download PDF",
                data=pdf_output,
                file_name="response.pdf",
                mime="application/pdf",
            )
        elif export_format == "TXT":
            txt_output = BytesIO(st.session_state["response"].encode())
            st.download_button(
                label="Download TXT",
                data=txt_output,
                file_name="response.txt",
                mime="text/plain",
            )
        elif export_format == "DOCX":
            # Implement DOCX export if needed
            pass
    else:
        st.error("No response to export.")

# Satisfaction option
satisfied = st.radio("Are you satisfied with the answer?", ("Yes", "No"))

if satisfied == "Yes":
    st.success("Glad you're satisfied!")
else:
    st.warning("Sorry to hear that, we'll try to improve.")
