import os

import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from llm import LLM
from generator.generator import GeneratorRunnable
from evaluator.evaluator import EvaluatorRunnable

from ingestion.pipeline import IngestionPipeline

from processing.processor import DocumentProcessor

from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever

from cache.redis_client import RedisClient
from cache.cache_manager import CacheManager

from workflow.workflow import GeneratorEvaluatorWorkflow


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Streamlit Configuration
# ============================================================

st.set_page_config(
    page_title="Evaluator-Generator QA",
    page_icon="",
    layout="wide",
)


# ============================================================
# Application Initialization
# ============================================================

@st.cache_resource
def initialize_application():

    # --------------------------------------------------------
    # 1. Generator LLM
    # --------------------------------------------------------

    generator_llm = LLM.get_gen_llm()

    # --------------------------------------------------------
    # 2. Evaluator LLM
    # --------------------------------------------------------

    evaluator_llm = LLM.get_eval_llm()

    # --------------------------------------------------------
    # 3. Embedding Model
    # --------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    )

    # --------------------------------------------------------
    # 4. Vector Store
    # --------------------------------------------------------

    vector_store = VectorStore(
        embeddings=embeddings
    )

    # --------------------------------------------------------
    # 5. Retriever
    # --------------------------------------------------------

    retriever = Retriever(
        vector_store=vector_store,
        k=5,
    )

    # --------------------------------------------------------
    # 6. Generator
    # --------------------------------------------------------

    generator = GeneratorRunnable(
        llm=generator_llm
    )

    # --------------------------------------------------------
    # 7. Evaluator
    # --------------------------------------------------------

    evaluator = EvaluatorRunnable(
        llm=evaluator_llm
    )

    # --------------------------------------------------------
    # 8. Redis
    # --------------------------------------------------------

    redis_client = RedisClient()

    cache_manager = CacheManager(
        redis_client=redis_client.client,
        ttl=3600,
    )

    # --------------------------------------------------------
    # 9. Workflow
    # --------------------------------------------------------

    workflow = GeneratorEvaluatorWorkflow(
        generator=generator,
        evaluator=evaluator,
        retriever=retriever,
        cache=cache_manager,
        max_iterations=4,
    )

    # --------------------------------------------------------
    # 10. Ingestion
    # --------------------------------------------------------

    ingestion = IngestionPipeline(
        vector_store=vector_store,
        chunk_size=1000,
        chunk_overlap=150,
    )

    return (
        workflow,
        ingestion,
        redis_client,
    )


# ============================================================
# Initialize
# ============================================================

workflow, ingestion, redis_client = (
    initialize_application()
)


# ============================================================
# Main UI
# ============================================================

st.title("🤖 Evaluator-Generator QA System")

st.write(
    "Upload knowledge sources and ask questions "
    "using the Generator-Evaluator workflow."
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("Knowledge Base")


# ============================================================
# File Upload
# ============================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload a document",
    type=[
        "pdf",
        "docx",
        "txt",
        "md",
        "py",
        "java",
        "cpp",
        "c",
        "h",
        "js",
        "ts",
        "pptx",
        "wav",
    ],
)


if uploaded_file is not None:

    os.makedirs(
        "./data/uploads",
        exist_ok=True,
    )

    file_path = os.path.join(
        "./data/uploads",
        uploaded_file.name,
    )

    with open(
        file_path,
        "wb",
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    if st.sidebar.button(
        "Ingest File"
    ):

        try:

            with st.spinner(
                "Loading, processing and indexing..."
            ):

                number_of_chunks = (
                    ingestion.ingest_file(
                        file_path
                    )
                )

            st.sidebar.success(
                f"{number_of_chunks} chunks indexed."
            )

        except Exception as e:

            st.sidebar.error(
                f"Ingestion failed: {e}"
            )


# ============================================================
# URL Ingestion
# ============================================================

st.sidebar.subheader("Web Source")

url = st.sidebar.text_input(
    "Enter URL",
    placeholder="https://example.com",
)


if st.sidebar.button(
    "Ingest URL"
):

    if not url.strip():

        st.sidebar.warning(
            "Please enter a URL."
        )

    else:

        try:

            with st.spinner(
                "Loading and indexing URL..."
            ):

                number_of_chunks = (
                    ingestion.ingest_url(
                        url
                    )
                )

            st.sidebar.success(
                f"{number_of_chunks} chunks indexed."
            )

        except Exception as e:

            st.sidebar.error(
                f"URL ingestion failed: {e}"
            )


# ============================================================
# Wikipedia
# ============================================================

st.sidebar.subheader(
    "Wikipedia"
)

wiki_topic = st.sidebar.text_input(
    "Wikipedia topic",
)


if st.sidebar.button(
    "Ingest Wikipedia"
):

    if not wiki_topic.strip():

        st.sidebar.warning(
            "Please enter a topic."
        )

    else:

        try:

            with st.spinner(
                "Loading Wikipedia..."
            ):

                number_of_chunks = (
                    ingestion.ingest_wikipedia(
                        wiki_topic
                    )
                )

            st.sidebar.success(
                f"{number_of_chunks} chunks indexed."
            )

        except Exception as e:

            st.sidebar.error(
                f"Wikipedia ingestion failed: {e}"
            )


# ============================================================
# Redis Status
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "System Status"
)

try:

    if redis_client.ping():

        st.sidebar.success(
            "Redis: Connected"
        )

except Exception:

    st.sidebar.error(
        "Redis: Not connected"
    )


# ============================================================
# Question
# ============================================================

st.header(
    "Ask a Question"
)

question = st.text_area(
    "Your question",
    placeholder=(
        "Ask a question about the "
        "knowledge base..."
    ),
    height=120,
)


# ============================================================
# Run Workflow
# ============================================================

if st.button(
    "Generate Answer",
    type="primary",
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Running Generator-Evaluator workflow..."
            ):

                result = workflow.run(
                    question.strip()
                )

            # ------------------------------------------------
            # Final Answer
            # ------------------------------------------------

            st.subheader(
                "Final Answer"
            )

            st.write(
                result["answer"]
            )

            # ------------------------------------------------
            # Evaluation
            # ------------------------------------------------

            st.subheader(
                "Evaluation"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Decision",
                result["decision"].upper(),
            )

            col2.metric(
                "Score",
                f"{result['score']:.2f}",
            )

            col3.metric(
                "Iterations",
                result["iteration"] + 1,
            )

            # ------------------------------------------------
            # Feedback
            # ------------------------------------------------

            with st.expander(
                "Evaluator Feedback"
            ):

                st.write(
                    result["feedback"]
                )

            # ------------------------------------------------
            # Retrieved Context
            # ------------------------------------------------

            with st.expander(
                "Retrieved Context"
            ):

                st.write(
                    result["context"]
                )

        except Exception as e:

            st.error(
                f"Workflow failed: {e}"
            )