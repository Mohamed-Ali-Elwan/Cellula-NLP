import streamlit as st

from prompt import Prompt
from llm import LLM
from explain import Explain
from generate import Generate
from rag import RAG, embedding_model, persist_directory
from relevance_checker import RelevanceChecker
from updater import KnowledgeUpdater
from documents_loading import DocumentLoader
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="AI Coding Assistant",
    page_icon="",
    layout="wide"
)


# ============================================
# INITIALIZE LLM
# ============================================

llm = LLM.get_llm1()
chat= ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

# ============================================
# SESSION STATE
# ============================================

# Chat messages are only for displaying the
# current Streamlit conversation.
if "messages" not in st.session_state:
    st.session_state.messages = []


# Used when RAG says the retrieved knowledge
# is not relevant.
if "waiting_for_solution" not in st.session_state:
    st.session_state.waiting_for_solution = False


# Stores the original question that failed
# the relevance check.
if "failed_query" not in st.session_state:
    st.session_state.failed_query = None


# ============================================
# LOAD KNOWLEDGE BASE
# ============================================

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


if not st.session_state.documents_loaded:

    documents = DocumentLoader.load_base_data()

    rag = RAG(
        documents=documents,
        embedding_model=embedding_model,
        persist_directory=persist_directory
    )

    rag.create_vectorstore()

    st.session_state.rag = rag
    st.session_state.documents_loaded = True

else:

    rag = st.session_state.rag


# ============================================
# OTHER COMPONENTS
# ============================================

relevance_checker = RelevanceChecker(llm)

knowledge_updater = KnowledgeUpdater(
    embedding_model=embedding_model,
    persist_directory=persist_directory
)


# ============================================
# UI
# ============================================

st.title(" AI Coding Assistant")

st.caption(
    "Explain code, generate code, and retrieve "
    "programming knowledge."
)


# ============================================
# DISPLAY CHAT HISTORY
# ============================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ============================================
# USER INPUT
# ============================================

user_input = st.chat_input(
    "Ask a programming question..."
)


if user_input:

    # ========================================
    # DISPLAY USER MESSAGE
    # ========================================

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })


    # ========================================
    # CHECK USER SOLUTION
    # ========================================

    if st.session_state.waiting_for_solution:

        knowledge_updater.add_solution(
            query=st.session_state.failed_query,
            solution=user_input,
            language="python"
        )

        response = (
            "Thank you. I have added your solution "
            "to my knowledge base for future requests."
        )

        # Reset feedback state
        st.session_state.waiting_for_solution = False
        st.session_state.failed_query = None


    # ========================================
    # NORMAL REQUEST
    # ========================================

    else:

        # ====================================
        # INTENT CLASSIFICATION
        # ====================================

        classifier_prompt = Prompt.few_shot_prompt()

        formatted_classifier_prompt = (
            classifier_prompt.format(
                input=user_input
            )
        )

        classifier_response = llm.invoke(
            formatted_classifier_prompt
        )

        classification = (
            classifier_response.content
            .strip()
            .lower()
        )


        # ====================================
        # EXPLAIN CODE
        # ====================================

        if classification == "explain":

            response = Explain.explain_code(
                user_input
            )


        # ====================================
        # GENERATE CODE
        # ====================================

        elif classification == "generate":

            # --------------------------------
            # RAG RETRIEVAL
            # --------------------------------

            documents = rag.retrieve(
                user_input,
                k=5
            )


            # --------------------------------
            # RELEVANCE CHECK
            # --------------------------------

            relevance = relevance_checker.check(
                user_input,
                documents
            )


            # --------------------------------
            # RELEVANT
            # --------------------------------

            if relevance == "Relevant":

                generation_prompt = (
                    Prompt.relevant_prompt()
                )

                context = "\n\n".join(
                    document.page_content
                    for document in documents
                )

                formatted_prompt = (
                    generation_prompt.format(
                        query=user_input,
                        context=context
                    )
                )

                response = Generate.generate_code(
                    formatted_prompt
                )


            # --------------------------------
            # NOT RELEVANT
            # --------------------------------

            else:

                response = (
                    "I couldn't find relevant knowledge "
                    "for your request.\n\n"
                    "Please provide the correct solution "
                    "so I can learn it for future requests."
                )

                # The NEXT user message will be treated
                # as the solution.
                st.session_state.waiting_for_solution = True

                st.session_state.failed_query = user_input


        # ====================================
        # NEITHER
        # ====================================

        else:

            response = (
                "I'm an AI Coding Assistant. "
                "Please ask me a programming-related "
                "question."
            )


    # ========================================
    # DISPLAY AI RESPONSE
    # ========================================

    with st.chat_message("assistant"):

        st.write(response.data if hasattr(response, "data") else response)


    # ========================================
    # SAVE AI RESPONSE TO CHAT UI
    # ========================================

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })