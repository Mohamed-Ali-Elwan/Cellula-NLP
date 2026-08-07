import streamlit as st
from prompt import Prompt
from llm import LLM
from explain import Explain


llm1=LLM.get_llm1()

st.title("AI Code Classifier")
st.set_page_config(page_title="AI Code Classifier", page_icon=":robot_face:", layout="wide")

st.file_uploader("Upload a file", type=["txt", "py", "ipynb", "md", "json", "xml", "html", "js", "css", "java", "c", "cpp", "rb", "go", "rs", "ts", "php"])
prompt = st.text_area("Enter your input here:", height=200)
if st.button("Submit"):
    if prompt:
        few_shot_prompt = Prompt.few_shot_prompt()

        formatted_prompt = few_shot_prompt.format(
            input=prompt
        )

        response = llm1.invoke(formatted_prompt)
        if response.content.strip() == "Explain":
            explanation = Explain.explain_code(prompt)
            st.subheader("Response")
            st.write(explanation.data)

       

    else:
        st.warning("Please enter input.")