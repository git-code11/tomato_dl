import streamlit as st


st.set_page_config(layout="wide")

st.header("Tomato Prediction Application")

st.markdown("""
The project using an hybrid model which is an esemble of CNN and
Transformer predicition.

The growth stages of the tomato are:
1. Leaving Stage
2. Flowering Stage
3. Fruiting Stage
""")

# st.sidebar.markdown("# Main Page 🎉")
# st.sidebar.markdown("# Main Page 🎉")

with st.container(width=500, border=True, horizontal_alignment="center"):
    message_group = st.container(height=300, border=False)

    prompt = st.chat_input(
        "Enter text here...", accept_file=True, file_type=("jpg", "png"),)
    if prompt:
        history = st.session_state.get("messages", [])
        history.append(prompt)
        st.session_state['messages'] = history

    for i, message in enumerate(st.session_state.get("messages", []), start=1):
        message_group.write(f"{i}. {message.text}")
    # message_group.write(f"User has sent the following prompt: {prompt}")
