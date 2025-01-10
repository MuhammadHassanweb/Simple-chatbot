# Install necessary libraries
!pip install streamlit transformers --quiet

# Create the Streamlit app
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained model and tokenizer
@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-medium"  # You can choose small, medium, or large
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit app
def main():
    st.title("🤖 Chatbot with DialoGPT")
    st.write("Hi! I'm a chatbot powered by the DialoGPT model. Type a message below to start chatting!")
    
    # Session state to store chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = None

    # User input
    user_input = st.text_input("You:", key="input")
    
    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            # Encode user input and append to chat history
            new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
            chat_history_ids = (
                new_input_ids
                if st.session_state["chat_history"] is None
                else torch.cat([st.session_state["chat_history"], new_input_ids], dim=-1)
            )
            st.session_state["chat_history"] = chat_history_ids

            # Generate response
            response_ids = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            bot_response = tokenizer.decode(response_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

            # Display the chat
            st.text_area("Chat:", value=f"You: {user_input}\nBot: {bot_response}", height=200)
    
    st.markdown("Type 'exit' to reset the conversation.")

if __name__ == "__main__":
    main()
# Simple-chatbot
