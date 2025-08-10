import streamlit as st
from main import brother

################## Create the Siderbar ########################3##

st.sidebar.title("About the App")

st.sidebar.write("""This app lets you explore Sri Lanka through an interactive chatbot that answers your questions
                based on trusted reference material. Learn about the island’s geography, history, culture, economy,
                people, traditions, and more — all in one place. Whether you’re a student, traveler, or just curious,
                you can discover Sri Lanka’s rich heritage and modern life in a simple, conversational way.""")

### New chat
new_chat = st.sidebar.button("🆕 Clear chat")

if new_chat:
    st.session_state.message = []
    

###################### Middle of the page ########################
st.title("💬 Sri Lanka Interactive Knowledge Assistant")


if "brother" not in st.session_state:
    st.session_state.brother = brother()


### Initialize the chat history
if "message" not in st.session_state:
    st.session_state.message = []

    
### Display the chat message history
for history_item in st.session_state.message:
    
    with st.chat_message(history_item["role"]):
        st.markdown(history_item["content"])     


### react to user input    
if user_input := st.chat_input("What is up?"):
        
    ### Display the user message in chat message container
    st.chat_message("user").markdown(user_input)
    
    ### Defining the assistant response
    response = st.session_state.brother.rag_chain.invoke({"input": user_input})
    assistant_response = response["answer"] 
    
    ### Display the assistant message in chat message container
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
      
    ### Add user input into message history  
    st.session_state.message.append({"role": "user", "content": user_input})  
    
    ### Add assistant responce into message history
    st.session_state.message.append({"role": "assistant", "content": assistant_response})
    



#1️⃣ st.chat_input("What is up?")
#    This is a Streamlit function that shows a chat-style text input box in your app.
#    The string "What is up?" is the placeholder text shown in the input box.
#    When the user types something and presses Enter, it returns the entered text (a string).
#    If the user hasn’t entered anything, it returns None.

#2️⃣ prompt := ... (Walrus operator)
#    := is the assignment expression operator (added in Python 3.8).
#    It assigns the returned value from st.chat_input() to the variable prompt while also checking it in the if condition.
#    This means:
#        If the user types something, prompt will get that text and the if condition will be True.
#        If the user hasn’t typed anything yet, prompt will be None and the if condition will be False.
   
  