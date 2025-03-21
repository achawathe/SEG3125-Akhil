import datetime
import gradio as gr
import os
from groq import Groq

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Initialize conversation history
conversation_history_flashcards = []
notes_history = []

def download_chat_flashcards():
    global conversation_history_flashcards
    chat_text = "\n".join([
        f"User: {item['content']}" if item["role"] == "user" else f"Bot: {item['content']}"
        for item in notes_history
    ])
    file_path = f"conversation_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(file_path, "w") as f:
        f.write(chat_text)
    return file_path

def download_chat_notes():
    global notes_history
    chat_text = "\n".join([
        f"User: {item['content']}" if item["role"] == "user" else f"Bot: {item['content']}"
        for item in notes_history
    ])
    file_path = f"conversation_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(file_path, "w") as f:
        f.write(chat_text)
    return file_path

def chat_with_bot_stream(user_input, temperature=0.7, top_p=0.5):
    global conversation_history_flashcards
    conversation_history_flashcards.append({"role": "user", "content": user_input})
    
    if len(conversation_history_flashcards) == 1:
        conversation_history_flashcards.insert(0, {
            "role": "system",
            "content": "You are an AI-powered flashcard creator that helps students study efficiently. A student will a set of notes. You will turn them into effective Q&A-style flashcards. Each flashcard should have a well-structured question to prompt recall and a clear, accurate answer. Focus on essential information, simplify complex ideas, and prioritize foundational concepts for broad topics. Make sure to respond in the same language as the input. If the student provides a topic, generate a set of flashcards covering key concepts, definitions, and important details."
        })
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation_history_flashcards,
        temperature=temperature,
        max_tokens=2048,
        top_p=top_p,
        stream=True,
        stop=None,
    )
    
    response_content = ""
    for chunk in completion:
        response_content += chunk.choices[0].delta.content or ""
    
    conversation_history_flashcards.append({"role": "assistant", "content": response_content})
    
    return [(msg["content"] if msg["role"] == "user" else None, 
             msg["content"] if msg["role"] == "assistant" else None) 
            for msg in conversation_history_flashcards]

# Function to generate note summaries that can be used to create flashcards
def summarize_notes(user_input, temperature=0.7, top_p=0.5):
    global notes_history
    notes_history.append({"role": "user", "content": user_input})
    
    if len(notes_history) == 1:
        notes_history.insert(0, {
            "role": "system",
            "content": "You are an AI-powered note summarizer designed to help students study efficiently. A student will provide either a topic or a set of notes. If given a topic, generate a structured and concise summary covering key concepts, definitions, and important details. If given notes, extract and condense essential information into a well-organized summary, ensuring clarity and coherence. Simplify complex ideas, prioritize foundational concepts for broad topics, and maintain the original language of the input. Aim for summaries that are easy to review and retain. In the case you are given a topic, you can generate a structured and concise summary covering key concepts, definitions, and important details. Answer the user in the same language as the input."
        })
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=notes_history,
        temperature=temperature,
        max_tokens=2048,
        top_p=top_p,
        stream=True,
        stop=None,
    )
    
    response_content = ""
    for chunk in completion:
        response_content += chunk.choices[0].delta.content or ""
    
    notes_history.append({"role": "assistant", "content": response_content})
    
    return [(msg["content"] if msg["role"] == "user" else None, 
             msg["content"] if msg["role"] == "assistant" else None) 
            for msg in notes_history]

TITLE = """
<style>
h1 { text-align: center; font-size: 24px; margin-bottom: 10px; }
</style>
<h1>📚 Flashcards</h1>
"""

TITLE2 = """
<style>
h1 { text-align: center; font-size: 24px; margin-bottom: 10px; }
</style>
<h1>📃 Notes Summarizer</h1>
"""

with gr.Blocks(theme=gr.themes.Ocean(primary_hue="blue", secondary_hue="violet", neutral_hue="slate")) as flashcard_generator:
    with gr.Tabs():
        with gr.TabItem("Note Summarizer"):
            gr.HTML(TITLE2)
            chatbot1 = gr.Chatbot(label="📃 Note Summarizer", placeholder="Paste or type your notes here for summarization.")
            temperature_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.05, value=0.5, label="How creative should the AI be? (0.5: safe, 2.0: creative)",interactive=True)
            top_p_slider = gr.Slider(minimum=0, maximum=1, step=0.005, value=0.5, label="How much do you want it to take words from the text?",interactive=True)
            examples = gr.Radio(
            choices=[
                "What are the key concepts in this text?",
                "Can you summarize this text in a few sentences?",
                "What are the main points of this text?",
                "Can you provide a brief summary of this text?",
                "Make a summary of the industrial revolution."
            ],
            label="🙌 What would you like the AI to do?"
            )
            examples.change(
            fn=lambda x: summarize_notes(x, temperature_slider.value, top_p_slider.value),
            inputs=examples,
            outputs=chatbot1
            )
            
            with gr.Column():
                user_input = gr.Textbox(
                    label="Your Notes",
                    placeholder="Paste or type your notes here for summarization. Alternatively, ask a question or provide a topic.",
                    lines=5
                )
                user_input.submit(fn=lambda x: summarize_notes(x, temperature_slider.value, top_p_slider.value), inputs=user_input, outputs=chatbot1, queue=True).then(
                    fn=lambda _: "",
                    inputs=user_input,
                    outputs=user_input
                )
                send_button = gr.Button("✋Summarize Notes")
                download_btn = gr.Button("✍️ Download Chat")
            
            # Note summarization functionality
            send_button.click(
                fn=lambda x: summarize_notes(x, temperature_slider.value, top_p_slider.value),
                inputs=user_input,
                outputs=chatbot1,
                queue=True
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=user_input
            )
            download_btn.click(fn=download_chat_notes, inputs=[], outputs=gr.File())
            
        with gr.TabItem("Flashcard Generator"):
            gr.HTML(TITLE)
            
            with gr.Column():
                chatbot2 = gr.Chatbot(label="📚 Flashcard Creator", placeholder="Paste notes or prompt the chatbot with a topic!")
                temperature_slider = gr.Slider(minimum=0.5, maximum=2.0, step=0.005, value=0.5, label="How creative should the AI be? (0.5: safe, 2.0: creative)",interactive=True)
                top_p_slider = gr.Slider(minimum=0, maximum=1, step=0.005, value=0.5, label="How much do you want it to take words from the text?",interactive=True)
                
                examples = gr.Radio(
                choices=[
                    "Make flashcards for this text.",
                    "Generate flashcards for this topic.",
                    "Create flashcards for this content.",
                    "Create Multiple Choice Questions for this text."
                ],
                label="🙌 What would you like the AI to do?",
                )
                user_input = gr.Textbox(
                    label="Your Message",
                    placeholder="What flashcards do you want to generate?",
                    lines=1
                )
                user_input.submit(fn=lambda x: chat_with_bot_stream(x, temperature_slider.value, top_p_slider.value), inputs=user_input, outputs=chatbot2, queue=True ).then(
                fn=lambda _: "",
                inputs=user_input,
                outputs=user_input
                )
                send_button = gr.Button("✋Generate Flashcards")
                download_btn = gr.Button("✍️ Download Chat (This will be plaintext)")
            
            # Chatbot functionality
            send_button.click(
                fn=lambda x: chat_with_bot_stream(x, temperature_slider.value, top_p_slider.value),
                inputs=user_input,
                outputs=chatbot2,
                queue=True
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=user_input
            )
            download_btn.click(fn=download_chat_flashcards, inputs=[], outputs=gr.File())

flashcard_generator.launch()