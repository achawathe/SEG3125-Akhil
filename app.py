import gradio as gr
import os
from groq import Groq

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Initialize conversation history
conversation_history_flashcards = []
notes_history = []


def chat_with_bot_stream(user_input):
    global conversation_history_flashcards
    conversation_history_flashcards.append({"role": "user", "content": user_input})
    
    if len(conversation_history_flashcards) == 1:
        conversation_history_flashcards.insert(0, {
            "role": "system",
            "content": "You are an AI-powered flashcard creator that helps students study efficiently. A student will a set of notes. You will turn them into effective Q&A-style flashcards. Each flashcard should have a well-structured question to prompt recall and a clear, accurate answer. Focus on essential information, simplify complex ideas, and prioritize foundational concepts for broad topics. Make sure to respond in the same language as the input."
        })
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation_history_flashcards,
        temperature=0.9,
        max_tokens=2048,
        top_p=0.5,
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
def summarize_notes(user_input, temperature=0.7):
    global notes_history
    notes_history.append({"role": "user", "content": user_input})
    
    if len(notes_history) == 1:
        notes_history.insert(0, {
            "role": "system",
            "content": "You are an AI-powered note summarizer designed to help students study efficiently. A student will provide either a topic or a set of notes. If given a topic, generate a structured and concise summary covering key concepts, definitions, and important details. If given notes, extract and condense essential information into a well-organized summary, ensuring clarity and coherence. Simplify complex ideas, prioritize foundational concepts for broad topics, and maintain the original language of the input. Aim for summaries that are easy to review and retain. In the Case you are given a topic, you can generate a structured and concise summary covering key concepts, definitions, and important details."
        })
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=notes_history,
        temperature=temperature,
        max_tokens=2048,
        top_p=1,
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
<h1>Flashcards</h1>
"""

TITLE2 = """
<style>
h1 { text-align: center; font-size: 24px; margin-bottom: 10px; }
</style>
<h1>Notes Summarizer</h1>
"""

with gr.Blocks(theme=gr.themes.Ocean(primary_hue="blue", secondary_hue="violet", neutral_hue="slate")) as flashcard_generator:
    with gr.Tabs():
        with gr.TabItem("Note Summarizer"):
            
            
            
            gr.HTML(TITLE2)
            chatbot1 = gr.Chatbot(label="Note Summarizer", placeholder="Paste or type your notes here for summarization.")
            temperature_slider = gr.Slider(minimum=0.5, maximum=1.0, step=0.05, value=0.7, label="How creative should the AI be? (0.5: safe, 1.0: creative)")
            examples = gr.Radio(
            choices=[
                "What are the key concepts in this text?",
                "Can you summarize this text in a few sentences?",
                "What are the main points of this text?",
                "Can you provide a brief summary of this text?",
                "Make a summary of the industrial revolution."
            ],
            label="Not sure where to start? 🤔 Explore GymBro’s powerful features with these questions:"
            )
            examples.change(
            fn=lambda x: chat_with_bot_stream(x, temperature_slider.value),
            inputs=example_questions,
            outputs=chatbot
            )
            
            with gr.Column():
                user_input = gr.Textbox(
                    label="Your Notes",
                    placeholder="Paste or type your notes here for summarization.",
                    lines=5
                )
                user_input.submit(summarize_notes, inputs=user_input, outputs=chatbot1, queue=True).then(
                    fn=lambda _: "",
                    inputs=user_input,
                    outputs=user_input
                )
                send_button = gr.Button("✋Summarize Notes")
            
            # Note summarization functionality
            send_button.click(
                fn=summarize_notes,
                inputs=user_input,
                outputs=chatbot1,
                queue=True
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=user_input
            )
        with gr.TabItem("Flashcard Generator"):
            gr.HTML(TITLE)
            chatbot2 = gr.Chatbot(label="Flashcard Creator")
            with gr.Column():
                user_input = gr.Textbox(
                    label="Your Message",
                    placeholder="What flashcards do you want to generate?",
                    lines=1
                )
                user_input.submit(chat_with_bot_stream, inputs=user_input, outputs=chatbot2, queue=True).then(
                fn=lambda _: "",
                inputs=user_input,
                outputs=user_input
                )
                send_button = gr.Button("✋Generate Flashcards")
            
            # Chatbot functionality
            send_button.click(
                fn=chat_with_bot_stream,
                inputs=user_input,
                outputs=chatbot2,
                queue=True
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=user_input
            )

demo.launch()