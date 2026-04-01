import gradio as gr
from .rag import query_rag

def create_interface(pipeline):
    interface = gr.Interface(
        fn=lambda question: query_rag(question, pipeline),
        inputs=gr.Textbox(lines=2, placeholder="Posez votre question ici..."),
        outputs=[gr.Textbox(label="Réponse"), gr.Textbox(label="Sources")],
        title="Chatbot Médical",
        description="Posez des questions sur les documents médicaux."
    )
    return interface