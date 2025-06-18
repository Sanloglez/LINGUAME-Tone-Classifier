import csv
import datetime
import gradio.themes as themes
import gradio as gr
import joblib
import numpy as np
import os

from collections import Counter
from fpdf import FPDF
from langdetect import detect
from sentence_transformers import SentenceTransformer, util

feedback_log = []
history = []

custom_theme = themes.Base(
    primary_hue="purple",
    font=["Inter", "sans-serif"]
)

# Explanations per label
label_explanations = {
    "assertive_critical": {
        "interpretation": "This message expresses confident critique and points out issues directly.",
        "suggestion": "Consider softening the tone to maintain cooperation and avoid defensiveness."
    },
    "controlling": {
        "interpretation": "The message reflects a need for control, possibly limiting others’ autonomy.",
        "suggestion": "Try rephrasing to invite collaboration rather than imposing directions."
    },
    "empathetic_mature": {
        "interpretation": "This message shows emotional intelligence and clarity.",
        "suggestion": "No changes needed — the message is well-balanced and respectful."
    },
    "evasive": {
        "interpretation": "The message avoids addressing the core issue, which can create confusion.",
        "suggestion": "Consider being more direct and transparent to ensure clear communication."
    },
    "insecure_submissive": {
        "interpretation": "The message shows hesitation and lack of confidence.",
        "suggestion": "Try rephrasing with more assertiveness to build trust and credibility."
    },
    "manipulative_guilt": {
        "interpretation": "The tone appeals to guilt or obligation to influence others.",
        "suggestion": "Use open and honest communication instead of emotional pressure."
    },
    "overjustification": {
        "interpretation": "The message over-explains, possibly signaling defensiveness.",
        "suggestion": "Be concise and trust that your reasoning is clear without too much justification."
    },
    "passive_aggressive": {
        "interpretation": "The message hides resentment or disagreement behind polite language.",
        "suggestion": "Consider expressing concerns directly to avoid misunderstandings."
    },
    "professional_clear": {
        "interpretation": "The tone is neutral, structured, and respectful.",
        "suggestion": "No changes needed — this tone is appropriate for professional settings."
    },
    "veiled_reproach": {
        "interpretation": "The message implies criticism without expressing it directly.",
        "suggestion": "Try rephrasing the message to be more open and constructive."
    }
}

# Load model
clf = joblib.load("linguame_model.pkl")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Example inputs
examples = [
    ["Hi Sarah,\n\nI noticed that once again the report was submitted without the necessary attachments..."],
    ["Hello team,\n\nI'm happy to report that we’ve met our targets this quarter. I appreciate everyone's effort."],
    ["Sure, if that’s what you want. I guess it doesn’t matter what I think anyway."]
]

# Dummy labeled examples
examples_with_labels = [
    ("Hi Sarah,\n\nI noticed that once again the report was submitted without the necessary attachments...", "passive_aggressive"),
    ("Hello team,\n\nI'm happy to report that we’ve met our targets this quarter. I appreciate everyone's effort.", "empathetic_mature"),
    ("Sure, if that’s what you want. I guess it doesn’t matter what I think anyway.", "insecure_submissive")
]

# Classification function
def classify_text(text):
    try:
        lang = detect(text)
    except:
        return {}, "Could not detect language.", "Please enter a valid message."

    if lang != "en":
        return {}, "Unsupported language detected.", "Please submit your message in English."

    embedding = embedder.encode([text])
    probs = clf.predict_proba(embedding)[0]
    labels = clf.classes_

    top_index = np.argmax(probs)
    top_label = labels[top_index]
    history.append(top_label)
    explanation = label_explanations[top_label]

    prob_dict = {label: round(float(prob), 3) for label, prob in zip(labels, probs)}

    return prob_dict, explanation["interpretation"], explanation["suggestion"]

# Stats
def show_stats():
    if not history:
        return "No predictions yet."
    
    count = len(history)
    freq = Counter(history)
    most_common = freq.most_common(1)[0]
    
    stats_text = (
        f"🔢 Total predictions: {count}\n"
        f"🏆 Most frequent label: {most_common[0].replace('_', ' ').title()} ({most_common[1]} times)\n\n"
        "📊 Label distribution:\n"
    )
    
    for label, n in freq.most_common():
        stats_text += f"- {label.replace('_', ' ').title()}: {n}\n"
    
    return stats_text

# Feedback
def submit_feedback(text, predicted_label, is_correct):
    timestamp = datetime.datetime.now().isoformat()
    feedback_log.append({
        "timestamp": timestamp,
        "text": text,
        "predicted_label": predicted_label,
        "is_correct": is_correct
    })
    return "✅ Feedback received. Thank you!"

# Download Feedback
def download_feedback_csv():
    if not feedback_log:
        return None

    filename = "/tmp/feedback_log.csv"
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "text", "predicted_label", "is_correct"])
        writer.writeheader()
        for row in feedback_log:
            writer.writerow(row)

    return filename

# CLASSIFIER TAB with Blocks (logo + welcome + input)
with gr.Blocks(theme=custom_theme) as classifier_interface:
    gr.Image("logo.png", width=150, show_label=False)
    gr.Markdown("### 👋 Welcome to LINGUAME™\nPaste your message below to analyze its tone.")
    input_box = gr.Textbox(lines=3, label="Message", placeholder="Type your professional email or message here...")
    classify_btn = gr.Button("Classify")
    output_probs = gr.Label(num_top_classes=10, label="Predicted Probabilities")
    output_interp = gr.Textbox(label="Interpretation")
    output_sugg = gr.Textbox(label="Suggestion")
    classify_btn.click(fn=classify_text, inputs=input_box, outputs=[output_probs, output_interp, output_sugg])
    gr.Examples(examples=examples, inputs=input_box)

# STATS TAB
stats_interface = gr.Interface(
    fn=show_stats,
    inputs=[],
    outputs=gr.Textbox(label="Prediction Stats"),
    title="LINGUAME™ — Stats Dashboard",
    description="Live statistics of predictions made during this session.",
    theme=custom_theme
)

# FEEDBACK TAB
feedback_interface = gr.Interface(
    fn=submit_feedback,
    inputs=[
        gr.Textbox(label="Original Text", lines=3, placeholder="Paste the text you just classified"),
        gr.Textbox(label="Predicted Label", placeholder="Copy the predicted label shown above"),
        gr.Radio(["Correct", "Incorrect"], label="Was the prediction accurate?")
    ],
    outputs=gr.Textbox(label="System Response"),
    title="LINGUAME™ — Feedback",
    description="Help us improve the model by confirming or correcting its prediction.",
    theme=custom_theme
)

download_interface = gr.Interface(
    fn=download_feedback_csv,
    inputs=[],
    outputs=gr.File(label="Download Feedback CSV"),
    title="Download Feedback Data",
    description="Click the button to download all feedback collected during this session as a CSV file.",
    theme=custom_theme
)

# TABS
interface = gr.TabbedInterface(
    [classifier_interface, stats_interface, gr.TabbedInterface([feedback_interface, download_interface], ["📝 Submit Feedback", "⬇️ Download CSV"])],
    ["🔍 Classifier", "📊 Stats", "📝 Feedback"]
)

# LAUNCH
if __name__ == "__main__":
    interface.launch()



