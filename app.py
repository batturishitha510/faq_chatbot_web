from flask import Flask, request, render_template, session
import json
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load FAQ data
with open("faq_data.json", "r") as f:
    faq_data = json.load(f)

questions = [faq["Question"] for faq in faq_data]
answers = [faq["Answer"] for faq in faq_data]

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

cleaned_questions = [clean_text(q) for q in questions]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_questions)

# Fun facts
subject_facts = [
    "🐍 Python was named after Monty Python, not the snake!",
    "🌐 HTML stands for HyperText Markup Language.",
    "🎨 CSS lets you style HTML with fonts, colors, and layouts.",
    "⚡ JavaScript can make websites interactive and dynamic.",
    "☕ Java’s slogan used to be ‘Write once, run anywhere’.",
    "📦 Flask is a lightweight web framework in Python.",
    "🔧 'Pip' stands for 'Pip Installs Packages'!",
    "💻 Debugging is harder than writing the code itself.",
    "🧠 VS Code is one of the most popular editors for developers.",
    "🧩 GitHub hosts over 200 million code repositories worldwide.",
    "🖥️ The first computer bug was an actual moth stuck in a relay!",
    "📚 Python is used for AI, web apps, data science, and automation.",
    "🪄 CSS animations can be used to create interactive designs.",
    "🧮 Computers work in binary — just 1s and 0s!",
    "🎯 Flask apps can run locally or be deployed on the cloud.",
    "🚀 Google, Netflix, and Instagram all use Python heavily.",
    "🔢 Variables are like labeled boxes that store data in code.",
    "🧬 Machine Learning is a branch of Artificial Intelligence.",
    "💾 Always use version control (like Git) to save your progress!",
    "🧱 HTML forms the structure, CSS styles it, and JS makes it alive!",
    "🔍 In Python, indentation defines the structure of the code.",
    "🧑‍💻 Learning small projects helps you understand big concepts.",
    "🎓 ‘print(“Hello, World!”)’ is usually the first line beginners write.",
    "🔒 Never share your API keys or passwords in public code!",
    "🕸️ Web development is like building digital cities — one tag at a time.",
    "🧊 Lists, tuples, and dictionaries are core Python data types.",
    "🌍 The first website went live in 1991, created by Tim Berners-Lee.",
    "💬 Chatbots like this one are built using Natural Language Processing (NLP).",
    "📈 Python’s popularity keeps growing because it’s simple and powerful.",
    "🧾 HTML tags come in pairs — opening and closing — like <p> and </p>."
]

def get_answer(user_input):
    user_input_clean = clean_text(user_input)
    selected_facts = random.sample(subject_facts, 3)

    if not user_input_clean.strip():
        return random.choice(subject_facts), selected_facts, []

    user_vec = vectorizer.transform([user_input_clean])
    similarities = cosine_similarity(user_vec, X)

    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    if best_score > 0.2:
        best_answer = answers[best_match_index]
        related_indices = similarities.argsort()[0][-4:-1][::-1]
        related_questions = [
            questions[i] for i in related_indices if i != best_match_index
        ]
    else:
        best_answer = "I'm not sure. Can you rephrase your question?"
        related_questions = []

    return best_answer, selected_facts, related_questions

@app.route("/", methods=["GET", "POST"])
def home():
    if "chat_history" not in session:
        session["chat_history"] = []

    answer = ""
    related = []
    facts = random.sample(subject_facts, 4)

    if request.method == "POST":
        user_question = request.form["question"]
        answer, facts, related = get_answer(user_question)
        session["chat_history"].append({"question": user_question, "answer": answer})

    return render_template(
        "index.html",
        answer=answer,
        facts=random.sample(subject_facts, 4),
        related_questions=related,
        chat_history=session["chat_history"]
    )

if __name__ == "__main__":
    app.run()