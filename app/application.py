from flask import Flask, render_template, request, session, redirect, url_for
from app.components.retriever import create_qa_chain
from dotenv import load_dotenv
from markupsafe import Markup
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Secret key from environment variable
app.secret_key = os.getenv("SECRET_KEY", "carebot_default_secret")


# Convert newlines to HTML breaks
def nl2br(value):
    return Markup(value.replace("\n", "<br>\n"))


app.jinja_env.filters["nl2br"] = nl2br


# Load QA Chain only once when app starts
try:
    print("Loading QA Chain...")
    qa_chain = create_qa_chain()
    print("QA Chain loaded successfully.")
except Exception as e:
    print(f"Error initializing QA Chain: {e}")
    qa_chain = None


@app.route("/", methods=["GET", "POST"])
def index():

    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":

        user_input = request.form.get("prompt", "").strip()

        if user_input:

            messages = session["messages"]

            # Store user message
            messages.append({
                "role": "user",
                "content": user_input
            })

            try:

                if qa_chain is None:
                    raise Exception(
                        "QA Chain initialization failed. Check vector store, embeddings, or Groq API."
                    )

                response = qa_chain.invoke(
                    {
                        "query": user_input
                    }
                )

                result = response.get(
                    "result",
                    "No response generated."
                )

                # Clean unwanted HTML tags
                result = result.replace("<br>", "\n")
                result = result.replace("<br/>", "\n")
                result = result.replace("<br />", "\n")

                result = result.strip()

                messages.append(
                    {
                        "role": "assistant",
                        "content": result
                    }
                )

                session["messages"] = messages

            except Exception as e:

                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Error: {str(e)}"
                    }
                )

                session["messages"] = messages

        return redirect(url_for("index"))

    return render_template(
        "index.html",
        messages=session.get("messages", [])
    )


@app.route("/clear")
def clear():

    session.pop("messages", None)

    return redirect(url_for("index"))


@app.route("/health")
def health():
    return {
        "status": "running",
        "qa_chain_loaded": qa_chain is not None
    }


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )
