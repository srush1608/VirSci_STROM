from flask import Flask, render_template, request
from workflow import create_workflow
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    final_abstract = ""
    
    if request.method == "POST":
        topic = request.form.get("topic")
        
        if topic:
            workflow = create_workflow()

            state_obj = {
                'topic': topic,
                's1_response': '',
                's2_response': '',
                's3_response': '',
                'final_abstract': '',
                'additional_notes': ''
            }

            app_flow = workflow.compile()
            result = app_flow.invoke(state_obj)

            final_abstract = result['final_abstract']
    
    return render_template("index.html", final_abstract=final_abstract)

if __name__ == "__main__":
    app.run(debug=True)
