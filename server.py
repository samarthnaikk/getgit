from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    repo_link = None
    if request.method == 'POST':
        repo_link = request.form.get('repo_link')
    return render_template('index.html', repo_link=repo_link)

if __name__ == '__main__':
    app.run(debug=True)
