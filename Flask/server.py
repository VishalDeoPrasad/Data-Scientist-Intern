from flask import Flask, render_template

app = Flask(__name__) #intilize the flask object

@app.route('/') #crate a route/endpoint and bind to some function
def index():
    return render_template("home_page.html")

@app.route('/about')
def about():
    return "This is about Page"

if __name__ == '__main__':
    app.run()from flask import Flask, render_template

app = Flask(__name__) #intilize the flask object

@app.route('/') #crate a route/endpoint and bind to some function
def index():
    return render_template("home_page.html")