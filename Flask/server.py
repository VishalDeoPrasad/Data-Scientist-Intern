from flask import Flask

app = Flask(__name__) #intilize the flask object

@app.route('/') #crate a route/endpoint and bind to some function
def index():
    return "Welcome to this application"

@app.route('/about')
def about():
    return "This is about Page"

if __name__ == '__main__':
    app.run()