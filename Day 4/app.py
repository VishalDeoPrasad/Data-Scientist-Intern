from flask import Flask, request, render_template

app = Flask(__name__) #intilize the flask object

#########################################################
#crate a route/endpoint and bind to some function
@app.route('/') 
def index():
    return render_template("home_page.html")

@app.route('/thankyou')
def thankyou_fun():
    return render_template("thankyou_page.html")
   

########################################################

if __name__ == '__main__':
    app.run(debug=True)