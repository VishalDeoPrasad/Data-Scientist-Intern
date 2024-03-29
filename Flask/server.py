from flask import Flask, request, render_template

app = Flask(__name__) #intilize the flask object

#########################################################
#crate a route/endpoint and bind to some function
@app.route('/') 
def index():
    return render_template("/home_page.html")

@app.route('/magic')
def add_fun():
   var_1 = int(request.args.get('a'))
   var_2 = int(request.args.get('b'))
   return str(var_1+var_2)

########################################################

if __name__ == '__main__':
    app.run(debug=True)