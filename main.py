from flask import Flask, render_template, request , jsonify
from utils.actions import login_validation_check , selling_injection_in_sql , generate_response , signup_sql_injection ,compute_plan_agri
import os

from yolo.appledetection.appletrack import apple_count
from yolo.plantdiseasedetection.leaftraining import leaf_disease_detection
from yolo.weeddetection.weedtraining import weed_detection

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')  # Path to uploads folder relative to app.py
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/farmerlogin")
def farmerlogin():
    return render_template("farmerlogin.html")

@app.route("/buyerlogin")
def buyerlogin():   
    return render_template("buyerlogin.html")

@app.route("/farmerloginauth", methods=["POST"])
def farmerloginauth():
    number = request.form.get("number")
    password = request.form.get("password")
    validation_result = login_validation_check(number, password,"farmer")
    if validation_result:
        return render_template("homepage.html")
    else:
        return "Wrong password", 401

@app.route("/buyerloginauth", methods=["POST"])
def buyerloginauth():
    number = request.form.get("number")
    password = request.form.get("password")
    validation_result = login_validation_check(number, password, "buyer")
    if validation_result:
        return render_template("homepage.html")
    else:
        return "Wrong password", 401

@app.route("/buyersignup")
def buyersignup():
    return render_template("signup.html")

@app.route("/farmersignup")
def farmersignup():
    return render_template("signupfarmer.html")  

@app.route("/signupprocessfarmer", methods=["POST"])
def signupprocessfarmer():
    name = request.form.get('fname') + " " + request.form.get('lname')
    mobile_number = int(request.form.get('mobileno'))
    password = request.form.get('password')
    address = request.form.get('address')
    gender = request.form.get('gender')
    age = int(request.form.get('age'))
    dateofbirth = request.form.get('dob')
    email = request.form.get('email')
    blood_group = request.form.get('bloodgroup')
    unique_id = request.form.get('uniqueid')
    state = request.form.get('state')
    country = request.form.get('country')

    # Insert data into the database
    signup_sql_injection(name, mobile_number, password, address, gender, age, dateofbirth, email, blood_group, unique_id, state, country)
    return render_template("homepage.html")

@app.route("/signupprocessbuyer", methods=["POST"])
def signupprocessbuyer():
    return render_template("homepage.html")

@app.route("/newspage")
def newspage():
    return render_template("news.html")  

@app.route("/communicationpage")
def communicationpage():
    return render_template("communication.html") 

@app.route("/storepage")
def storepage():
    return render_template("store.html") 

@app.route("/quickstartpage")
def quickstartpage():
    return render_template("quickfarm.html") 

@app.route("/compute_plan",methods=['POST'])
def compute_plan():
    landMeasurements = request.form.get("landMeasurements")
    budget = request.form.get("budget")
    machinery = request.form.get("machinery")
    labours = request.form.get("labours")
    soilType = request.form.get("soilType")
    irrigationMethod = request.form.get("irrigationMethod")
    storageFacilities = request.form.get("storageFacilities")

    response = compute_plan_agri(landMeasurements,budget,machinery,labours,soilType,irrigationMethod,storageFacilities)
    return render_template('chatbot_response.html', user_input=response)

@app.route("/sellingpage")
def sellingpage():
    return render_template("selling_page.html") 

@app.route("/financialpage")
def financialpage():
    return render_template("financial.html")

@app.route("/Insurancepage")
def Insurancepage():
    return render_template("insurance.html")

@app.route("/loanpage")
def loanpage():
    return render_template("loan.html")

@app.route("/loanformpage")
def loanformpage():
    return render_template("loanform.html")

@app.route("/insuranceformpage")
def insuranceformpage():
    return render_template("insuranceform.html")

@app.route("/chatbotpage")
def chatbotpage():
    return render_template("chatbot.html")

@app.route("/dashboardpage")
def dashboardpage():
    return render_template("dashboard.html")


@app.route("/sellingprocess", methods=["POST"])
def sellingprocess():
    name = request.form.get("name")
    
    email = request.form.get("email")
    contact = request.form.get("contact")
    address = request.form.get("address")
    product_name = request.form.get("productName")
    product_type = request.form.get("productType")
    quantity = request.form.get("quantity")
    price = request.form.get("price")
    unique_id = request.form.get("uniqueId")
    password = request.form.get("password")
    
    if selling_injection_in_sql(name, email, contact, address, product_name, product_type, quantity, price, unique_id, password):
        return render_template("confirmation_post.html", 
                               name=name,
                               product_name=product_name, 
                               quantity=quantity, 
                               price=price, 
                               address=address,
                               unique_id=unique_id)
    
@app.route("/cvpage")
def cvpage():
    return render_template("cv.html")


@app.route('/leafbase', methods=['GET', 'POST'])
def leafbase():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            # Call the leaf model function here
            result = leaf_disease_detection(filepath)
            return render_template('leafresult.html', result=result)
    return render_template('upload_form.html', task='leaf')

@app.route('/weedbase', methods=['GET', 'POST'])
def weedbase():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            # Call the leaf model function here
            result = weed_detection(filepath)
            return render_template('weedresult.html', result=result)
    return render_template('upload_form.html', task='weed')

@app.route('/countbase', methods=['GET', 'POST'])
def countbase():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            # Call the leaf model function here
            result = apple_count(filepath)
            return render_template('countresult.html', result=result)
    return render_template('upload_form.html', task='count')

@app.route('/chatprocess', methods=['POST'])
def chatprocess():
    user_input = request.form['user_input'] 
    llm_type = request.form['llm_type'] 
    category = request.form['category']
    response = generate_response(user_input,llm_type,category)
    return render_template('chatbot_response.html', user_input=response)

if __name__ == "__main__":
    app.run(debug=True)
