from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the saved model and encoder during app initialization
with open('stress_model.pkl', 'rb') as f:
    model = pickle.load(f)
    encoder = pickle.load(f)

# Define routes for your existing HTML pages
@app.route('/',methods=["GET", "POST"])
def home():
    return render_template('home.html')

@app.route('/about',methods=["GET", "POST"])
def about():
    return render_template('about.html')

@app.route('/register',methods=["GET", "POST"])
def register():
    return render_template('register.html')  # Assuming a registration form

@app.route('/login',methods=["GET", "POST"])
def login():
    return render_template('login.html')  # Assuming a login form

# Route for prediction with form handling
@app.route('/survey', methods=['POST'])
def survey():
    anxiety_level = float(request.form['anxiety_level'])
    mental_health_history = float(request.form['mental_health_history'])
    depression = float(request.form['depression'])
    headache = float(request.form['headache'])
    sleep_quality = float(request.form['sleep_quality'])
    breathing_problem = float(request.form['breathing_problem'])
    living_conditions = float(request.form['living_conditions'])
    academic_performance = float(request.form['academic_performance'])
    study_load = float(request.form['study_load'])
    future_career_concerns = float(request.form['future_career_concerns'])
    extracurricular_activities = float(request.form['extracurricular_activities'])

    user_input = pd.DataFrame([[anxiety_level, mental_health_history, depression, headache,
                               sleep_quality, breathing_problem, living_conditions,
                               academic_performance, study_load, future_career_concerns,
                               extracurricular_activities]],
                              columns=['anxiety_level', 'mental_health_history', 'depression', 'headache',
                                       'sleep_quality', 'breathing_problem', 'living_conditions',
                                       'academic_performance', 'study_load', 'future_career_concerns',
                                       'extracurricular_activities'])

    # Preprocess user input (apply same encoding as used during training)
    user_input = encoder.transform(user_input)

    predicted_stress = model.predict(user_input)[0]

    return render_template('survey.html', prediction=predicted_stress)  # Replace with your HTML template

if __name__ == '__main__':
    app.run(debug=True)
