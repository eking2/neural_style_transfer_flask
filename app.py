from pathlib import Path
from flask import Flask, render_template
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField, IntegerField, TextAreaField, DecimalField
from wtforms.validators import DataRequired

basedir = Path(__file__).absolute()

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']
UPLOAD_FOLDER = Path(basedir, 'static/uploads')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdfasdjf1k@$@$@#Sfasf@@DaDA@#'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 mb max upload


class UploadForm(FlaskForm):

    weights = '''{"conv1_1" : 1,
"conv2_1" : 0.75,
"conv3_1" : 0.2,
"conv4_1" : 0.2,
"conv5_1" : 0.2}'''

    content = FileField('Content image', validators=[FileAllowed(ALLOWED_EXTENSIONS, 'Invalid image'),
                        FileRequired()])

    style = FileField('Style image', validators=[FileAllowed(ALLOWED_EXTENSIONS, 'Invalid image'),
                      FileRequired()])

    size = IntegerField('Max image size', default=400, validators = [DataRequired()])
    steps = IntegerField('Total steps', default=2_000, validators = [DataRequired()])
    alpha = DecimalField('Content recon weight (\u03B1)', default=1, validators = [DataRequired()])
    beta = DecimalField('Style recon weight (\u03B2)', default=1e6, validators = [DataRequired()])

    style_weights = TextAreaField('Style layer weights', default=weights, validators = [DataRequired()], render_kw={'rows' : 10, 'cols' : 40})
    submit = SubmitField('Submit')


@app.route('/', methods=['GET', 'POST'])
def home():

    form = UploadForm()

    return render_template('index.html', form=form)


if __name__ == '__main__':

    app.run(debug=True)
