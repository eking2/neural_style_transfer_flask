from pathlib import Path
from flask import Flask, render_template, send_from_directory, abort
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField, IntegerField, TextAreaField, DecimalField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from modules import nst, model, utils
from zipfile import ZipFile
import torch

basedir = Path.cwd().absolute()

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']
UPLOAD_FOLDER = Path(basedir, 'static/uploads')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdfasdjf1k@$@$@#Sfasf@@DaDA@#'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 mb max upload
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # disable caching


# # disable cache
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response


class UploadForm(FlaskForm):

    # default feature map weights
    weights = '''{"conv1_1" : 1,
"conv2_1" : 0.75,
"conv3_1" : 0.2,
"conv4_1" : 0.2,
"conv5_1" : 0.2}'''

    content = FileField('Content image', validators=[FileAllowed(ALLOWED_EXTENSIONS, 'Invalid content image'),
                        FileRequired()])

    style = FileField('Style image', validators=[FileAllowed(ALLOWED_EXTENSIONS, 'Invalid style image'),
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

    if form.validate_on_submit():

        # save images
        content_fn = secure_filename(form.content.data.filename)
        style_fn = secure_filename(form.style.data.filename)

        content_path = Path(app.config['UPLOAD_FOLDER'], content_fn)
        style_path = Path(app.config['UPLOAD_FOLDER'], style_fn)
        form.content.data.save(content_path)
        form.style.data.save(style_path)

        # get hyperparameters
        size = form.size.data
        steps = form.steps.data
        alpha = float(form.alpha.data)
        beta = float(form.beta.data)
        style_weights = form.style_weights.data

        # run nst and show intermediate images
        layers_df = utils.layers_from_json(style_weights)
        content, style = nst.resize_images(content_path, style_path, size)
        nst_model = model.VGG(layers_df)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        content = content.to(device)
        style = style.to(device)
        nst_model = nst_model.to(device)

        nst.style_transfer(content, style, nst_model, steps, alpha, beta, layers_df)

        # zip images to display
        results = [x for x in Path('static').iterdir() if x.suffix == '.png']
        results = sorted(results, key=lambda x: int(x.name.split('_')[1].split('.')[0]))

        with ZipFile('./static/outputs.zip', 'w') as zip:
            for fi in results:
                zip.write(fi, fi.name)

        return render_template('index.html', form=form, results=results)

    # delete old images
    old_images = [x for x in Path('static').iterdir() if x.suffix == '.png']
    old_uploads = Path(app.config['UPLOAD_FOLDER']).iterdir()
    [x.unlink() for x in Path('static').glob('*.zip')]
    [image.unlink() for image in old_images]
    [image.unlink() for image in old_uploads]

    return render_template('index.html', form=form, results=None)

@app.route('/download')
def download():
    return send_from_directory(directory='static', filename='outputs.zip', as_attachment=True)

if __name__ == '__main__':

    app.run(debug=True)
