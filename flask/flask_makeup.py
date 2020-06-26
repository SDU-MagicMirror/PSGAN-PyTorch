from flask import Flask, request, render_template, Response

from data_loaders.makeup_utils import *
from solver_psgan import Solver_PSGAN

app = Flask(__name__)


@app.route('/generate_makeup', methods=['POST', 'GET'])
def generate_makeup():
    if request.method == 'POST':
        try:
            example_image = request.files.get('example_image')
            user_image = request.files.get('user_image')
            shade_alpha = request.form.get("shade_alpha")
            local_flag = request.form.get("local_flag")
            lip_flag = request.form.get("lip_flag")
            eye_flag = request.form.get("eye_flag")
            face_flag = request.form.get("face_flag")

            if str(local_flag) == '1':
                print('local flag')
                images = [user_image, example_image]
                mask2use = get_mask(Image.open(images[1]), lip_flag, eye_flag, face_flag)
                images = [Image.open(image) for image in images]
                images = [preprocess_image(image) for image in images]
                transferred_image = Solver_PSGAN.partial_test(*(images[0]), *(images[1]), *(images[0]), mask2use, shade_alpha=float(shade_alpha))
            else:
                images = [user_image, example_image]
                images = [Image.open(image) for image in images]
                images = [preprocess_image(image) for image in images]
                transferred_image = Solver_PSGAN.image_test(*(images[0]), *(images[1]), shade_alpha=float(shade_alpha))
            transferred_image.save('./static/images/transferred_image.png')

            with open('static/images/transferred_image.png', 'rb') as f:
                image = f.read()
                resp = Response(image, status=200, mimetype="image/png")
            return resp
        except Exception as e:
            print(str(e))
            resp = Response(response='Exception', status=500, content_type='text/html;charset=utf8')
            return resp

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
