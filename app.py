
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

def crop_image(img):
    target_size = np.min(img.shape[0:2])
    return img[0:target_size, 0:target_size, :]

def process_request_image(img):
    img_ = np.fromstring(img, np.uint8)
    img_ = cv2.imdecode(img_, cv2.IMREAD_COLOR)
    return crop_image(img_)

def create_data_uri(img):
    _, img_ = cv2.imencode('.jpg', img)
    img_ = img_.tobytes()
    img_ = base64.b64encode(img_)
    img_ = img_.decode()
    mime = "image/jpeg"
    return "data:%s;base64,%s" % (mime, img_)

def get_model():

    model = tf.keras.models.load_model('/model')

    return model


'# Blindness Detection'
# return render_template('index.html', prediction=None, img=None)

st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Choose an eye picture", type="jpg")


st.sidebar.write('# Model used')



if uploaded_file is not None:

    data = uploaded_file

    st.write(data)

    # Handle request
    # img = request.files['file'].read()

    # A tiny bit of preprocessing
    # img = process_request_image(img)

    # Convert image to data URI so it can be displayed without being saved
    # uri = create_data_uri(img)

    import cv2
    import numpy as np

    image_bytes = data.read()

    st.image(image_bytes, caption='Le Wagon', use_column_width=False)

    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

    # your Pillow code
    import io
    from PIL import Image
    img = np.array(Image.open(io.BytesIO(image_bytes)))




    # Convert to VGG16 input
    img = cv2.resize(img, (224, 224))

    img = np.reshape(img, [1, 224, 224, 3])

    model = get_model()

    # Classify image
    predictions = model.predict(img)
    print(predictions[0])
    a = list(predictions[0])
    max_index = a.index(max(a))
    print(max_index)
    print(a[max_index])
    classes = ['not sick', 'Mild Nonproliferative', 'Moderate Nonproliferative','Severe Nonproliferative', 'Proliferative Diabetic']
    labels = classes[max_index]

    if max_index == 0:
        st.markdown('# The retinopathy is at a stage: Non existant'
'''
> Your eyes are healthy


''')
        f'''Percentage of confidence :
     {round(a[max_index]*100)} %'''
    if max_index == 1:
        st.markdown('# The retinopathy is at a stage: Mild Nonproliferative'
'''
> Microaneurysms occur.
> They are small areas of balloon-like swelling in the retina's tiny blood vessels.''')
        f'''Percentage of confidence :
     {round(a[max_index]*100)} %'''


    if max_index == 2:
        st.markdown('# The retinopathy is at a stage: Moderate Nonproliferative'
            '''
> Blood vessels that nourish the retina are blocked.
> Areas of the retina send signals to the body to grow new blood vessels for nourishment.''')
        f'''Percentage of confidence :
     {round(a[max_index]*100)} %'''

    if max_index == 3:
        st.markdown('''# The retinopathy is at a stage: Severe Nonproliferative

> Many more blood vessels are blocked, depriving several areas of
> The retina with their blood supply. ''')
        f'''Percentage of confidence :
     {round(a[max_index]*100)} %'''







    # return render_template('index.html', prediction=labels, img=uri)
