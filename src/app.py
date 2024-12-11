import streamlit as st
import webbrowser
import helper
from streamlit.components.v1 import html

def open_page(url):
    open_script= """
        <script type="text/javascript">
            window.open('%s', '_blank').focus();
        </script>
    """ % (url)
    html(open_script)
    
st.title('Food Classifier :hamburger::pizza::spaghetti:  ')

st.write('Take a picture of the food and let us handle form there.')

# Image input
img_file_buffer = st.camera_input("Take a picture of the food.")
image = None
if img_file_buffer is not None:
    image = img_file_buffer
if not image:
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg', 'webp'])
    if uploaded_file:
        image = uploaded_file
        st.image(uploaded_file)

if image:
    food,prob = helper.predict_image(image)
    if prob<0.3:
        st.write(f'Food: {food} \n Probability: {prob:.2f}')
        st.warning('Image might not be a food or model might not know the food.', icon="⚠️")
    else:
        st.write(f'Food in the image: {food}\n Probability: {prob:.2f}')
        st.write(f'Hey there! Do you want to make your own {food}?')
    recipe_url = f'https://www.google.com/search?q={food.replace('_','+')}+recipe&ie=UTF-8&oe=UTF-8'
    
    st.button('Find Recipe', on_click=open_page(recipe_url))
    st.markdown(
        f'<a href="{recipe_url}" target="_blank" style="text-decoration: none; color: white; background-color: #4CAF50; padding: 10px 20px; border-radius: 5px;">Go to Example</a>',
        unsafe_allow_html=True
    )

footer_html = """<div style='margin-top:40px'>
  <p>Developed with ❤️ ❤️ by Team NoName.</p>
</div>"""
st.markdown(footer_html, unsafe_allow_html=True)
