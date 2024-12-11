import streamlit as st
import webbrowser
import helper

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

def search_recipe(food):
    food= food.replace('_','+')
    webbrowser.open(f'https://www.google.com/search?q={food}+recipe&ie=UTF-8&oe=UTF-8')  # Go to example.com


if st.button(' Get Recipe '):
    search_recipe(food)


footer_html = """<div style='margin-top:40px'>
  <p>Developed with ❤️ ❤️ by Team NoName.</p>
</div>"""
st.markdown(footer_html, unsafe_allow_html=True)