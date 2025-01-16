import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit as st
from classical_painting_styles import *
from io import BytesIO
import tensorflow_hub as hub

# Tải mô hình Style Transfer từ TensorFlow Hub
hub_model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

STYLE_IMAGES = {
    "Classical Painting Styles": "style_images/classical_style.jpg",
    "Modern and Abstract Styles": "style_images/modern_abstract_style.jpg",
    "Cultural and Folk Styles": "style_images/cultural_folk_style_1.jpg",
    "Natural Styles": "style_images/natural_style_1.jpg",
    "Technological and Futuristic Styles": "style_images/technological_futuristic_style_2.jpg",
    "Vintage and Retro Styles": "style_images/vintage_retro_style.jpg",
    "Movie and Story Inspired Styles": "style_images/movie_storyInspired_style_3.jpg"
}

def main():
    st.header("Picture style conversion support tool")

    # Style selection
    style_option = st.selectbox(
        "Style Options:", 
        list(STYLE_IMAGES.keys())
    )

    # Image upload
    uploaded_file = st.file_uploader("Choose an image ...", type=["jpg", "jpeg", "png"])

    # st.markdown(
    #     """
    #     <style>
    #     .divider {
    #         height: 2px;
    #         background: linear-gradient(to right, red, yellow, green, cyan, blue, violet);
    #         border: none;
    #     }
    #     </style>
    #     <hr class="divider">
    #     """,
    #     unsafe_allow_html=True
    # )

    st.markdown(
        """
        <div style="width: 100%; height: 10px; background: linear-gradient(to right, #ff0000, #ffff00, #00ff00, #00ffff, #0000ff);">
        </div>
        """,
        unsafe_allow_html=True
    )

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)

        # Hiển thị ảnh gốc và ảnh kết quả trên cùng một hàng
        cols = st.columns(3)

        with cols[0]:
            st.image(image, caption="Original Image", use_container_width=True)

        # Add a button to start processing
        with cols[2]:
            if st.button("Apply Style Transfer"):
                try:
                    # Load style image
                    style_image_path = STYLE_IMAGES[style_option]
                    style_image = Image.open(style_image_path)

                    # Process images
                    with st.spinner("Processing..."):
                        content_tensor = load_and_process_image(image)
                        style_tensor = load_and_process_image(style_image)

                        # Perform style transfer using TensorFlow Hub model
                        result_tensor = hub_model(tf.convert_to_tensor(content_tensor), tf.convert_to_tensor(style_tensor))[0]

                        # Convert result to image
                        result_image = deprocess_image(result_tensor)

                        with cols[1]:
                            st.image(result_image.resize(image.size), caption="Stylized Image", use_container_width=True)

                        # Add download button for the result
                        buf = BytesIO()
                        result_image.save(buf, format="PNG")
                        st.download_button(
                            label="Download stylized image",
                            data=buf.getvalue(),
                            file_name="stylized_image.png",
                            mime="image/png"
                        )
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
