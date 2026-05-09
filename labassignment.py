# Program title: Storytelling App

# Import part
import streamlit as st

# Program title: Storytelling App

# Import part
import streamlit as st
from transformers import pipeline

# function part
# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", 
                                   model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story
def text2story(text):
    story_pipe = pipeline("text-generation", 
                          model="pranavpsv/genre-story-generator-v2")
    story_results = story_pipe(text)
    story = story_results[0]['generated_text']
    return story
    
# text2audio
def text2audio(story_text):
    audio_pipe = pipeline("text-to-audio", 
                          model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story_text)
    return audio_data

# main part
def main():
    st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
    st.header("ISOM5240: Turn Your Image to Audio Story")
    
    uploaded_file = st.file_uploader("Select an Image...")
    
    if uploaded_file is not None:
# Save file locally
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        scenario = img2text(uploaded_file.name)
        story = text2story(scenario)
        
        if st.button("Play Audio"):
            audio_data = text2audio(story)
            audio_array = audio_data["audio"]
            sample_rate = audio_data["sampling_rate"]
            st.audio(audio_array, sample_rate=sample_rate)

main()
