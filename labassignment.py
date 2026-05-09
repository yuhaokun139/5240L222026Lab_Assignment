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
    prompt = (
        f"Look at the image uploaded by the user: {text}\n\n"
        f"Write a complete, happy, and simple children's story (50 to 100 words) for a kid aged 3-10 based only on the image. "
        f"The story must match the scene exactly. Do not overly repeat any words or phrases. "
        f"Make the story fun, easy to understand, and end nicely.\n\n"
        f"Do not add new characters or events not mentioned in the image.\n\n"
        f"Story: "
        )
    
    story_pipe = pipeline("text-generation", 
                          model="pranavpsv/genre-story-generator-v2")
    story_results = story_pipe(
        prompt, 
        max_new_tokens = 150,
        min_new_tokens = 80,
        do_sample = True,
        temperature = 0.4,
        repetition_penalty = 1.2,
        return_full_text = False
    )
    
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

        if st.button("Play Audio"):
            with st.spinner("Loading image..."):
                scenario = img2text(uploaded_file.name)
                st.write(f"**Scenario:** {scenario}")

            with st.spinner("Generating a story..."):
                story = text2story(scenario)
                st.write(f"**Story:** {story}")

            with st.spinner("Generating audio data..."):
                audio_data = text2audio(story)
                audio_array = audio_data["audio"]
                sample_rate = audio_data["sampling_rate"]
                st.audio(audio_array, sample_rate=sample_rate)

main()
