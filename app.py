import streamlit as st
import speech_recognition as sr
import pandas as pd
import time

r = sr.Recognizer()


def main():
    st.title("Speech to Speech Translation Using Machine Learning")
    st.subheader(
        "This is a demo of a speech to speech translation app using machine learning"
    )
    st.markdown(
        """
        To use this app, select whether you want to input speech in English or Kannada, then click the "Start Input" button.
        Speak into your microphone to input your speech, and the app will recognize and translate it to the other language.
        The translated output will be played for you as audio.
        """
    )


def voice_capture():
    st.title("Recognizing Speech")
    with sr.Microphone() as source:
        st.write("Say something!")
        audio = r.listen(source)
        st.write("Time over, thanks")
    return audio


def kannada_audio_creater(text):
    from gtts import gTTS

    language = "kn"
    myobj = gTTS(text=text, lang=language, slow=False)
    myobj.save("Kannada.mp3")
    audio_file = open("Kannada.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")


def english_audio_creater(text):
    from gtts import gTTS

    language = "en"
    myobj = gTTS(text=text, lang=language, slow=False)
    myobj.save("english.mp3")
    audio_file = open("english.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")


def kannada_input(audio):
    try:
        input_text = r.recognize_google(audio, language="kn-IN")
        st.write("You said: " + "'" + input_text + "'")
    except:
        st.write("Your Audio Wasn't Clear Plz Try again")
    return input_text


def english_output(input_text):
    import evaluate_txt as e

    try:
        output = e.evaluate_eng(input_text)
        st.write("Translation: " + "'" + output + "'")
    except:
        st.write("Error Occurred while Translation, Please Try Again")
    return output


def english_input(audio):
    try:
        input_text = r.recognize_google(audio, language="en-IN")
        st.write("You said: " + "'" + input_text + "'")
    except:
        st.write("Your Audio Wasn't Clear Plz Try again")
    return input_text


def kannada_output(input_text):
    import evaluate_txt as e

    try:
        output = e.evaluate_kan(input_text)
        st.write("Translation: " + "'" + output + "'")
    except:
        st.write("Error Occurred while Translation, Please Try Again")
    return output


if __name__ == "__main__":
    main()
    input_language = st.selectbox("Select Input Language", ["Kannada", "English"])
    if st.button("Start Input"):
        audio = voice_capture()
        st.info("Analyzing voice...")
    if input_language == "English":
        try:
            english_input = english_input(audio)
            time.sleep(1)
            st.info("Translating...")
            kannada_output = kannada_output(english_input)
            time.sleep(1)
            st.info("Playing translated audio...")
            kannada_audio_creater(kannada_output)
        except:
            pass
    else:
        try:
            kannada_input = kannada_input(audio)
            time.sleep(1)
            st.info("Translating...")
            english_output = english_output(kannada_input)
            time.sleep(1)
            st.info("Playing translated audio...")
            english_audio_creater(english_output)
        except:
            pass