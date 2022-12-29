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


def Voice():
    st.title("Recognizing Speech")
    with sr.Microphone() as source:
        st.write("Say something!")
        audio = r.listen(source)
        st.write("Time over, thanks")
    return audio


def KannadaAudioCreater(Text):
    from gtts import gTTS

    language = "kn"
    myobj = gTTS(text=Text, lang=language, slow=False)
    myobj.save("Kannada.mp3")
    audio_file = open("Kannada.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")
    return


def EnglishAudioCreater(Text):
    from gtts import gTTS

    language = "en"
    myobj = gTTS(text=Text, lang=language, slow=False)
    myobj.save("english.mp3")
    audio_file = open("english.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")
    return


def EnglishSpeaker(Text):
    import pyttsx3

    engine = pyttsx3.init()
    engine.say(Text)
    engine.runAndWait()
    return


# kannada Speech to English Speech


def KannadaInput(audio):
    try:
        InputText = r.recognize_google(audio, language="kn-IN")
        st.write("You said: " + "'" + InputText + "'")
    except:
        st.write("Your Audio Wasn't Clear Plz Try again")
    return InputText


def EnglishOutput(InputText):
    import evaluate_txt as e

    try:
        output = e.evaluate_eng(InputText)
        st.write("Translation: " + "'" + output + "'")
    except:
        st.write("Error Occurred while Translation, Please Try Again")
    return output


# English Speech to Kannada Speech


def EnglishInput(audio):
    try:
        InputText = r.recognize_google(audio, language="en-IN")
        st.write("You said: " + "'" + InputText + "'")
    except:
        st.write("Your Audio Wasn't Clear Plz Try again")
    return InputText


def KannadaOutput(InputText):
    import evaluate_txt as e

    try:
        output = e.evaluate_kan(InputText)
        st.write("Translation: " + "'" + output + "'")
    except:
        st.write("Error Occurred while Translation, Please Try Again")
    return output


if __name__ == "__main__":
    main()
    if st.button("Show Data Set"):
        data = pd.read_csv("eng-kannada.csv")
        st.write("Retrieving Data Set Plz Wait...")
        time.sleep(1)
        st.dataframe(data)

    if st.button("Kannada Input"):
        audio = Voice()
        st.write("Analyzing voice...")
        kannada_input = KannadaInput(audio)
        time.sleep(1)
        st.write("Translating...")
        english_output = EnglishOutput(kannada_input)
        time.sleep(1)
        st.write("Play this for audio...")
        EnglishAudioCreater(english_output)

    if st.button("English Input"):
        audio = Voice()
        st.write("Analyzing voice...")
        english_input = EnglishInput(audio)
        time.sleep(1)
        st.write("Translating...")
        kannada_output = KannadaOutput(english_input)
        time.sleep(1)
        st.write("Play this for audio...")
        KannadaAudioCreater(kannada_output)