import streamlit as st
import speech_recognition as sr
import time
r = sr.Recognizer()

def main():
    st.title('Speech to Speech Translation Using Machine Learning')
    st.subheader('This is a demo of a speech to speech translation app using machine learning')

def Voice():
    st.title('Recognizing Speech')
    with sr.Microphone() as source:
        st.write("Say something!")
        audio = r.listen(source)
        st.write("Time over, thanks")
    return audio

def kannadaAudioCreater(Text):
    from gtts import gTTS
    import os
    language = 'kn'
    myobj = gTTS(text=Text, lang=language, slow=False)
    myobj.save("Kannada.mp3")
    os.system("Kannada.mp3")
    return 

def EnglishSpeaker(Text):
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(Text)
    engine.runAndWait()
    return

#kannada Speech to English Speech
 
def KannadaInput(audio):
    try:
        InputText = r.recognize_google(audio,language='kn-IN')
        st.write("You said " + InputText )
    except :
        st.write("Your Audio Wasn't Clear Plz Try again")
    return InputText


def EnglishOutput(InputText):
    from deep_translator import GoogleTranslator
    outputText = GoogleTranslator(source='kn', target='en').translate(InputText)
    return outputText

#English Speech to Kannada Speech
def EnglishInput(audio):
    try:
        InputText = r.recognize_google(audio,language='en-IN')
        st.write("You said " + InputText )
    except :
        st.write("Your Audio Wasn't Clear Plz Try again")
    return InputText

def KannadaOutput(InputText):
    from deep_translator import GoogleTranslator
    outputText = GoogleTranslator(source='en', target='kn').translate(InputText)
    return outputText

if __name__ == '__main__':
    main()
    if st.button('Kannada Input'):
        audio = Voice()
        kannada_input =KannadaInput(audio)
        st.write(kannada_input)
        # kannadaAudioCreater(kannada_input)
        english_output = EnglishOutput(kannada_input)
        st.write(english_output)
        EnglishSpeaker(english_output)

    if st.button('English Input'):
        audio = Voice()
        english_input = EnglishInput(audio)
        st.write(english_input)
        # EnglishSpeaker(english_input)
        kannada_output = KannadaOutput(english_input)
        st.write(kannada_output)
        kannadaAudioCreater(kannada_output)
    

