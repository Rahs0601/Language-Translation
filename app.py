import streamlit as st

def main():
    st.title('Speech to Speech Translation Using Machine Learning')
    st.subheader('This is a demo of a speech to speech translation app using machine learning')

def KannadaInput():
    st.title('Kannada Input')
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Say something!")
        audio = r.listen(source)
        st.write("Time over, thanks")
    try:
        InputText = r.recognize_google(audio,language='kn-IN')
        st.write("You said " + InputText )
    except sr.UnknownValueError:
        st.write("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        st.write("Could not request results from Google Speech Recognition service; {0}".format(e))
    return InputText

def audiocreater(InputText):
    st.write('Creating Audio')
    st.write(InputText)
    from gtts import gTTS
    import os
    language = 'kn'
    myobj = gTTS(text=InputText, lang=language, slow=False)
    myobj.save("Kannada.mp3")
    os.system("Kannada.mp3")
    return 
    
def EnglishOutput(InputText):
    from deep_translator import GoogleTranslator
    outputText = GoogleTranslator(source='kn', target='en').translate(InputText)
    return outputText

def EnglishSpeech(outputText):
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(outputText)
    engine.runAndWait()
    return

if __name__ == '__main__':
    main()
    if st.button('Kannada Input'):
        kannada_input =KannadaInput()
        st.write(kannada_input)
        audiocreater(kannada_input)
        english_output = EnglishOutput(kannada_input)
        st.write(english_output)
        EnglishSpeech(english_output)