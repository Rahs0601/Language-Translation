import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1


def tensorFromSentence(lang, sentence):
    indexes = [lang[word] for word in sentence.split(" ")]
    indexes.append(EOS_token)

    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def evaluate_eng(encoder, decoder, sentence, max_length=15):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang_kan, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] += encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        if topi.item() == EOS_token:
            decoded_words.append("<EOS>")
            break
        else:
            decoded_words.append(output_lang_eng[topi.item()])

        decoder_input = topi.squeeze().detach()
        sen = " ".join(decoded_words)

    return sen


def evaluate_kan(encoder, decoder, sentence, max_length=15):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang_eng, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] += encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        if topi.item() == EOS_token:
            decoded_words.append("<EOS>")
            break
        else:
            decoded_words.append(output_lang_kan[topi.item()])

        decoder_input = topi.squeeze().detach()
        sen = " ".join(decoded_words)

    return sen


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


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
    try:
        output = evaluate_eng(
            encoder, decoder, input_text , max_length=MAX_LENGTH
        )
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
        output = evaluate_kan(encoder, decoder, input_text, max_length=MAX_LENGTH)
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