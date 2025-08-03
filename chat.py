import pyaudio
import wave
import numpy as np
import openai
import time
from openai import OpenAI
import os
from tempfile import NamedTemporaryFile
from pydantic import BaseModel


client = OpenAI(api_key="")

def ask_image_cap(base64_image):
        print("Sending prompt to ChatGPT...")
        response = client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": "what's in this image?" },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ],
        )     
        chat_response = response.output_text
        return chat_response
def ask_chatgpt(prompt):
        print("Sending prompt to ChatGPT...")
        response = client.responses.parse(
            model="gpt-4o-2024-08-06",
            input =[
            {"role": "system", "content": (
                    "You are an assistant that interprets user speech for the Yanshee robot. "
                    "You will respond conversationally, and also output robot movement commands when appropriate.\n\n"
                    "When a user's prompt requires the robot to move, provide the following structured movement parameters:\n"
                    "- move_name: list[str] — the name of each movement\n"
                    "- move_direction: list[str] — the direction for each movement (if applicable)\n"
                    "- move_repeat: list[int] — how many times to repeat the movement\n"
                    "- move_speed: list[str] — Speed of motion execution：very slow/slow/normal/fast/very fast\n"
                    "- move_timestamp: list[int] — timestamp in milliseconds since start to execute the movement\n\n"
                    "All movement parameter lists must be of equal length and ordered to match each sequential movement.\n"
                    "If no movement is required for a response, set all movement parameters to null.\n\n"
                    "If the movement is asked five times like \"raise your hand five times\" you should return one raise movement and one repeat value of 5"
                    "if you get a prompt implying crouch or bow use the one that is relevant of both, only one of them not both"
                    "The japaness greeting is bowing, so if you get the instruction to give the japaness greeting, bow"
                    "always answer in english and only in english even if the prompt was is another language"
                    "Acceptable values for move_name and corresponding valid move_direction are:\n"
                    "move_name      move_direction\n"
                    "crouch  (no direction needed)\n"
                    "bow     (no direction needed)\n"
                    "raise          left/right/both\n"
                    "stretch        left/right/both\n"
                    "come on        left/right/both\n"
                    "wave           left/right/both\n"
                    "bend           left/right\n"
                    "turn around    left/right\n"
                    "walk           forward/backward/left/right\n"
                    "head           forward/left/right\n\n"
                    )
            },
            {"role": "user", "content": prompt},
            ],
            text_format = ChatResponse,
        )
        chat_response = response.output_parsed

        return chat_response


class ChatResponse(BaseModel):
    resp: str
    move_name: list[str]
    move_direction: list[str]
    move_repeat: list[int]
    move_speed: list[str]
    move_timestamp: list[int]


# ==== Configuration ====
THRESHOLD = 700      # Audio level threshold for detecting speech
CHUNK = 1024          # Size of audio buffer
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100          # Sample rate
SILENCE_DURATION = 2  # Seconds of silence before stopping recording

# ==== Helper Functions ====
def is_silent(data_chunk):
    audio_data = np.frombuffer(data_chunk, dtype=np.int16)
    return np.abs(audio_data).mean() < THRESHOLD

def record_audio_until_silence():
    print("Listening... Start speaking.")
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    silent_chunks = 0
    recording = False

    while True:
        data = stream.read(CHUNK)
        silent = is_silent(data)

        if not silent:
            if not recording:
                print("Speech detected. Recording...")
            recording = True
            silent_chunks = 0
            frames.append(data)
        elif recording:
            silent_chunks += 1
            frames.append(data)
            if silent_chunks > (SILENCE_DURATION * RATE / CHUNK):
                print("Silence detected. Stopping recording.")
                break

    stream.stop_stream()
    stream.close()
    p.terminate()

    return frames

def save_audio(frames, filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio(filepath):    
    
    print("Transcribing audio with Whisper...")
    with open(filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe", 
            file=audio_file
        ) 
        print(transcription)
        return transcription.text


# ==== Main Program ====
def action_on_chat():
    frames = record_audio_until_silence()

    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_path = temp_audio.name
        save_audio(frames, audio_path)

    try:
        transcription = transcribe_audio(audio_path)
        print(f"Transcribed Text: {transcription}")
        reply = ask_chatgpt(transcription)
        print(f"\nChatGPT Response:\n{reply}")
        return reply
    finally:
        os.remove(audio_path)
