import pyaudio
import wave
from third_party import Recorder


# returns wav file name, puts wav in wav/
def record_wav():
    file_name = raw_input('Enter wav file name (without .wav): ')
    length = input('Enter length in seconds: ')
    recorder = Recorder()
    with recorder.open('wav/' + file_name + '.wav', 'wb') as record_file:
        print('Recording...')
        record_file.record(length)
        print('Done.')
    return file_name


# plays wav file
def play_wav(file_name):
    print('Playing ' + file_name)
    chunk = 1024
    wf = wave.open(file_name, 'rb')
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=audio.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True)
    data = wf.readframes(chunk)
    while data != '':
        stream.write(data)
        data = wf.readframes(chunk)
    stream.close()
    audio.terminate()
    print('Done.')
