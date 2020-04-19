import numpy as np
from IPython.display import Audio, Javascript
from scipy.io import wavfile
from google.colab import output
from base64 import b64decode


FRAMERATE = 44100
RECORD = """
const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader()
  reader.onloadend = e => resolve(e.srcElement.result)
  reader.readAsDataURL(blob)
})
var record = time => new Promise(async resolve => {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  recorder = new MediaRecorder(stream)
  chunks = []
  recorder.ondataavailable = e => chunks.push(e.data)
  recorder.start()
  await sleep(time)
  recorder.onstop = async ()=>{
    blob = new Blob(chunks)
    text = await b2text(blob)
    resolve(text)
  }
  recorder.stop()
})
"""


def record_audio_raw(sec=3):
    print("Start recording...")
    display(Javascript(RECORD))
    s = output.eval_js('record(%d)' % (sec*1000))
    b = b64decode(s.split(',')[1])
    print("Stop recording...")
    return b


def record_audio_in_file(file_name, sec=3):
    data = record_audio_raw(sec=sec)
    with open(file_name,'wb') as f:
        f.write(data)
    print("Created {} audio file".format(file_name))
    return data


def record_audio(sec=3):
    raw_data = record_audio_raw(sec=sec)
    return Audio(raw_data, rate=FRAMERATE, autoplay=False)


def load_audio(file_name):
    with open(file_name, "rb") as f:
        raw_data = f.read()
    return Audio(raw_data, rate=FRAMERATE, autoplay=False)