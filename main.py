"""
Зависимости:
apt install python3-pip
pip3 install ffmpeg
pip3 install pydub
pip3 install vosk
pip3 install torch
pip3 install transformers
"""

import json
import os
import subprocess

from pydub import AudioSegment
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(0)

# Проверяем наличие моделей.
if not os.path.exists('model'):
    print("Please download the model from https://alphacephei.com/vosk/models "
          "and unpack as 'model' in the current "
          "folder.")
    exit(1)

if not os.path.exists('recasepunc'):
    print("Please download the model from https://alphacephei.com/vosk/models "
          "and unpack as 'Punctuation models' in the current "
          "folder.")
    exit(1)

# Устанавливаем Frame Rate.
print('---Устанавливаем Frame Rate---')
FRAME_RATE = 16000
CHANNELS = 1

model = Model('model')
rec = KaldiRecognizer(model, FRAME_RATE)
rec.SetWords(True)

# Предобработка аудио.
print('---Предобработка аудио---')
mp3 = AudioSegment.from_mp3('song.mp3')
mp3 = mp3.set_channels(CHANNELS)
mp3 = mp3.set_frame_rate(FRAME_RATE)

# Преобразуем вывод в text.
rec.AcceptWaveform(mp3.raw_data)
result = rec.Result()
text = json.loads(result)['text']

# Добовляем пунктуацию.
print('---Добовляем пунктуацию---')
cased = subprocess.check_output("python recasepunc/recasepunc.py predict recasepunc/checkpoint", shell=True, text=True,
                                input=text, encoding='utf-8')

# Записываем результат в файл.
print('---Записываем результат в файл--')
with open('data.txt', 'w') as f:
    json.dump(cased, f, ensure_ascii=False, indent=4)


def main():
    print('Работа окончена :)')


if __name__ == '__main__':
    main()
