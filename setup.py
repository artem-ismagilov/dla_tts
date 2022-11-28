
from yadisk import YaDisk

d = YaDisk()

print('downloading best model weights...')
d.download_public('https://disk.yandex.ru/d/GItNjWFyPum9Vg', 'fast_speech_2_best.pth')
print('download complete')
