# dla_tts

## Подготовка окружения
1. Создать venv: `python3 -m venv .venv`
2. Активировать venv: `source .venv/bin/activate`
3. Установить зависимости и скачать нужные данные: `bash setup.sh`
4. (только если нужно вызывать train) Подготовить данные для обучения `python3 preprocess.py`

## Запустить инференс

Только на машине с GPU.

`python3 synthesize.py -t 'I wish I started doing this homework earlier' --speed 0.7 --pitch 1.0 --energy 1.0 -o result.wav`

## Дополнительно

 - Чекпоинт скачивается автоматически, но есть тут https://disk.yandex.ru/d/GItNjWFyPum9Vg
 - Воспроизвести обучение можно, запустив `python3 train.py -c hw_tts/configs/config.json`
 - Проект в W&B https://wandb.ai/anismagilov/dla_tts
