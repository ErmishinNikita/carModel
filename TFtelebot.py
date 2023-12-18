import telebot
import traceback
import config
from handler import *
from keras.models import load_model
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np

bot = telebot.TeleBot(config.TOKEN)
classes = ['AVANTE', 'CASPER', 'EV6', 'G70', 'GRANDEUR', 'GV60', 'IONIQ5', 'IONIQ6', 'K5', 'K8', 'K9', 'KONA',
           'Korando', 'Morning', 'NEXO', 'NiroEV', 'PALISADE', 'Ray', 'Rexton', 'SANTAFE', 'SONATA', 'STARIA', 'Seltos',
           'Sorento', 'Sportage', 'Stinger', 'TUCSON', 'Tivoli', 'Torres', 'VENUE', 'qm6', 'sm6', 'xm3']
model = load_model('cars.h5')


def get_photo(message):
    photo = message.photo[1].file_id
    file_info = bot.get_file(photo)
    file_content = bot.download_file(file_info.file_path)
    return file_content


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id,
                     'Привет! Пришли фото сюда, а нейронная сеть определит есть ли здесь авто.')


@bot.message_handler(content_types=['photo'])
def repeat_all_messages(message):
    try:
        file_content = get_photo(message)
        image = byte2image(file_content)

        size = (128, 128)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict_on_batch(img_array).flatten()
        predictions = np.where(predictions < 0.5, 0, 1)

        bot.send_message(message.chat.id, text=f'На этом фото {classes[predictions[0]]}')

    except Exception:
        traceback.print_exc()
        bot.send_message(message.chat.id, 'Что-то пошло не так')


if __name__ == '__main__':
    import time

    while True:
        try:
            bot.polling(none_stop=True)
        except Exception as e:
            time.sleep(15)
            print('Restart!')
