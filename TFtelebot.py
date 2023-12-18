import telebot
import traceback
import config
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np

bot = telebot.TeleBot(config.TOKEN)

classes = ['AVANTE', 'CASPER', 'EV6', 'G70', 'GRANDEUR', 'GV60', 'IONIQ5', 'IONIQ6', 'K5', 'K8', 'K9', 'KONA',
           'Korando', 'Morning', 'NEXO', 'NiroEV', 'PALISADE', 'Ray', 'Rexton', 'SANTAFE', 'SONATA', 'STARIA', 'Seltos',
           'Sorento', 'Sportage', 'Stinger', 'TUCSON', 'Tivoli', 'Torres', 'VENUE', 'qm6', 'sm6', 'xm3']

model = tf.keras.models.load_model('cars.h5')


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id,
                     'Привет! Пришли фото сюда, а нейронная сеть определит модель авто.')


@bot.message_handler(content_types=['photo'])
def repeat_all_messages(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        with open("image.jpg", 'wb') as new_file:
            new_file.write(downloaded_file)

        image = Image.open("image.jpg")
        size = (128, 128)
        image = image.convert("RGB")
        image = ImageOps.fit(image, size, Image.LANCZOS)
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array).flatten()
        predictions = np.where(predictions < 0.5, 0, 1)

        bot.send_message(message.chat.id, text=f'На этом фото {classes[predictions[0]]}')


    except Exception as e:
        traceback.print_exc()
        bot.send_message(message.chat.id, 'Упс, что-то пошло не так')

bot.polling(none_stop=True)