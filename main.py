from telegram.ext import *
from io import BytesIO
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

TOKEN = '6688871116:AAHypbVgAz9_5ny9Ls8mwMcyLjwbRh5Lpy8'

pretrained_model = load_model('grocery_model.h5')
classes = ['fruit','packs(milk,yoghurt)','vegetables']

def start(update, context):
    update.message.reply_text("Вас приветсвует бот Ашан GPT!")

def help(update, context):
    update.message.reply_text("""
                              /start - Start
                              /возврат - Возврат товара
                              /help - Show this message
                              """)
    
def handle_message(update,context):
    update.message.reply_text("Пожалуйста пришлите фото товара, который хотите вернуть")

def handle_photo(update,context):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()),dtype=np.uint8)
    
    img = cv2.imdecode(file_bytes,cv2.INTER_AREA)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    resize = image.resize(img, (150,150))
    
    y_pred = pretrained_model(np.expand_dims(resize/255,0))
    if np.max(y_pred)<0.4:
        update.message.reply_text('Пожалуйста загрузите фотографию правильно!')
    else:
        
        y_pred = np.argmax(y_pred)
        update.message.reply_text(f"Этот товар принадлежит категории {classes[y_pred]}")

app = Application.builder().token(TOKEN).build()

app.add_handler(CommandHandler("start",start))
app.add_handler(CommandHandler("help", help))
app.add_handler(MessageHandler(filters.TEXT, handle_message))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

app.run_polling()
app.idle()