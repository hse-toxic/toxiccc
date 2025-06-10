import logging
from enum import Enum
from models.model import predict, set_model, transform
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram import InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import ContextTypes
from telegram.ext import ConversationHandler
from uuid import uuid4


logger = logging.getLogger(__name__)


class Models(str, Enum):
    NAIVEBYAS = 'Ngrams_NaiveByas'
    TFIDF_LOGREG = 'tfidf_logreg'


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="I'm a bot, please talk to me!"
    )


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    result = await predict(update.message.text)
    if result == 1.0:
        msg = 'Toxic'
    else:
        msg = 'Non-toxic'
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f'{msg}'
    )


async def caps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg_caps = ' '.join(context.args).upper()
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg_caps
    )


async def detox(update: Update, context: ContextTypes.DEFAULT_TYPE):
    result = await transform(update.message.text)
    msg = result
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f'{msg}'
    )


async def inline_echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query.query
    if not query:
        return

    result = await predict(query)
    msg = f'Message "{query}" is {"toxic" if result == 1.0 else "non-toxic"}'

    results = []
    results.append(
        InlineQueryResultArticle(
            id=str(uuid4()),
            title='toxic?',
            input_message_content=InputTextMessageContent(msg)
        )
    )
    await context.bot.answer_inline_query(update.inline_query.id, results)


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Sorry, I didn't understand that command."
    )


async def model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reply_keyboard = [[Models.NAIVEBYAS, Models.TFIDF_LOGREG]]
    await update.message.reply_text(
        "Choose model",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard,
            resize_keyboard=True,
            one_time_keyboard=True,
            input_field_placeholder="model"
        ),
    )
    return 0


async def model_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    user_model = update.message.text
    logger.info(f'User {user} trying to set {user_model}')

    status_code, response = await set_model(user_model)
    logger.info(f'{status_code}, {response}')

    if status_code != 200:
        await update.message.reply_text(
            f'Error: {response}',
            reply_markup=ReplyKeyboardRemove()
        )
    else:
        await update.message.reply_text(
            f'Current model is set to {user_model}',
            reply_markup=ReplyKeyboardRemove()
        )

    return ConversationHandler.END


async def unknown_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    await update.message.reply_text(
        'Unknown model',
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    logger.info(f'User {user} cancelled model selection')
    await update.message.reply_text(
        'Cancelled model selection',
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END


async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    button1 = InlineKeyboardButton("ℹ️ Help", callback_data="help_button")
    button2 = InlineKeyboardButton(
        f'1️⃣ Select "{Models.NAIVEBYAS.value}" model',
        callback_data=Models.NAIVEBYAS.value
    )
    button3 = InlineKeyboardButton(
        f'2️⃣ Select "{Models.TFIDF_LOGREG.value}" model',
        callback_data=Models.TFIDF_LOGREG.value
    )
    keyboard = [[button1], [button2], [button3]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        '📋 Menu',
        reply_markup=reply_markup
    )


async def menu_button_callback(update: Update,
                               context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    button = query.data

    if button == "help_button":
        await query.edit_message_text(
            text="Available commands: /menu /model"
        )
    else:
        status_code, response = await set_model(button)
        logger.info(f'{status_code}, {response}')

        if status_code != 200:
            await query.edit_message_text(
                text=f'Error: {response}'
            )
        else:
            await query.edit_message_text(
                text=f'Current model is set to {button}'
            )
