import os
import asyncio
import logging
from handlers.handlers import Models
import handlers.handlers as handlers
from telegram.ext import CommandHandler, MessageHandler, InlineQueryHandler
from telegram.ext import ConversationHandler, CallbackQueryHandler
from telegram.ext import filters, ApplicationBuilder
from dotenv import load_dotenv


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def main():
    load_dotenv()
    TOKEN = os.getenv('TOKEN')

    application = ApplicationBuilder().token(TOKEN).build()

    start_handler = CommandHandler('start', handlers.start)

    menu_handler = CommandHandler('menu', handlers.menu)
    menu_button_callback_handler = CallbackQueryHandler(
        handlers.menu_button_callback
    )

    caps_handler = CommandHandler('caps', handlers.caps)

    model_handler = CommandHandler('model', handlers.model)
    model_filter = f"^({Models.NAIVEBYAS.value}|{Models.TFIDF_LOGREG.value})$"
    conv_handler = ConversationHandler(
        entry_points=[model_handler],
        states={
            0: [
                MessageHandler(
                    filters.Regex(
                        model_filter
                    ),
                    handlers.model_answer),
                MessageHandler(None, handlers.unknown_model)
            ],
        },
        fallbacks=[CommandHandler("cancel", handlers.cancel)],
    )

    echo_handler = MessageHandler(
        filters.TEXT & (~ filters.COMMAND), handlers.echo)
    inline_echo_handler = InlineQueryHandler(handlers.inline_echo)
    unknown_handler = MessageHandler(filters.COMMAND, handlers.unknown)

    application.add_handler(conv_handler)
    application.add_handler(start_handler)
    application.add_handler(menu_handler)
    application.add_handler(menu_button_callback_handler)
    application.add_handler(caps_handler)
    application.add_handler(echo_handler)
    application.add_handler(inline_echo_handler)
    application.add_handler(unknown_handler)

    application.run_polling()


if __name__ == '__main__':
    main()
