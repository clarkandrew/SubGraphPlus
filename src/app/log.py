import logging
import os
import datetime
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich.rule import Rule
# Define output directory locally to avoid circular imports
DEFAULT_OUTPUT_DIR = "."
from rich.markup import escape
DEBUG_MODE = False
# Initialize Rich Console with a theme
CONSOLEx = Console(color_system="standard", theme=Theme({
    "info": "green1",
    "debug": "magenta",
    "warning": "yellow",
    "error": "bold italic red",
    "critical": "bold white on red3"
}))

# Configuration Constants
if DEBUG_MODE == True:
    CONSOLEx.log(DEBUG_MODE, True)
    CONSOLE_LOG_LEVEL = logging.DEBUG
    FILE_LOG_LEVEL = logging.DEBUG
else:
    CONSOLEx.log(DEBUG_MODE, False)
    CONSOLE_LOG_LEVEL = logging.INFO
    FILE_LOG_LEVEL = logging.DEBUG

class CustomFormatter(logging.Formatter):
    formats = {
        logging.INFO: "[green1]%(message)s",
        logging.DEBUG: "[magenta]%(message)s",
        logging.WARNING: "[black on yellow]%(message)s",
        logging.ERROR: "[bold italic red]%(message)s",
        logging.CRITICAL: "[bold white on red3]%(message)s",
    }

    def format(self, record):
        self._style._fmt = self.formats.get(record.levelno, self._style._fmt)
        return super().format(record)
# Singleton Meta Class for Logger
class SingletonType(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

# Singleton Logger Class
class Logger(metaclass=SingletonType):
    def __init__(self):
        self.log_file_path = self._get_log_file_path()
        self.logger = self._setup_logger()

    def _get_log_file_path(self):
        # Generate a log file path with a timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return os.path.join(DEFAULT_OUTPUT_DIR, f"logs/{timestamp}.log")

    def _setup_logger(self):
        # Setup the main logger
        logger = logging.getLogger("main")
        logger.setLevel(CONSOLE_LOG_LEVEL)  # Capture all messages
        logger.handlers = []

        # Setup File Handler with a standard format
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(FILE_LOG_LEVEL)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        # Setup RichHandler for the console with the custom formatter
        console_handler = RichHandler(console=CONSOLEx, show_time=True, show_level=True, tracebacks_show_locals=True, rich_tracebacks=True,markup=True)
        console_handler.setLevel(CONSOLE_LOG_LEVEL)
        console_handler.setFormatter(CustomFormatter())

        # Add both handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.propagate = False
        return logger

    def get_logger(self):
        return self.logger

# Initialize and use the logger
logger = Logger().get_logger()
current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
logger.info(f"Started at {current_timestamp}")
if DEBUG_MODE:
    CONSOLEx.print(Rule())
    logger.debug("DEBUG MODE ENABLED")
    # logger.error("This is an error message")
    # logger.warning("This is a warning message")
    # logger.debug("This is a debug message")
    # logger.info("This is an info message")
    CONSOLEx.print(Rule())

def log_and_print(text, level="DEBUG"):
    if text:
        text = str(text)
        if len(text) > 200:
            mid = len(text) // 2 - 10 # middle index of the string (minus 10 to get the center)
            print_text = f"{text[:100]}...{text[mid:mid+20]}...{text[-80:]}"
        if level == "INFO":
            logger.info(text)
        elif level == "DEBUG":
            logger.debug(text)

        elif level == "WARNING":
            logger.warning(text)

        elif level == "ERROR":
            logger.error(text)

        elif level == "CRITICAL":
            logger.critical(text)

        else:
            logger.error(f"Unsupported log level: {escape(level)}\n{escape(text)}")

            CONSOLEx.print(f"Unsupported log level: {escape(level)}\n{escape(text)}")
        try:
            CONSOLEx.print(text)  # Assuming CONSOLEx has a write method that accepts newlines
        except:
            CONSOLEx.print(escape(text))
            pass
    else:
        logger.warning("text is none")
