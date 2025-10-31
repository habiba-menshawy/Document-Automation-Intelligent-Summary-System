import logging

class Logger:
    _logger = None

    @classmethod
    def get_logger(cls, name):
        if cls._logger is None:
            # Create a logger instance
            cls._logger = logging.getLogger("shared_logger")
            cls._logger.setLevel(logging.DEBUG)  # Set the default level
            
            # Avoid adding handlers multiple times
            if not cls._logger.hasHandlers():
                # Create handlers for file and console
                file_handler = logging.FileHandler("./logger/app.log")
                console_handler = logging.StreamHandler()

                # Set level for handlers
                file_handler.setLevel(logging.DEBUG)
                console_handler.setLevel(logging.DEBUG)

                # Create a formatter that includes the module name
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

                # Add formatter to handlers
                file_handler.setFormatter(formatter)
                console_handler.setFormatter(formatter)

                # Add handlers to the logger
                cls._logger.addHandler(file_handler)
                cls._logger.addHandler(console_handler)

        # Return a logger with the specified name (module name)
        return cls._logger.getChild(name)
