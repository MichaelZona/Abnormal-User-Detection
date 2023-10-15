import logging
# create logger with name
# if not specified, it will be root
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# create a handler, write to log.txt
# logging.FileHandler(self, filename, mode='a', encoding=None, delay=0)
# A handler class which writes formatted logging records to disk files.
fh = logging.FileHandler('log.txt')
fh.setLevel(logging.DEBUG)

# create another handler, for stdout in terminal
# A handler class which writes logging records to a stream
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)

# set formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
sh.setFormatter(formatter)

# add handler to logger
logger.addHandler(fh)
logger.addHandler(sh)

# log it
logger.debug('Debug')
logger.info('Info')