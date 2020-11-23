from logging import getLogger

logger = getLogger(__name__)

if __name__ == '__main__':
    logger.debug('hogehoge1')
    logger.info('hogehoge2')
    logger.error('hogehoge3')
    logger.critical('hogehoge4')
    pass
