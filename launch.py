"""launcher"""
from asyncio.log import logger
from utils import Parser
from runner import Runner
import logging

logger = logging.getLogger(__name__)


def main():
    """main entry"""
    config = Parser().config
    logger.info("launching runner ...")
    logger.info(f"configs : {config}")
    runner = Runner(config)
    if config.setup.do_train:
        runner.train()
    if config.setup.do_predict:
        runner.predict()
    logger.info("runner finished! ᕦ(･ㅂ･)ᕤ")


if __name__ == "__main__":
    main()
