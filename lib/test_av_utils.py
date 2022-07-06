import unittest
import debugpy
import logging

logger = logging.getLogger(__name__)


class AVUtilTest(unittest.TestCase):
    pass


if __name__ == "__main__":

    # port = 5678
    # logger.info("Listening for incoming debug connection on port {}".format(
    #     5678))
    # debugpy.listen(("0.0.0.0", 5678))
    # debugpy.wait_for_client()

    print('foo')

    import sys

    print(sys.path)

    import lib
