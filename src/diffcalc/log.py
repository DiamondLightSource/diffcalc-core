"""Logging configuration."""
import getpass
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
    filename="/tmp/diffcalc_%s.log" % getpass.getuser(),
    level=logging.DEBUG,
)
