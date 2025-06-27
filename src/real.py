import sys
import os
import logging
from tests import health_check
from input.hub import Hub
from drone.transmission import Transmission
from drone.balancer import Balancer


logger = logging.getLogger(__name__)

def start_drone():
  health_check()
  transmission = Transmission()
  hub = Hub(transmission)
  balancer = Balancer(transmission)
  hub.check()
  balancer.check()
  return hub, balancer

if __name__ == "__main__":
  try:
    hub, balancer = start_drone()
    hub.start()
    balancer.start()
    logger.info("Drone started")
  except Exception as e:
    logger.error(f"Error starting drone: {e}")





