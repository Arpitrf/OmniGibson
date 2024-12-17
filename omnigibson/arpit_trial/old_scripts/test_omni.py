import os
os.environ["CARB_TELEMETRY_ENABLE"] = "0"
os.environ["OMNIGIBSON_HEADLESS"] = "1"
os.environ["ISAAC_SIGHT_ENABLED"] = "0"

from omnigibson.simulator import Simulator
import time

def main():
    s = Simulator(mode='headless')
    time.sleep(1)
    s.disconnect()

if __name__ == "__main__":
    main()