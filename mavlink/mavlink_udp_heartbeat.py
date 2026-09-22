##########################################################################################
# This test script establishes a MAVLink UDP link, waits for the first heartbeat, and then
# prints incoming GLOBAL_POSITION_INT (GPS) messages. Use it to confirm the link to the
# BlueBoat works before running anything else in this folder.

# IMPORTANT NOTE: We successfully tested this using GCS_Client_Link port and IP.
# The boat-side endpoint was 192.168.2.2:14550 at the time, which this script listens for
# by binding locally with 'udpin:192.168.2.1:14550'. Both can be found in the MavLink
# Endpoints menu, in pirate mode. Override with --connection if your setup differs.
##########################################################################################

import argparse

from pymavlink import mavutil

# default endpoint: bind locally on the base-station side of the BlueBoat network
DEFAULT_CONNECTION = 'udpin:192.168.2.1:14550'


def main():
    parser = argparse.ArgumentParser(description="Check the MAVLink UDP link to the BlueBoat.")
    parser.add_argument("--connection", default=DEFAULT_CONNECTION,
                        help=f"MAVLink connection string (default: {DEFAULT_CONNECTION}).")
    args = parser.parse_args()

    # Start a connection listening to a UDP port
    print(f'Connecting to {args.connection} ...')
    connection = mavutil.mavlink_connection(args.connection)

    # Wait for the first heartbeat
    # This sets the system and component ID of remote system for the link
    connection.wait_heartbeat()
    print(f'Heartbeat from the system (system {connection.target_system} '
          f'component {connection.target_component})')

    try:
        while True:
            msg = connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
            if msg:
                print(msg)
    except KeyboardInterrupt:
        print("Stopped by user")


if __name__ == "__main__":
    main()
