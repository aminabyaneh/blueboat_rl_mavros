##########################################################################################
# This script reads the GLOBAL_POSITION_INT (GPS) and RAW_IMU messages from the MAVLink
# connection, prints them to the console, and writes them to CSV on exit.
# The UDP connection is pre-configured to work with the BlueBoat.

# NOTE: We successfully tested this using GCS_Client_Link port and IP.
# The boat-side endpoint was 192.168.2.2:14550 at the time, which this script listens for
# by binding locally with 'udpin:192.168.2.1:14550'. Both can be found in the MavLink
# Endpoints menu, in pirate mode. Override with --connection if your setup differs.

# NOTE: This script is fully tested. However, it requires a MAVLink connection
# to the boat to work. One can opt to deploy this on the boat itself, depending on the
# memory and processing power available.
##########################################################################################

import argparse
import os

import pandas as pd
from pymavlink import mavutil

# default endpoint: bind locally on the base-station side of the BlueBoat network
DEFAULT_CONNECTION = 'udpin:192.168.2.1:14550'


def main():
    parser = argparse.ArgumentParser(description="Log BlueBoat GPS and IMU messages to CSV.")
    parser.add_argument("folder_name", help="Sub-folder of logs/ to write the CSV files into.")
    parser.add_argument("--connection", default=DEFAULT_CONNECTION,
                        help=f"MAVLink connection string (default: {DEFAULT_CONNECTION}).")
    args = parser.parse_args()

    log_dir = os.path.join('logs', args.folder_name)
    os.makedirs(log_dir, exist_ok=True)

    # Start a connection listening to a UDP port
    print(f'Connecting to {args.connection} ...')
    connection = mavutil.mavlink_connection(args.connection)

    # Wait for the first heartbeat
    # This sets the system and component ID of remote system for the link
    connection.wait_heartbeat()
    print(f'Heartbeat from the system (system {connection.target_system} '
          f'component {connection.target_component})')

    # Collected rows, converted to DataFrames on exit
    gps_data = []
    odom_data = []

    print('Logging. Press Ctrl+C to stop and write the CSV files.')
    try:
        while True:
            msg_gps = connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
            if msg_gps:
                msg_gps = msg_gps.to_dict()
                print("\n-------------------------- GPS ------------------------------")
                print(msg_gps)
                print("---------------------------------------------------------------\n")

                gps_data.append({
                    'lat': msg_gps['lat'],
                    'lon': msg_gps['lon'],
                    'alt': msg_gps['alt'],
                    'time_boot_ms': msg_gps['time_boot_ms'],
                    'relative_alt': msg_gps['relative_alt'],
                    'vx': msg_gps['vx'],
                    'vy': msg_gps['vy'],
                    'vz': msg_gps['vz'],
                    'hdg': msg_gps['hdg'],
                })

            msg_odom = connection.recv_match(type='RAW_IMU', blocking=True)
            if msg_odom:
                msg_odom = msg_odom.to_dict()
                print("\n-------------------------- ODO ------------------------------")
                print(msg_odom)
                print("---------------------------------------------------------------\n")

                odom_data.append({
                    'xacc': msg_odom['xacc'],
                    'yacc': msg_odom['yacc'],
                    'zacc': msg_odom['zacc'],
                    'xgyro': msg_odom['xgyro'],
                    'ygyro': msg_odom['ygyro'],
                    'zgyro': msg_odom['zgyro'],
                    'time_usec': msg_odom['time_usec'],
                })
    except KeyboardInterrupt:
        print("Logging interrupted by user")
    finally:
        # always flush whatever was collected, including on an unexpected error
        timestamp = pd.Timestamp.now().strftime('%m%d_%H%M')
        gps_path = os.path.join(log_dir, f'gps_data_{timestamp}.csv')
        odom_path = os.path.join(log_dir, f'odom_data_{timestamp}.csv')

        pd.DataFrame(gps_data).to_csv(gps_path, index=False)
        pd.DataFrame(odom_data).to_csv(odom_path, index=False)
        print(f'Wrote {len(gps_data)} GPS rows to {gps_path}')
        print(f'Wrote {len(odom_data)} IMU rows to {odom_path}')


if __name__ == "__main__":
    main()
