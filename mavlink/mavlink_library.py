##########################################################################################
# MAVLink Communication Module
#
# This module provides a set of functions to interact with a MAVLink compatible vehicle.
# It includes functionalities for establishing a connection, sending commands, and
# managing missions.
#
# Original Author: Stian Bernhardsen (2024)
# Updated and modified: Amin Abyaneh (2025)
##########################################################################################


import csv
import math
import time
from time import localtime, strftime

from pymavlink import mavutil


####################################
# General Purpose MAVLink commands #
####################################

def ack(the_connection, keyword):
    """
    Informs if messages are received and acted upon.

    Args:
        the_connection (object): The connection object to communicate with.
        keyword (str): The type of message you want to acknowledge.

    Returns:
        None
    """

    message = the_connection.recv_match(type=keyword, blocking=True, timeout=1)
    print(f'\t MAVLINK - ACK - Message read: {message}')


def establish_heartbeat(udp):
    """
    Establishes a MAVLink connection and waits for the first heartbeat.

    This function starts a connection listening to a specified UDP port,
    waits for the first heartbeat from the MAVLink system, and prints
    the system and component ID of the remote system.

    Args:
        udp (str): The UDP endpoint to connect to (e.g., 'udp:127.0.0.1:14550').

    Returns:
        mavutil.mavlink_connection: The established MAVLink connection object.
    """

    # Start a connection listening to a UDP port
    print('MAVLINK - Trying to establish connection...')
    the_connection = mavutil.mavlink_connection(udp)

    # Wait for the first heartbeat
    #  This sets the system and component ID of remote systemf or the link
    the_connection.wait_heartbeat()
    print('MAVLINK - Heartbeat from the system (system %u component %u)' %
        (the_connection.target_system, the_connection.target_component))
    return the_connection


def arm_disarm(the_connection, param):
    """
    Arms or disarms the boat.

    Args:
        the_connection (object): The connection object to the MAVLink device.
        param (int): 'ARM' (1) to arm the boat or 'DISARM' (0) to disarm the boat.

    Returns:
        None
    """

    print('MAVLINK - Arming USV' if param else 'MAVLINK - Disarming USV')
    the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,
                                      mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, param, 0, 0, 0, 0, 0, 0)
    ack(the_connection, "COMMAND_ACK")


def get_gps(the_connection):
    """
    Retrieves the latest GPS reading from the given MAVLink connection.

    Args:
        the_connection: The MAVLink connection object from which to receive the GPS data.

    Returns:
        The latest GPS reading as a MAVLink message of type 'GLOBAL_POSITION_INT'.

    Raises:
        TimeoutError: If no GPS data is received within the specified timeout period.
    """

    return the_connection.recv_match(type='GLOBAL_POSITION_INT',blocking=True, timeout=1)


def wait_for_gps(the_connection):
    """
    Waits for a valid GPS message from the MAVLink connection.

    This function continuously loops until it receives a GPS message with valid coordinates
    (i.e., latitude and longitude are not zero). It considers (latitude = 0, longitude = 0) as invalid coordinates.

    Args:
        the_connection: The MAVLink connection object from which to receive GPS messages.

    Returns:
        The GPS message with valid coordinates.
    """

    gps = None
    while gps == None or (gps.lat == 0) or (gps.lon == 0): # GPS not received or invalid
        gps = the_connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    return gps


def set_mode(the_connection, MAV_MODE):
    """
    Sets the mode using the MAV_MODE enum.

    Args:
        the_connection (object): The connection object to the MAVLink device.
        MAV_MODE (int): The mode to set, as defined in the MAV_MODE enum.
                        See https://mavlink.io/en/messages/common.html#MAV_MODE for an overview.

    Returns:
        None
    """

    the_connection.set_mode(MAV_MODE)
    ack(the_connection, "COMMAND_ACK")


def takeoff(the_connection):
    """
    Runs takeoff command in MAVLINK.

    Args:
        the_connection (object): The connection object to the MAVLink device.

    Returns:
        None
    """

    print('MAVLINK - Running Takeoff command')

    # If you pass lat/long as 0 it uses the current position
    the_connection.mav.command_long_send(the_connection.target_system,
                                         the_connection.target_component, 0,
                                         mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                                         0, 0, 0, 0,
                                         0, 0, 0, 0)
    ack(the_connection, "COMMAND_ACK")


def land(the_connection):
    """
    Runs the land command in MAVLINK.

    Args:
        the_connection (object): The MAVLINK connection object.

    This function sends a MAV_CMD_NAV_LAND command to the MAVLINK connection,
    instructing the vehicle to land at its current position. The latitude and
    longitude parameters are set to 0, which means the vehicle will use its
    current position for landing.

    Note:
        Ensure that the connection object is properly initialized and connected
        to the vehicle before calling this function.

    """

    print('MAVLINK - Running Land command')

    the_connection.mav.command_long_send(the_connection.target_system,
                                         the_connection.target_component, 0,
                                         mavutil.mavlink.MAV_CMD_NAV_LAND,
                                         0, 0, 0, 0, 0, 0, 0, 0)
    ack(the_connection, "COMMAND_ACK")


def set_home(the_connection, home_location, altitude):
    """
    Sets the home location for the vehicle.

    Args:
        the_connection (object): The connection object to communicate with the vehicle.
        home_location (list): A list containing latitude and longitude. If set to [0, 0], the current location will be used.
        altitude (float): The altitude above sea level in meters.

    Returns:
        None
    """

    the_connection.mav.command_long_send(the_connection.target_system,
                                         the_connection.target_component,
                                         mavutil.mavlink.MAV_CMD_DO_SET_HOME, 1, 0, 0, 0, 0,
                                         home_location[0], # Latitude
                                         home_location[1], # Longitude
                                         altitude)         # Altitude
    ack(the_connection, "COMMAND_ACK")


def go_home(the_connection):
    """
    Commands the vehicle to return to its home position.

    Args:
        the_connection (object): The MAVLink connection object.

    Returns:
        None
    """

    ''' Goes to Home position.'''
    the_connection.mav.command_long_send(the_connection.target_system,
                                         the_connection.target_component,
                                         mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0, 242, 0, 0, 0, 0, 0, 0)
    ack(the_connection, "COMMAND_ACK")
    home_location = the_connection.recv_match(type=['HOME_POSITION'], blocking=True, timeout=1)
    latitude = home_location.latitude  # Latitude 10^7
    longitude = home_location.longitude # Longitude 10^7
    altitude = home_location.altitude  # Altitude 10^7
    the_connection.mav.send(mavutil.mavlink.MAVLink_set_position_target_global_int_message(
                                        10,
                                        the_connection.target_system,
                                        the_connection.target_component,
                                        0, # MAV_FRAME_GLOBAL = 0
                                        int(0b11011111100), # POSITION_TARGET_TYPEMASK
                                        latitude,
                                        longitude,
                                        altitude,
                                        0, # x_velocity in NED frame
                                        0, # y_velocity in NED frame
                                        0, # z_velocity in NED frame
                                        0, # x_acceleration in NED frame
                                        0, # y_acceleration in NED frame
                                        0, # z_acceleration in NED frame
                                        1.57, # yaw setpoint
                                        0.5)) # yaw rate setpoint

def read_homelocation(the_connection):
    """
    Reads the current home position of the vehicle.

    Note: Altitude is redundant for USVs.

    Args:
        the_connection (object): The MAVLink connection object.

    Returns:
        the WGS84-coordinates [latitude, longitude]
    """

    the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,
                                        mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0, 242, 0, 0, 0, 0, 0, 0)

    home_location = the_connection.recv_match(type=['HOME_POSITION'],blocking=True, timeout=1)
    latitude  = home_location.latitude  # Latitude 10^7
    longitude = home_location.longitude # Longitude 10^7
    altitude  = home_location.altitude  # Altitude 10^7
    return [latitude, longitude]


####################################
# Mission-related MAVLink commands #
####################################

class MissionItem():
    """
    A class to represent a mission item for MAVLink communication.

    Attributes:
    -----------
    seq : int
        Waypoint ID (sequence number). Starts at zero. Increases monotonically for each waypoint, no gaps in the sequence (0,1,2,3,4).
    frame : int
        MAV_FRAME_GLOBAL : Global (WGS84) coordinate frame + altitude relative to mean sea level (MSL).
    current : int
        Indicates if the waypoint is the current waypoint. false:0, true:1.
    autocontinue : int
        Autocontinue to next waypoint. 0: false, 1: true. Set false to pause mission after the item completes.
    command : int
        The scheduled action for the waypoint.
    param1 : float
        Hold time [s].
    param2 : float
        Accept Radius [m].
    param3 : float
        Pass radius [m].
    param4 : float
        Yaw [deg] (NaN to use the current system yaw heading mode).
    latitude : int
        Global: latitude in degrees * 10^7.
    longitude : int
        Global: longitude in degrees * 10^7.
    altitude : float
        Global: altitude in meters (relative or absolute, depending on frame).
    mission_type : int
        MAV_MISSION_TYPE assigned to main mission.

    Methods:
    --------
    __init__(self, seq, current, autocontinue, x, y):
        Constructs all the necessary attributes for the mission item object.
    """
    def __init__(self, seq, current, autocontinue, x, y):
        self.seq          = seq
        self.frame        = mavutil.mavlink.MAV_FRAME_GLOBAL
        self.current      = current
        self.autocontinue = autocontinue
        self.command      = mavutil.mavlink.MAV_CMD_NAV_WAYPOINT

        self.param1       = 0
        self.param2       = 2
        self.param3       = 15
        self.param4       = math.nan
        self.latitude     = int(x*10**7)
        self.longitude    = int(y*10**7)
        self.altitude     = 0
        self.mission_type = 0



def upload_mission(the_connection, mission_items):
    """
    Uploads a list of mission items to the MAVLink connection.

    Args:
        the_connection (object): The MAVLink connection object.
        mission_items (list[MissionItem]): A list of mission_item objects to be uploaded.

    Returns:
        None
    """
    ''' Uploads a list of mission items where the elements are defined as mission_item objects.
        params:
            the_connection = connection object
            mission_items  = List of mission_item objects

    '''
    n = len(mission_items)
    print('MAVLINK - Uploading Mission Items...')

    # Informing MavLink that we are sending n mission items.
    the_connection.mav.mission_count_send(the_connection.target_system, the_connection.target_component, n, 0)

    ack(the_connection, "MISSION_REQUEST")

    for waypoint in mission_items:
        print(f'\t MAVLINK - Sending mission item ({mission_items.index(waypoint)+1}/{n})')
        the_connection.mav.mission_item_int_send(the_connection.target_system,
                                                 the_connection.target_component,
                                                 waypoint.seq,
                                                 waypoint.frame,
                                                 waypoint.command,
                                                 waypoint.current,
                                                 waypoint.autocontinue,
                                                 waypoint.param1,
                                                 waypoint.param2,
                                                 waypoint.param3,
                                                 waypoint.param4,
                                                 waypoint.latitude,
                                                 waypoint.longitude,
                                                 waypoint.altitude,
                                                 waypoint.mission_type)

    if waypoint != mission_items[n-1]:
        ack(the_connection, "MISSION_REQUEST")

    ack(the_connection, "MISSION_ACK")


def start_mission(the_connection):
    """
    Starts a MAVLink mission.

    This function sends a MAVLink command to start a mission on the connected vehicle.

    Args:
        the_connection (object): The MAVLink connection object.

    Returns:
        None
    """

    print('MAVLINK - Starting Mission')
    the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,
                                    mavutil.mavlink.MAV_CMD_MISSION_START, 0, 0, 0, 0, 0, 0, 0, 0)
    ack(the_connection, "COMMAND_ACK")


def clear_mission(the_connection):
    """
    Deletes all mission items at once.

    Args:
        the_connection (object): The MAVLink connection object used to communicate with the vehicle.

    Returns:
        None
    """

    print('MAVLINK - Deleting ALL mission items')
    the_connection.waypoint_clear_all_send()
    ack(the_connection, "MISSION_ACK")


def print_mission_items(the_connection):
    """
    Prints the mission items from the MAVLink connection.

    This function requests the list of mission items from the MAVLink connection,
    retrieves each mission item, and prints its sequence number, latitude, and longitude.

    Args:
        the_connection (object): The MAVLink connection object.

    Returns:
        list: A list of mission items.
    """

    print('MAVLINK - Printing mission items:')
    the_connection.waypoint_request_list_send()  # Request the list of mission items
    msg = the_connection.recv_match(type=['MISSION_COUNT'], blocking=True, timeout=1)
    waypoint_count = msg.count if msg else 0
    print(f'\t MAVLINK - Counted {waypoint_count} waypoints')

    mission_items = []
    sequence = []
    for i in range(waypoint_count):
        the_connection.waypoint_request_send(i)
        msg = the_connection.recv_match(type=['MISSION_ITEM'], blocking=True, timeout=1)
        if msg:
            mission_items.append(msg)
            sequence.append(msg.seq)
            print(f'\t MAVLINK - Mission item {msg.seq}: [Latitude:{round((msg.x), 10)}, Longitude:{round((msg.y), 10)}] (Values are 10^7)')

    for i, waypoint_number in enumerate(sequence):
        if (waypoint_number != sequence[-1]) and (waypoint_number > sequence[i + 1]):
            MAV_RESULT = 4  # If the sequence is out of order, give an error
            break
    else:
        MAV_RESULT = 0

    return mission_items


def wait_for_mission_completion(the_connection, n, mission_timeout=999999999999):
    """
    Waits for the mission to complete by monitoring mission item reached messages.

    Args:
        the_connection (object): The MAVLink connection object.
        n (int): The number of mission items.
        mission_timeout (int, optional): The timeout period in seconds before the mission is considered failed. Default is a very large number.

    Returns:
        bool: True if the mission is completed within the timeout period, False otherwise.
    """
    end_time = time.time() + mission_timeout

    while time.time() < end_time:
        msg = the_connection.recv_match(type=['MISSION_ITEM_REACHED'], blocking=True, timeout=2)
        if msg:
            if msg.seq == (n - 1):
                print('MAVLINK - Mission Completed')
                return True

    print('MAVLINK - ERROR - Mission NOT COMPLETED within specified timeframe!')
    return False


def log_until_completion(the_connection, identifier, n, logpath, mission_timeout=999999999999):
    """
    Logs the GPS coordinates of the vehicle while undergoing a mission.

    This function continuously logs the GPS coordinates of the vehicle at regular intervals
    while checking if the mission is complete. It stops logging once the mission is completed
    or the timeout period is reached.

    Args:
        the_connection (object): The MAVLink connection object.
        identifier (str): An identifier for the log file.
        n (int): The number of mission items.
        logpath (str): The path to the log file.
        mission_timeout (int, optional): The timeout period in seconds before the mission is considered failed. Default is a very large number.

    Returns:
        bool: True if the mission is completed within the timeout period, False otherwise.
    """

    logging_interval = 3  # Logging every 3 seconds
    next_logging_time = time.time() + logging_interval

    end_time = time.time() + mission_timeout

    while time.time() < end_time:
        msg = the_connection.recv_match(type=['MISSION_ITEM_REACHED'], blocking=True, timeout=1)

        if time.time() >= next_logging_time:
            next_logging_time = time.time() + logging_interval
            with open(f'{logpath}/mission_log_position_{identifier}.csv', 'a', newline='\n') as file:
                writer = csv.writer(file, delimiter=';')
                gps = the_connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
                if gps:
                    writer.writerow([strftime("%d/%m/%Y_%H:%M:%S", localtime()), gps.lat / 10**7, gps.lon / 10**7])

        if msg:
            print(f'{msg.seq} / {n-1}')  # Prints MISSION_ITEM_REACHED message (Used for Debugging)
            if msg.seq == (n - 1):
                print('MAVLINK - Mission Completed')
                return True

    print('MAVLINK - ERROR - Mission NOT COMPLETED within specified timeframe!')
    return False


def set_mission_current(the_connection, seq):
    """
    Sets the specified mission item as the current one.

    Args:
        the_connection (object): The connection object to the MAVLink device.
        seq (int): The sequence number of the desired current mission item.

    Returns:
        None
    """

    print(f'MAVLINK - Setting mission item {seq} as current item')
    the_connection.mav.command_long_send(the_connection.target_system,
                                         the_connection.target_component,
                                         mavutil.mavlink.MAV_CMD_DO_SET_MISSION_CURRENT,
                                         seq, 1, 0, 0, 0, 0, 0, 0)
    ack(the_connection, "COMMAND_ACK")

