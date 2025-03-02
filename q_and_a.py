#!/usr/bin/env python3
'''Main node for the Q&A system. This node is responsible for playing audio clips, logging responses, and handling joystick input.'''

import csv
import json
from datetime import datetime
import random
import socket
import os
import subprocess
import threading
import signal
from sensor_msgs.msg import Joy
import tkinter as tk
from PIL import Image, ImageTk
import queue
from std_srvs.srv import Empty, EmptyRequest

# Added python imports for de-ROSed version
import pygame
import time

# ROS dependencies commented out
# import rospy
# import roslib.packages
# from quori_osu.srv import GetQuestion, GetQuestionRequest, KeyID, KeyIDResponse

# Get the package's base directory (e.g., /opt/quori/src/quori_osu)
# package_base_path = roslib.packages.get_pkg_dir('quori_osu')

# Replacing ROS package path retrieval with a hardcoded or configurable path
# package_base_path = os.path.expanduser("~/quori_osu")

# Define paths
quori_supplimental_path = os.path.join('quori_osu_supplemental')
simple_audio_path = os.path.join(quori_supplimental_path, 'q_and_a_audiofiles/', 'simple/')
complex_audio_path = os.path.join(quori_supplimental_path, 'q_and_a_audiofiles/', 'complex/')

# Placeholder values
id_string = "default_id"
key_id_string = "7"
scale_type = "Likert"
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Paths for logs and questions
logging_location = os.path.join('logs')
questions_location = os.path.join(quori_supplimental_path, 'questions')
key_file_path = os.path.join(questions_location, f'key_{key_id_string}.json')
csv_file_path = os.path.join(logging_location, f'{id_string}_key{key_id_string}_log_{current_time}.csv')
masterlist_name = 'masterlist.json'
masterlist_file_path = os.path.join(questions_location, masterlist_name)

# Global variables (sorry needed for threading) for questions and answers
question_id_list = []
simple_question_list = []
simple_answer_list = []
simple_audio_list = []
complex_question_list = []
complex_answer_list = []
complex_audio_list = []
complexity_list = []
response_list = []

# Delay times in seconds
delay_times = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
# Indexes and counters
current_text_index = 0
current_audio_index = 0
current_simple_writing_index = 0
current_complex_writing_index = 0
current_simple_pub_index = 0
current_complex_pub_index = 0
next_button_count = 0
current_delay = 0
total_questions = 0
rating = -1
current_process = None
task_queue = queue.Queue()
introduction_played = False
gui_started = False
updated_id = False
all_questions_exhausted = False
audio_playing = False
on_default_face = True

# Initialize Pygame for audio playback
pygame.mixer.init()

def init():
    """Initialize the audio output and set the delay for the first question."""
    global current_delay

    # Set the audio output to "Headphones - Built In Audio"
    set_audio_output('alsa_output.pci-0000_00_1f.3.analog-stereo')

    # Set the delay for the first question
    current_delay = random.choice(delay_times)


def send_command(command, host="127.0.0.1", port=5000):
    """Sends a command to the GUI server."""

    # # Example usage: Sending commands from the main node
    # send_command("start_gui")
    # send_command("change_face:thinking_face")
    # send_command("stop_gui")

    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host, port))
        client.send(command.encode())
        client.close()
    except Exception as e:
        print(f"Failed to send command: {e}")


# def init_service_clients():
#     """Initialize the service clients for starting and stopping the GUI and switching faces."""
#     # rospy.wait_for_service('start_gui')
#     # rospy.wait_for_service('stop_gui')
#     # start_gui_service = rospy.ServiceProxy('start_gui', Empty)
#     # stop_gui_service = rospy.ServiceProxy('stop_gui', Empty)
    
#     # create dictionary of face services
#     faces = ['default_face', 'thinking_face', 'talking_face']
#     [rospy.wait_for_service(face) for face in faces]
#     face_service_dict = {face: rospy.ServiceProxy('/'+face, Empty) for face in faces}
#     return start_gui_service, stop_gui_service, face_service_dict

def set_audio_output(sink_name):
    """Set the default audio output sink using pactl."""
    try:
        subprocess.run(['pactl', 'set-default-sink', sink_name], check=True)
    except subprocess.CalledProcessError as e:
        # rospy.logerr(f"Failed to set audio output: {e}")
        print(f"Failed to set audio output: {e}")


def load_json_file(file_path, default_file_path=None):
    """Load JSON data from a file, falling back to a default file if the primary file is not found."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        # rospy.logwarn(f"File '{file_path}' doesn't exist, opening the default file.")
        print(f"File '{file_path}' doesn't exist.")
        if default_file_path:
            with open(default_file_path, 'r') as default_file:
                return json.load(default_file)
        else:
            raise FileNotFoundError(f"Neither '{file_path}' nor a default file is available.")


def filter_questions(master_data, key_data):
    """Filter questions from the master data based on the key data."""
    filtered_questions = []
    key_questions = key_data['questions']  # Get the list of question IDs from the key file

    for key_id in key_questions:
        for question in master_data['questions']:
            if question['id'] == key_id:  # Match the question ID with the key ID
                filtered_questions.append(question)
                break  # Stop searching after finding the match

    return filtered_questions


def initialize_questions_and_answers():
    """Initialize the questions and answers based on the key data."""
    global simple_question_list, simple_answer_list, simple_audio_list
    global complex_question_list, complex_answer_list, complex_audio_list
    global question_id_list, complexity_list, total_questions

    # Load the master list and key data
    master_data = load_json_file(masterlist_file_path)
    key_data = load_json_file(key_file_path)

    # rospy.loginfo(f"Key Data: {key_data}")
    print(f"Key Data: {key_data}")

    # Filter the questions based on the key data
    filtered_questions = filter_questions(master_data, key_data)

    # Log the filtered questions for debugging
    # rospy.loginfo(f"Filtered Questions: {filtered_questions}")
    print(f"Filtered Questions: {filtered_questions}")

    # Initialize the lists
    simple_question_list = [q['question'] for q in filtered_questions if q['type'] == 'simple']
    simple_answer_list = [q['answer'] for q in filtered_questions if q['type'] == 'simple']
    simple_audio_list = [q['audio_file'] for q in filtered_questions if q['type'] == 'simple']

    complex_question_list = [q['question'] for q in filtered_questions if q['type'] == 'complex']
    complex_answer_list = [q['answer'] for q in filtered_questions if q['type'] == 'complex']
    complex_audio_list = [q['audio_file'] for q in filtered_questions if q['type'] == 'complex']

    question_id_list = [q['id'] for q in filtered_questions]

    complexity_list = [q['type'] for q in filtered_questions]

    total_questions = len(simple_question_list) + len(complex_question_list)

    # rospy.loginfo(f"Total questions: {total_questions}")
    # rospy.loginfo(f"Simple Questions: {len(simple_question_list)}, Complex Questions: {len(complex_question_list)}")
    print(f"Total questions: {total_questions}")


# Functions


def update_csv_file_path():
    """Update the CSV file path based on the user ID and key ID."""
    global csv_file_path
    csv_file_path = os.path.join(logging_location, f'{id_string}_key{key_id_string}_log_{current_time}.csv')
    # rospy.loginfo(f"Updated CSV file path to: {csv_file_path}")
    print(f"Updated CSV file path to: {csv_file_path}")


def write_to_file():
    """Append the original question ID, current question, answer, delay, rating, and timestamp to a CSV file."""
    global current_complex_writing_index, current_simple_writing_index, current_audio_index, rating
    # rospy.loginfo("Writing data to CSV file. Current Text Index: %d vs Response List %d", current_text_index, len(response_list))
    print(f"Writing data to CSV file. Current Text Index: {current_text_index} vs Response List {len(response_list)}")

    log_index = len(response_list) - 1 
    original_question_id = question_id_list[log_index]  # Ensure correct indexing
    rating_index = response_list[-1] if response_list else None

    # Determine if the question is simple or complex based on the complexity list
    if complexity_list[log_index] == 'complex':
        index = current_complex_writing_index 
        # rospy.loginfo(f"Writing Index: {index}")
        print(f"Writing Index: {index}")
        current_question = complex_question_list[index]
        current_answer = complex_answer_list[index]
        current_audio_file = complex_audio_list[index]
        current_complexity = 'Complex'
        current_complex_writing_index += 1
        # rospy.loginfo(f"Complex Index: {current_complex_writing_index}")
        print(f"Complex Index: {current_complex_writing_index}")
        # current_audio_index += 1
    else:
        index = current_simple_writing_index
        # rospy.loginfo(f"Writing Index: {index}")
        print(f"Writing Index: {index}")
        current_question = simple_question_list[index]
        current_answer = simple_answer_list[index]
        current_audio_file = simple_audio_list[index]
        current_complexity = 'Simple'
        current_simple_writing_index += 1
        # rospy.loginfo(f"Simple Index: {current_simple_writing_index}")
        print(f"Simple Index: {current_simple_writing_index}")
        # current_audio_index += 1
    
    if rating_index is not None:
        if scale_type == "Triad":
            # rospy.loginfo(f"Triad Writing")
            print(f"Triad Writing")
            if rating_index == 0:
                rating = "Too Slow"
            elif rating_index == 1:
                rating = "Somewhat Slow"
            elif rating_index == 2:
                rating = "Not Slow"
        else:
            # rospy.loginfo(f"Likert Writing")
            print(f"Likert Writing")
            if rating_index == 0:
                rating = "Strongly Disagree"
            elif rating_index == 1:
                rating = "Disagree"
            elif rating_index == 2:
                rating = "Neutral"
            elif rating_index == 3:
                rating = "Agree"
            elif rating_index == 4:
                rating = "Strongly Agree"

    # Get the current system time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Check if the CSV file already exists and write the header if it doesn't
    file_exists = os.path.isfile(csv_file_path)
    
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:  # If the file doesn't exist, write the header
            writer.writerow(['Question ID', 'Question', 'Answer', 'Complexity', 'Delay', 'Rating', 'Rating Index', 'Scale Type','Timestamp', 'Audio File', 'Masterlist'])
        
        # Append the actual data, including the original question ID
        writer.writerow([original_question_id, current_question, current_answer, current_complexity, current_delay, rating, rating_index, scale_type, current_time, current_audio_file, masterlist_name])
    
    print(f"Data logged: Question ID: {original_question_id}, Question: {current_question}, Answer: {current_answer}, Complexity: {current_complexity}, "
                  f"Delay: {current_delay}, Rating: {rating}, Rating Index: {rating_index}, Scale Type: {scale_type} Time: {current_time}, File: {current_audio_file}, Masterlist: {masterlist_name}")
    # rospy.loginfo(f"Data logged: Question ID: {original_question_id}, Question: {current_question}, Answer: {current_answer}, Complexity: {current_complexity}, "
    #               f"Delay: {current_delay}, Rating: {rating}, Rating Index: {rating_index}, Scale Type: {scale_type} Time: {current_time}, File: {current_audio_file}, Masterlist: {masterlist_name}")
    
    # Check if we have exhausted all questions
    # It needs to be here because its the last thing that happens in the order of functions
    if all_questions_exhausted:
        print("All out of questions nothing written to file.")
        # rospy.loginfo("All out of questions nothing written to file.")   


def handle_key_service(req):
    """Handle the KeyID service request."""
    global key_id_string, key_file_path, key_data, id_string, updated_id, scale_type
    # rospy.loginfo(f"Received User ID: {req.user_id}, Key ID: {req.key_id}, Scale Type: {req.scale_type}")
    print(f"Received User ID: {req.user_id}, Key ID: {req.key_id}, Scale Type: {req.scale_type}")

    key_id_string = req.key_id
    id_string = req.user_id
    scale_type = req.scale_type  # If needed, handle `scale_type` for further logic
    
    new_key_file_path = os.path.join(questions_location, f'key_{key_id_string}.json')
    
    # Load the new key file
    try:
        key_data = load_json_file(new_key_file_path, key_file_path)
        updated_id = True
        key_file_path = new_key_file_path
        initialize_questions_and_answers()
        
        # Log initialized data
        # rospy.loginfo(f"Simple Questions Initialized: {simple_question_list}")
        # rospy.loginfo(f"Complex Questions Initialized: {complex_question_list}")
        print(f"Simple Questions Initialized: {simple_question_list}")
        print(f"Complex Questions Initialized: {complex_question_list}")

        # Update CSV file path to include the key_id
        update_csv_file_path()

        # return KeyIDResponse(success=True)
        return True    
    
    except Exception as e:
        # rospy.logerr(f"Error processing key file: {str(e)}")
        print(f"Error processing key file: {str(e)}")

        # return KeyIDResponse(success=False) 
        return False


def swap_faces(face_service):
    """Swap faces using the face service."""
    try:
        face_service(EmptyRequest())
        # rospy.loginfo(f"Face Swapped Successfully")
        print(f"Face Swapped Successfully")
    except Exception as e:
        # rospy.logerr(f"Failed to call face swap service: {e}")
        print(f"Failed to call face swap service: {e}")


def play_audio(file_path):
    """Original audio playing function."""
    global current_process
    if current_process is None or current_process.poll() is not None:
        current_process = subprocess.Popen(["mpg123", file_path])
        while current_process is not None and current_process.poll() is None:
            pass


def stop_audio():
    """Original stop audio function."""
    global current_process
    if current_process:
        current_process.terminate()
        current_process = None


def start_gui():
    """Sends a service call to start the GUI Node."""
    try:
        start_gui_service(EmptyRequest())
        # rospy.loginfo("Start GUI service called successfully.")
    except Exception as e:
        # rospy.logerr(f"Failed to call start_gui service: {e}")
        print(f"Failed to call start_gui service: {e}")


def stop_gui():
    """Sends a service call to stop the GUI Node."""
    try:
        stop_gui_service(EmptyRequest())
        # rospy.loginfo("Stop GUI service called successfully.")
        print("Stop GUI service called successfully.")
    except Exception as e:
        # rospy.loginfo("GUI is already stopped.")
        print(f"Failed to call stop_gui service: {e}")


def introduction():
    """Function to play the introduction audio. Currently unused."""
    global introduction_file
    # rospy.loginfo("Playing introduction.")
    print("Playing introduction.")
    play_audio(introduction_file)


def play_with_delay(file_path, delay):
    """Function to play audio after a delay using threading."""
    def delayed_play():
        global on_default_face
        if not on_default_face:
            return
        on_default_face = False
        if delay > 0:
            swap_faces(face_service_dict['thinking_face'])

        # rospy.loginfo(f"Waiting for {delay} seconds before playing.")
        print(f"Waiting for {delay} seconds before playing.")
        # rospy.sleep(delay)  # Sleep without blocking the entire program
        time.sleep(delay)

        # rospy.loginfo(f"Now playing: {file_path}")
        
        swap_faces(face_service_dict['talking_face'])
        play_audio(file_path)
        swap_faces(face_service_dict['default_face'])
        on_default_face = True

    threading.Thread(target=delayed_play).start()


def play_next_audio_clip():
    """Function to play the next audio clip based on the current audio index."""
    global current_audio_index, all_questions_exhausted, current_delay

    if not all_questions_exhausted:  
        if complexity_list[current_audio_index] == 'complex':
            folder_path = os.path.expanduser(complex_audio_path)
            file_name = complex_audio_list[current_complex_writing_index]
        else:
            folder_path = os.path.expanduser(simple_audio_path)
            file_name = simple_audio_list[current_simple_writing_index]

        file_path = os.path.join(folder_path, file_name)
        # rospy.loginfo(f"Playing audio clip: {file_path}")
        print(f"Playing audio clip: {file_path}")
        play_with_delay(file_path, current_delay)

    else:
        # rospy.loginfo("All questions have been exhausted.")
        print("All questions have been exhausted.")
        all_questions_exhausted = True


def handle_question_request(req):
    """Handle the question request from the service. Returns the next question as long as the question list hasnt been exhausted."""
    global current_text_index, next_button_count, current_audio_index, all_questions_exhausted, current_complex_pub_index, current_simple_pub_index, response_list, current_delay, rating
    # rospy.loginfo(f"Question Requested.\n Next Button Count: {next_button_count} vs Text Count: {current_text_index} vs Audio Count: {current_audio_index} vs Total Questions: {total_questions} vs Response List: {len(response_list)}")
    # rospy.loginfo(f"Length of List: Simple: {len(simple_question_list)} and Complex: {len(complex_question_list)}")
    print(f"Question Requested.\n Next Button Count: {next_button_count} vs Text Count: {current_text_index} vs Audio Count: {current_audio_index} vs Total Questions: {total_questions} vs Response List: {len(response_list)}")    
    print(f"Length of List: Simple: {len(simple_question_list)} and Complex: {len(complex_question_list)}")

    # Check if we have exhausted all questions
    if all_questions_exhausted:
        # rospy.loginfo("All out of questions.")
        print("All out of questions.")
        return "All Out of Questions"

    response = req.rating
    rating = response

    # If the response is not -1, it means we received a rating
    # -1 is used as a placeholder for the first question
    if response != -1:
        response_list.append(response)

        # You can add debugging info if needed to check the received values
        # rospy.loginfo(f"Received index: {response}")
        print(f"Received index: {response}")

        # Check if we have exhausted all questions
        if len(response_list) <= len(simple_question_list) + len(complex_question_list):
            write_to_file()
        else:
            print("All questions have been answered. No more logging.")
            # rospy.loginfo("All questions have been answered. No more logging.")
    else:
        # rospy.loginfo("First question requested received.")
        print("First question requested received.")

    # Update the current delay for the next question
    current_delay = delay_times[random.randint(0, len(delay_times) - 1)]
    
    # Check if we have exhausted all questions
    if current_text_index < total_questions and len(response_list) < total_questions:
        if complexity_list[current_text_index] == 'complex':
            # index = current_text_index - len(simple_question_list) - (len(complex_question_list) - len(simple_question_list))
            index = current_complex_pub_index
            question = complex_question_list[index]
            current_complex_pub_index += 1
        else:
            # index = current_text_index - len(complex_question_list) - (len(simple_question_list) - len(complex_question_list))
            index = current_simple_pub_index
            question = simple_question_list[index]
            current_simple_pub_index += 1
        # rospy.loginfo(f"Sending question at index {index}: {question}")
        print(f"Sending question at index {index}: {question}")

        current_audio_index = current_text_index
        current_text_index += 1

        # If this is the last question, mark all questions as exhausted
        if current_text_index > total_questions:
            all_questions_exhausted = True
            # rospy.loginfo("All questions sent. Setting all_questions_exhausted to True.")
            print("All questions sent. Setting all_questions_exhausted to True.")
    else:
        all_questions_exhausted = True
        # rospy.loginfo("All Out of Questions")
        print("All Out of Questions")
        question = "All Out of Questions"

    return question


def joy_callback(data):
    """Callback function for joystick input."""
    global introduction_played, gui_started

    # Start button for remote question updating
    if data.buttons == (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0):  
        # rospy.loginfo("Start button pressed.")
        print("Start button pressed.")

        # Uncomment if you want the robot to play the introduction audio before starting the GUI
        # # Play the introduction if it hasn't been played yet
        # if not introduction_played and not gui_started:
        #     task_queue.put(introduction)
        #     introduction_played = True
        # # Start the GUI if the introduction has been played
        # elif not gui_started and introduction_played:

        if not gui_started:
            #This was left in incase a remote start of the gui is desired
            start_gui()
            gui_started = True
        else:
            try:
                # rospy.wait_for_service('/remote_update')  # Ensure the service is available
                remote_update_service = rospy.ServiceProxy('/remote_update', GetQuestion)

                # Create the request with the appropriate fields
                req = GetQuestionRequest()
                req.rating = -2  # Set the rating to -2 to indicate a remote update

                # Call the service
                remote_update_service(req)
            except Exception as e:
                # rospy.logerr(f"Service call failed: {e}")
                print(f"Service call failed: {e}")

    # Select button for stopping audio
    elif data.buttons == (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0):  
        # rospy.loginfo("Select button pressed.")
        print("Select button pressed.")
        task_queue.put(stop_audio)

    # A button for next audio clip
    elif data.buttons == (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0):  
        # rospy.loginfo("A button pressed.")
        print("A button pressed.")
        if updated_id == True:
            task_queue.put(play_next_audio_clip)

    elif data.buttons == (0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0):  
        # rospy.loginfo("Start and Select buttons pressed.")
        print("Start and Select buttons pressed.")
        task_queue.put(stop_gui)
        signal_handler(None, None)  # Call the signal handler to shut down

        
def listener():
    """Listener function for joystick input. Its broken up this way so this can run in a separate thread."""
    rospy.Subscriber("/joy", Joy, joy_callback)
    rospy.spin()


def process_tasks():
    """"Process tasks in the task queue. Used for threading with tkinter."""
    try:
        while not task_queue.empty():
            task = task_queue.get_nowait()
            task()
    except queue.Empty:
        pass
    root.after(100, process_tasks)


def signal_handler(sig, frame):
    """Handle the shutdown signal."""
    rospy.signal_shutdown("Shutdown signal received.")
    root.quit()  # Stop the Tkinter main loop


if __name__ == '__main__':
    """Main function to initialize the ROS node and start the GUI."""
    
    # Initialize the ROS node
    # rospy.init_node('q_and_a', anonymous=True)
    # rospy.loginfo("Q&A node started.")

    print("Q&A node started.")
    init()

    # Initialize the service clients
    # start_gui_service, stop_gui_service, face_service_dict = init_service_clients()

    root = tk.Tk()
    root.withdraw()

    root.after(100, process_tasks)

    # Seperate thread for listening to the joy controller commands 
    listener_thread = threading.Thread(target=listener)
    listener_thread.start()

    rospy.Service('/get_question', GetQuestion, handle_question_request)
    rospy.Service('/key_id', KeyID, handle_key_service)

    signal.signal(signal.SIGINT, signal_handler)  # Catch the shutdown signal

    root.mainloop()
    listener_thread.join()  # Ensure the listener thread completes before exiting
    # rospy.loginfo("Q&A node stopped.")
