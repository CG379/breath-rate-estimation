import os
import serial.tools.list_ports
import time
import csv
from datetime import datetime

ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()


baudrate_variable = 112500
portsList = []

print("---- PRINTING PORTS AVALIABLE ----")
for foundport in ports:
    portsList.append(str(foundport))
    print(str(foundport))

if len(portsList) == 0:
    print("no ports found")

print("----------------------------------")

val = input("Select COM Port: ")

use = None  # Set a default value

for x in range(len(portsList)):
    if portsList[x].startswith("COM" + str(val)):
        use = "COM" + str(val)
        print(f"Selected port: {use}")
        break  # Exit loop once found

if use is None:
    print("Error: Selected COM port not found.")
    exit()  # Stop execution if no valid COM port is found

serialInst.baudrate = 112500
serialInst.port = use
serialInst.open()

"""
WRITING THE CODE DOWN BELOW
"""

data_storage = []


def clear():
    '''
    Clears the terminal screen and scroll back to present
    the user with a nice clean, new screen. Useful for managing
    menu screens in terminal applications.
    '''
    os.system('cls||echo -e \\\\033c')

def export_data():
    clear()
    print(">>>>>>> List of Commands <<<<<<<")
    command = input("Export data (Y/N): ")

    if command.lower() == "y":
        clear()
        print("Writing CSV file......")
        with open(f"breathrate_data_{datetime.now()}.csv", mode="w", newline='') as file:
            writer = csv.writer(file)

            for item in data_storage:
                writer.writerow([item])  # wrap in list to make each a row
        
        clear()
        print("CSV file done.")

        time.sleep(2)

        activation_menu()
        

    elif command.lower() == "n":
        activation_menu()

def start_sensor(no_breaths: int):
    
    while no_breaths != 0:
        signal_go = "BG"
        serialInst.write(signal_go.encode('utf-8')) # this sends the signal that we are ready to sense
        serialInst.flush()

        for j in range(6): #EXCLUSIVE, this is the loading bar
            clear()
            #spacer = "-"
            bar = 'LOADING BREATH RATE SENSOR GET READY USER' + '-'*j
            print(bar)
            time.sleep(1)

        time_interval = 0 # store the time interval
        
        for i in range(4,0, -1):
            clear()
            print(">>> Breathe in <<<")
            print(f">>>    {i}    <<<")
            start_time = time.time() # we start a clock here, this is used for an interval calculation. Start as soon as it is displayed

            while serialInst.in_waiting and time_interval <= 1: # we need the interval to be one second, we display the value, then record then display again.
                sensor_data = serialInst.readline().decode('utf-8').strip()
                #print(f"Sensor Value: {sensor_data}")
                data_storage.append(sensor_data)

                end_time = time.time()
                time_interval = end_time - start_time # need to keep updated in the loop for an exit condition

            #time.sleep(1)

            time_interval = 0 # reset the time interval and get ready for another iteration
        
        data_storage.append(int(0000)) # signal that we have breathed in

        for k in range(6, 0, -1):
            clear()
            print("<<< Exhale >>>")
            print(f">>>  {k}  <<<")
            start_time = time.time() 

            while serialInst.in_waiting and time_interval <= 1:
                sensor_data = serialInst.readline().decode('utf-8').strip()
                #print(f"Sensor Value: {sensor_data}")
                data_storage.append(sensor_data)

                end_time = time.time()
                time_interval = end_time - start_time 

            time_interval = 0
        
        data_storage.append(int(0000))
        
        no_breaths -= 1
    
    clear()
    print("Breath recorded, now returning.....")
    time.sleep(1)

    activation_menu() # return to the activation menu
    
def time_selection():
    clear()
    print(">>>>>>> List of Commands <<<<<<<")
    print("Input time in seconds for how many breaths to record")
    print("----------------------------------")
    record_value = input("Input: ")

    start_sensor(record_value) # we type in how many breaths we want to do here, this then passes the parameter for how many loops

def activation_menu():
    while True:
        clear()
        print("------- BREATH RATE SENSOR ------")
        print(">>>>>>> List of Commands <<<<<<<")
        print("1. Information")
        print("2. Begin")
        print("3. Transcript")
        print("4. Exit")
        print("----------------------------------")

        command = input("Input: ")

        # serialInst.out_waiting() check the
        

        if command.lower() == "begin" or command.lower() == "b" or command.lower() == "2":
            time_selection()
        
        elif command.lower() == "transcript" or command.lower() == "t" or command.lower() == "3":
            export_data()
        
        elif command.lower() == "exit":
            print("exit has been issued... ")
            print("---- TERMINATING ----")
            serialInst.close()
            break

activation_menu()