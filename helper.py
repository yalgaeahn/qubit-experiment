import os
import time
import json
import numpy as np
from datetime import datetime
from copy import deepcopy
import math

from qpu_types.transmon  import TransmonQubit




# ---------------------------------------------------------------------------------------------------------------------------
# STORE QUBIT PARAMETERS
# ---------------------------------------------------------------------------------------------------------------------------

# Custom function to handle serialization of complex numbers and numpy arrays
def custom_serializer(obj):
    if isinstance(obj, np.ndarray):
        print(obj)
        return {'real': obj.real.round(6).tolist(), 'imag': obj.imag.round(6).tolist()} # obj.tolist()  # Convert NumPy arrays to lists
    # if isinstance(obj, complex):
    #     return {'real': obj.real, 'imag': obj.imag}  # Convert complex numbers to dictionary
    # raise TypeError(f'Type {type(obj)} not serializable')


def save_qubit_parameters(qubits, save_folder = "./qubit_parameters/",timestamp=True, filename = 'qubit_parameters'):
    # create filepath
    if timestamp:
        t = time.localtime()
        timestamp = time.strftime('%Y%m%d-%H%M', t)
        qb_pars_file = os.path.abspath(
            os.path.join(save_folder, f"{timestamp}_{filename}.json")
        )
        # print(qubit_parameters)
        # return qubit_parameters
    else:
        qb_pars_file = os.path.abspath(
        os.path.join(save_folder, f"{filename}.json")
    )
    temp_qubits = deepcopy(qubits)
    # for qubit, dct in temp_qubits.items():
    obj_dict = {qubit : {slot: getattr(dct.parameters, slot) for slot in dir(dct.parameters) if not slot.startswith(("_","replace","drive_frequency","readout_frequency","copy"))} 
                for qubit, dct in temp_qubits.items()}  # Convert the object to a dictionary
    
    # Serialize the dictionary to JSON
    json_str = json.dumps(obj_dict, default=custom_serializer, indent=4)

    # Write the JSON string to a file
    with open(qb_pars_file, 'w') as f:
        f.write(json_str)  

    print(f"Qubit parameters stored as JSON in {qb_pars_file}")
    


# ---------------------------------------------------------------------------------------------------------------------------
# LOAD QUBIT PARAMETERS
# ---------------------------------------------------------------------------------------------------------------------------

# Custom decoder to rebuild complex numbers and np.ndarrays from JSON
def custom_decoder(dct):
    for key, value in dct.items():
        if isinstance(value, dict) and key == 'samples':
            if isinstance(value, dict) and 'real' in value and 'imag' in value:
                weights = np.array(value['real']) + 1j*np.array(value['imag']) 
                dct[key] = weights  # Convert lists back to NumPy arrays
    return dct

def find_latest_json(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    timestamps = []
    for file in files:
        try:
            # Extract a timestamp assuming it's included in the filename
            timestamp_str = file.rstrip('.json').split('_', 1)[0]  # Assuming YYYYMMDDHHMMSS format
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d-%H%M%S')
            timestamps.append((timestamp, file))
        except ValueError:
            continue  # Skip files that do not match the timestamp format

    # Find the most recent file
    if timestamps:
        latest_file = max(timestamps, key=lambda x: x[0])[1]
        return os.path.join(folder_path, latest_file)
    return None



def load_qubit_parameters(filename = 'qubit_parameters', save_folder='./qubit_parameters'):
    if filename == 'latest':
        qb_pars_file = find_latest_json(save_folder)
    else:
        qb_pars_file = os.path.abspath(
            os.path.join(save_folder, f"{filename}.json")
        )
    
    with open(qb_pars_file) as f:
        file = json.load(f, object_hook=custom_decoder)
    qubits = {}
    for qubit in file:
        qubits[qubit] = TransmonQubit(uid=qubit,signals={"drive":f"/logical_signal_groups/{qubit}/drive",
                                            "drive_ef":f"/logical_signal_groups/{qubit}/drive_ef",
                                            "drive_cr":f"/logical_signal_groups/{qubit}/drive_cr",
                                            "measure":f"/logical_signal_groups/{qubit}/measure",
                                            "acquire":f"/logical_signal_groups/{qubit}/acquire"})
        qubits[qubit].parameters = file[qubit]

    return qubits



def adjust_amplitude_for_output_range(initial_output_dbm, initial_amplitude, new_output_dbm):
    """
    Adjusts the amplitude to keep the overall output power constant when changing the output range.
    
    Parameters:
        initial_output_dbm (float): The initial output range in dBm.
        initial_amplitude (float): The initial amplitude scaling factor.
        new_output_dbm (float): The new output range you want to set in dBm.
    
    Returns:
        float: The new amplitude scaling factor to maintain the same output power.
    """
    # Convert dBm to linear power (mW)
    initial_power_mw = initial_amplitude * (10 ** (initial_output_dbm / 10))
    new_output_mw = 10 ** (new_output_dbm / 10)
    
    # Calculate the new amplitude
    new_amplitude = initial_power_mw / new_output_mw
    
    return new_amplitude


def dummy(x):
    print(1)