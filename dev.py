import os
import pandas as pd
import glob
from tabulate import tabulate
from itertools import count


def print_list(my_list: list) -> None:
    """
    Print a list with serial numbers, indices, and values.

    Args:
        my_list (list): The list to be printed.

    Returns:
        None
        
    Example:
    ```
    # Usage example:
    my_list = [10, 20, 30, 40]
    print_list(my_list)
    ```
    """
    # Print the type of the input list.
    print(type(my_list))

    # Print a header with labels 'S.No', 'Index', and 'Value', separated by underscores.
    print("S.No: \tIndex: \tValue:\n" + f'_' * 30)

    # Iterate through the list with an index ('serial') and corresponding value ('index' and 'value').
    for serial, (index, value) in enumerate(zip(range(1, len(my_list) + 1), my_list), start=1):
        # Print the serial number, index, and value.
        print(f"{serial}: \t{index - 1}: \t{value}")

def print_list2(my_list: list) -> None:
    """
    Print a list with index and value.

    Args:
        my_list (list): The list to be printed.

    Returns:
        None
    """
    # Use the 'map' function to create a formatted string for each item in 'my_list'.
    formatted_strings = map('{}: {}'.format, count(), my_list)

    # Print the formatted strings, separated by newline characters.
    return print(*formatted_strings, sep='\n')

def print_df(my_data_frame: pd.DataFrame) -> None:
    """
    Print a pandas DataFrame using the 'tabulate' library in console.

    Args:
        my_data_frame (pd.DataFrame): The DataFrame to be printed.

    Returns:
        None
    """
    # Print the DataFrame with tabulate using 'psql' style for headers.
    print(tabulate(my_data_frame, headers='keys', tablefmt='psql'))

def print_dict(my_dict: dict) -> None:
    """
    Print a dictionary with serial numbers, keys, and values.

    Args:
        my_dict (dict): The dictionary to be printed.

    Returns:
        None
    """
    # Print the type of the input dictionary.
    print(type(my_dict))

    # Print a header with labels 'S.No', 'Key', and 'Value', separated by underscores.
    print("S.No: \tKey: \tValue:\n" + f'_' * 30)

    # Iterate through the dictionary with an index ('serial') and corresponding key and value.
    for serial, (key, value) in enumerate(my_dict.items(), start=1):
        # Print the serial number, key, and value.
        print(f"{serial}: \t{key}: \t{value}")


