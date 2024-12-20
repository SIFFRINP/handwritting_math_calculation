from configuration import *


def separate_instructions(insts: str) -> tuple: 
    """
    Separate the operators from numbers in the instruction string. 

    :param insts: the instruction string. 
    :return: two lists, one containing every numbers and another containing 
             every operators. 
    """
    insts_len = len(insts)
    operator_list = []
    number_list = []
    
    # Check if the instruction string is longer than 1 char. 
    if (not insts or insts_len <= 1):
        return [], []


    # Check if the first char is a - or a +. 
    start_index = 0
    first_is_negative = False

    if not (insts[0].isdigit()):
        if insts[0] == "-": 
            first_is_negative = True
            start_index = 1
        
        elif (insts[0] == "+"): 
            start_index = 1

        else:
            return [], []


    # Parsing.
    i = start_index
    while (i < insts_len):
        
        # If the char is a digit, then read from this char to the next non-digit 
        # char, convert it to int and continue. 
        if (insts[i].isdigit()): 
            
            number = ""
            j = i

            while (j < insts_len): 
                if (insts[j].isdigit()): 
                    number += insts[j]
                    j += 1
                    continue
                
                break
            
            number_list.append(int(number) * (-1 if (i <= 1 and first_is_negative) else 1))
            i += j - i - 1

        # Check if the char is in the available operator charset and add it to 
        # the operator list. 
        elif (insts[i] in AVAILABLE_OPERATOR): 
            operator_list.append(insts[i])

        # If the end expression char is detected, stop the parsing. 
        elif (insts[i] == END_EXPRESSION_CHAR): 
            i = insts_len

        else:
            print("~[ERROR] Forbidden instruction.")
            return [], []

        i += 1

    return number_list, operator_list


def validate_array(number_list: list, operator_list: list) -> bool: 
    """
    Check if both lists as the correct amount of element from one another. 

    :param number_list: list of all numbers in the expression. 
    :param operator_list: list of all operators in the expression. 
    :return: 1 if there is the right amount of element, 0 otherwise. 
    """

    return len(number_list) - 1 == len(operator_list)


def perform_calc(number_list: list, operator_list: list) -> str: 
    """
    Calculate a parsed expression. 

    :param number_list: list of all numbers in the expression. 
    :param operator_list: list of all operators in the expression. 
    :return: the result of the whole expression. 
    """

    # Check if number_list is empty. 
    if (not number_list):
        return ""
    
    elif (not validate_array(number_list, operator_list)): 
        return ""

    # Return the only number if there is no operation to do. 
    elif (not operator_list): 
        return number_list[0]
    

    # Execute the priority calculation before hand. 
    perform_priority_calc(number_list, operator_list)


    # Execute secondary operator. 
    result = number_list[0]
    
    i = 0
    len_operator_list = len(operator_list)
    while (i < len_operator_list):
        result = operator_calc(result, number_list[i + 1], operator_list[i])
        i += 1

    return f"{result}"


def perform_priority_calc(number_list: list, operator_list: list): 
    """
    Perform every priority calculation (multiply and divide) before performing 
    the complete calculation. 

    :param number_list: list of all numbers in the expression. 
    :param operator_list: list of all operators in the expression. 
    """
    i = 0
    while (i < len(operator_list)): 

        # If the operator is not a priority operator, skip it. 
        if (operator_list[i] not in PRIORITY_OPERATOR):
            i += 1
            continue

        # Perform the calculation and modify the lists accordingly. 
        operator = operator_list.pop(i)
        nb = number_list.pop(i + 1)

        match (operator):
            case "×": 
                number_list[i] *= nb
            case "÷": 
                number_list[i] /= nb
    
    return


def operator_calc(nb1: int, nb2: int, op: str) -> int: 
    """
    Calculate an operation between two numbers and return the result.  

    :param nb1: first number. 
    :param nb2: second number. 
    :param op: the operation to do. 
    :return: the result of the operation. 
    """

    match op: 
        case "+":
            return nb1 + nb2 

        case "-": 
            return nb1 - nb2 

        case _: 
            return nb1
    
    return


def instruction_format(instruction: str) -> str:
    if (not instruction): 
        return ""

    if (instruction[-1] == "="): 
        return instruction

    return instruction + "=" 


if __name__ == "__main__": 
    print("\x1b[33m~[WARNING] This script is not meant to be executed.\x1b[0m"); 
