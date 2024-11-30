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
        # print(i, insts[i], insts[i].isdigit(), insts[i] in AVAILABLE_OPERATOR)

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

        else:
            print("~[ERROR] Forbidden instruction.")
            return [], []

        i += 1

    return number_list, operator_list