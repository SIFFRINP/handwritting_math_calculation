from functions import *


instructions = [
    "3+5*32+5+4/6=", 
    "8+4*10-3/2=",
    "7*6+12/4-5=",
    "9+3*8-4/2=",
    "15-7+9*2/3=",
    "5*5-3+12/6=",
    "10/2+7*3-8=",
    "6+5*9-15/3=",
    "4*4+3*7-2/1=",
    "12/4+8*5-7=",
    "20-3+6*2/4=",
    "14+8*3-10/5=",
    "6*7-3+8/2=",
    "11+4*9-6/3=",
    "9*9-5+12/4="
]

results = [
    168.66666666666666,
    46.5,
    40.0,
    31.0,
    14.0,
    24.0,
    18.0,
    46.0,
    35.0,
    36.0,
    20.0,
    36.0,
    43.0,
    45.0,
    79.0
]


if __name__ == "__main__":

    i = 0
    for inst, res in zip(instructions, results): 
        
        print(f"\n\n_ INSTRUCTION {i} ____________________________________________")
        print("BASE INSTRUCTION: ")
        print(f"\t~ inst: {inst}")

        numbers, operators = separate_instructions(inst)
        print("\nPARSING RESULT: ")
        print(f"\t~ nb_parsing: {numbers}")
        print(f"\t~ op_parsing: {operators}")

        result = perform_calc(numbers, operators)
        print("\nCALCULATION RESULT: ")
        print(f"\t= {result} ?= {res} | {"✅" if (res == result) else "❌"}")

        print("\nFINAL STRING RESULT: ")
        print(f"\t~ {inst}{result}")
        i += 1