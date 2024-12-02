from functions import *


INSTRUCTION = "5+32="


if __name__ == "__main__":
    numbers, operators = separate_instructions(INSTRUCTION)
    result = exec_calculation(numbers, operators)

    print("BASE INSTRUCTION: ")
    print(f"\t~ inst: {INSTRUCTION}")

    print("\nPARSING RESULT: ")
    print(f"\t~ nb_parsing: {numbers}")
    print(f"\t~ op_parsing: {operators}")

    print("\nCALCULATION RESULT: ")
    print(f"\t= {result}")

    print("FINAL STRING RESULT: ")
    print(f"\t~ {INSTRUCTION}{result}")
