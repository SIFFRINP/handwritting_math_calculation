from functions import *


INSTRUCTION = "10+4+3="


if __name__ == "__main__":
    numbers, operators = separate_instructions(INSTRUCTION)

    print(INSTRUCTION)
    print(numbers)
    print(operators)
