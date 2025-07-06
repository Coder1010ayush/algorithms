import os
import sys
import pathlib
import pygame
from game import __init_setup , gameLoop

# calling __init_setup
screen , clock , running_status = __init_setup(800 , 1000 , "Snake Game")
if __name__ == "__main__":
    gameLoop(screen , clock , running_status)
