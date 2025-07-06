# =============================== utf-8 encoding ===================================
import os
import sys
import pathlib
import pygame

# initialize the pygame instance and setup screen window for game
def __init_setup(width : int , height : int , caption : str):
    pygame.init()
    screen = pygame.display.set_mode(( height , width )) # dimension of game window
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()
    running_status = True # represents that game window is opened
    return screen , clock , running_status
    
def gameLoop(screen , clock , running_status):
    
    # game will be running until running_status is True
    while running_status:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_status = False

        # make game screen 
        screen.fill("black")
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
