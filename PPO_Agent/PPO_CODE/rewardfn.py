import numpy as np
np.set_printoptions(threshold=np.inf)

"""
minihack world info

char
32 = space
35 = # = corridor
43 = + = door
45 = - = opened door
47 = / = wand
60 = < = stair up (goal)
62 = > = stair down (spawn)
64 = @ = player
72 = H = monster (minotaur in the quest hard)
91 = [ = armor
125 = } = lava pool
61 = = = ring
46 = . = room floor after maze

print(curr_obs[0]) # glyphs
print(curr_obs[1]) # chars
print(curr_obs[2]) # colors
print(curr_obs[3]) # specials
print(curr_obs[4]) # blstats
print(curr_obs[5]) # message
print(curr_obs[6]) # inv_glyphs
print(curr_obs[7]) # inv_letters
print(curr_obs[8]) # inv_oclasses
print(curr_obs[9]) # 55-dimensional vector of ?
print(curr_obs[10]) # inv_strs
print(curr_obs[11]) # tty_chars
print(curr_obs[12]) # tty_colors
print(curr_obs[13]) # tty_cursor

The game is 21*79
long corridor (x=3 and x=19) y>=28
doors location (y=28 and y=38) x=11 
room location (9-13, 31-34)
exit location (11, 72)

ACTIONS:
    direction:
        0 = N
        1 = E
        2 = S
        3 = W
        4 = PICKUP

    not in use
    4 = NE
    5 = SE
    6 = SW
    7 = NW 
    8 = KICK
    10 = DOWN
    11 = OPEN
    12 = WIELD
    13 = ZAP
    TextCharacters:
    14 = PLUS = ord("+")  # Also SEESPELLS.
    15 = MINUS = ord("-")
    16 = SPACE = ord(" ")
    17 = APOS = ord("'")
    18 = QUOTE = ord('"')  # Also SEEAMULET.
    19 = NUM_0 = ord("0")
    20 = NUM_1 = ord("1")
    21 = NUM_2 = ord("2")
    22 = NUM_3 = ord("3")
    23 = NUM_4 = ord("4")
    24 = NUM_5 = ord("5")
    25 = NUM_6 = ord("6")
    26 = NUM_7 = ord("7")
    27 = NUM_8 = ord("8")
    28 = NUM_9 = ord("9")
    29 = DOLLAR = ord("$")  # Also SEEGOLD.



    Implement and take the text characters out

    Take diagonal moves out (NE,NW,SE,SW) there is no point moving diagonally in a maze

    Take kick out as our agent really like to kick walls in the maze (70% of time)

    Take pick up out as well, since we don;t know how to use the wand or any armor anyway
    All the reward for in the room are rebuild/comment out as we don't need it anymore (same reason as above)

    Take the encourage moving out
"""

"""
Reward function for solving the maze
"""
def reward_fn(env, prev_obs, action, curr_obs):
    reward = 0.0

    curr_x = curr_obs[4][1]
    curr_y = curr_obs[4][0]

    # reward for exploring
    # logic (less empty space = more explored map)
    if (prev_obs[1] == 32).sum() > (curr_obs[1] == 32).sum():
        print("Explore: gain reward")
        reward += 1.0

    # negative reward for forzen move
    if (prev_obs[1] == curr_obs[1]).all():
        print("Why are you not moving?")
        # v4 = 0.99 
        # v5 = 0.2 since there is 4 direction and action
        # v6 = 0.25
        # v7 = 0.99
        reward -= 0.99
        # if trying to push a door open
        if curr_x == 11 and (curr_y == 27 or curr_y == 37):
            if action == 1: # E
                print("trying to push a door open")
                reward += 1.5
        # in the room
        # if (curr_obs[4][1] >=9 and curr_obs[4][1] <=13) and (curr_obs[4][0] >=31 and curr_obs[4][0] <=34):
        #     reward += 1.5 # 1 to cancel out negative and 0.5 to encourage movement
        #     if(action == 4):
        #         print("Trying to pick up item.")
        #         reward += 0.5
    # else: # encourage moving
    #     print("Moving: gain tiny reward")
    #     reward += 0.02
    
    # punish for going into the long corridors
    # -5 was too heavy and affecting the rl agent too much, basically disencourage the agent to move
    # reduce to 0.5 -> 1.0
    # add conter reawrd for trying to move out
    if curr_y >= 28 and (curr_x == 3 or curr_x == 19):
        print("Why are you here?")
        reward -= 1.0
        if curr_y < prev_obs[4][0]:
            print("Good, you are leaving")
            reward += 1.0

    # reward for find door , no repeat reward
    if (curr_obs[1] == 43).sum() > (prev_obs[1] == 43).sum():
        print("Door located")
        reward += 10.0

    # reward for opening first door
    if curr_obs[1][11,28] == 45 and prev_obs[1][11,28] == 43:
        print("First door opened.")
        reward += 10.0

    # reward for opening second door    
    if curr_obs[1][11,38] == 45 and prev_obs[1][11,38] == 43:
        print("Second door opened.")
        reward += 10.0

    # negative reward for going back to the maze
    if curr_obs[1][11,28] == 45 and curr_y < 27:
        print("First door is opened. Why are you leaving?")
        reward -= 1.0

    # negative reward for going back to the maze
    if curr_obs[1][11,38] == 45 and curr_y < 27:
        print("Second door is opened. Why are you leaving?")
        reward -= 1.0

    # just give reward for surving after door is opened
    if curr_obs[1][11,28] == 45:
        reward += 2.0
        if curr_obs[1][11,38] == 45: # second door
            reward += 2.0

    # !!!!!
    # considering adding a shortest path algorithm to it
    # so if there exist a shortest path to the door, 
    # base on how close the agent is to the door give higher and higher reward
    # if the agent walk away then give punishment.

    return reward

"""
For the section below
Only start giving out reward or punishment after the door has opened
"""
def reward_fn_room(env, prev_obs, action, curr_obs):
    reward = 0.0

    prev_items_loc = []
    curr_items_loc = []
    if curr_obs[1][11,28] == 45:
        curr_item_count = 0
        prev_item_count = 0
        for x in range(9, 14, 1): #9-13
            for y in range(31, 35,1): #31-35
                if curr_obs[1][x,y] != 46 and curr_obs[1][x,y] != 64:
                    curr_item_count += 1
                    curr_items_loc.append([x,y])
                if prev_obs[1][x,y] != 46 and prev_obs[1][x,y] != 64:
                    prev_item_count += 1
                    prev_items_loc.append([x,y])

        # give small negative reward for having item on the floor
        reward -= curr_item_count*0.01

        # reward for find item
        if curr_item_count > prev_item_count:
            print("gain reward for find item")
            reward += 0.3 * (curr_item_count-prev_item_count)

        # if player was on a item and the action was not pick up give negative reward
        # don't ask me why, the order of x,y in this game is a mess
        player_last_pos = [prev_obs[4][1],prev_obs[4][0]]
        if curr_items_loc.count(player_last_pos):
            print("PUNISH for not pick up")
            reward -= 0.5
        
        # we can use message as well
        # if the previous state contains a message (appears when player stop on a item) not("it's a rock")
        # and if action != 9(pick up)
        # then punish player

    # room location = (31-34,9-13)
    # hummm idk maybe... idk
    #
    return reward


def reward_fn_final(env, prev_obs, action, curr_obs):
    reward = 0.0

    # negative reward for going back to the room ?

    # only start giving out reward or punishment after the second door has opened

    # if curr_obs[1][11,38] == 45:

    #     # basiclly telling the agent to go into lava
    #     if curr_obs[1][11,38] == 45:
    #         if curr_obs[4][0] > prev_obs[4][0]:
    #             reward += 50

    #     # reward for find exit
    #     if (curr_obs[1] == 62).sum() > (prev_obs[1] == 62).sum() and curr_obs[4][0] >= 50:
    #         print("Exit Found!")
    #         reward += 20.0

    #     # need to find how to use the wands one picked up 
    #     # funny when I choose Command.ZZAP or Comman.WIELD and I press
    #     # any other diection key it's non responsive
    #     # but 6 (SE) seems to be a option of something

    #     # reward fo reaching the exit    
    #     if curr_obs[4][0] == 72 and curr_obs[4][1] == 11:
    #         print("You are on the exit")
    #         reward += 30.0

    #     # the reward is given in the kill event as it's for terminal_requirement
    #     # there is a bug with this as well
    #     if (curr_obs[1] == 60).sum() > 0 and (curr_obs[1] == 62).sum() > 0 and (curr_obs[1] == 72).sum() == 0 and curr_obs[4][0] >= 50:
    #         print("monster killed")

    return reward