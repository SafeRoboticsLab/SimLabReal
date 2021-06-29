# Copyright (c) 2021, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )
#          Vicenc Rubies-Royo  ( vrubies@berkeley.edu )

from gym.envs.registration import register


register(
    id="dubins_car-v1",
    entry_point="gym_reachability.gym_reachability.envs:DubinsCarOneEnv"
)

register(
    id="dubins_car_cont-v0",
    entry_point="gym_reachability.gym_reachability.envs:DubinsCarOneContEnv"
)

register(
    id="zermelo_cont-v0",
    entry_point="gym_reachability.gym_reachability.envs:ZermeloContEnv"
)

register(
    id="zermelo_show-v0",
    entry_point="gym_reachability.gym_reachability.envs:ZermeloShowEnv"
)