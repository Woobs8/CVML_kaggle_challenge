#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

def step_decay(initial_lrate, drop, epochs_drop):
    def sched_decay_fnc(epochs):
        lrate = initial_lrate * math.pow(drop, math.floor((1+epochs)/epochs_drop))
        return lrate

    return sched_decay_fnc