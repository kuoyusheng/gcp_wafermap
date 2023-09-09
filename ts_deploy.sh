#!/bin/bash
torchserve --start \
           --model-store inference \
           --models ts-test=ts-test.mar