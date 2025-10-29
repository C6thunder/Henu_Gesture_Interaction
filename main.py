#!/usr/bin/env python
# -*- coding: utf-8 -*-
from hand_command import HandGestureRecognitionWithCommand

if __name__ == "__main__":
    app = HandGestureRecognitionWithCommand()
    app.keypoint_labels = ["Open", "Close", "Pointer", "OK", "Other1", "Other2"]
    app.run()
