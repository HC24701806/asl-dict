import os
import ffmpeg
from util import shift, resize_shift

shift('butterfly.mp4', 'new.mp4', 200, -100)