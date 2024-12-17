import os
import ffmpeg
import crop
import shift
import trim

crop.cropVid('butterfly.mp4','new.mp4',200,300,0,0)
shift.verticalShift('butterfly.mp4','new.mp4',200)