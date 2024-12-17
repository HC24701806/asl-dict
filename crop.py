import os
import ffmpeg

# inputFile: Input File Path
# outputFile: Output File Path
# width/height: Width and Height of Output
# x,y: Coordinates of top left corner of crop

def cropVid(inputFile,outputFile, width, height, x, y):
    if os.path.exists(outputFile):
        os.remove(outputFile)

    stream = ffmpeg.input(inputFile)
    cropped_stream = stream.crop(x,y,width,height)
    out = ffmpeg.output(cropped_stream, outputFile)
    ffmpeg.run(out)