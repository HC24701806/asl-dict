import os
import ffmpeg

# inputFile: Input File Path
# outputFile: Output File Path

# shift: horizontal shift (in pixels)
def horizontalShift(inputFile,outputFile, shift):
    if os.path.exists(outputFile):
        os.remove(outputFile)

    zero = 0
    stream = ffmpeg.input(inputFile)
    shifted_stream = stream.filter('crop', 
                                    w=f'iw-{shift}', 
                                    h=f'ih-{zero}', 
                                    x=shift, 
                                    y=zero)
    output = ffmpeg.output(shifted_stream, outputFile)
    ffmpeg.run(output)

# shift: vertical shift (in pixels)
def verticalShift(inputFile,outputFile, shift):
    if os.path.exists(outputFile):
        os.remove(outputFile)

    zero = 0
    stream = ffmpeg.input(inputFile)
    shifted_stream = stream.filter('crop', 
                                    w=f'iw-{zero}', 
                                    h=f'ih-{shift}', 
                                    x=zero, 
                                    y=shift)
    output = ffmpeg.output(shifted_stream, outputFile)
    ffmpeg.run(output)