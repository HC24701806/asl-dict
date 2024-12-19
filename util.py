import ffmpeg

def crop(inputFile, outputFile, width, height, x, y):
    ffmpeg.input(inputFile).crop(x, y, width, height).output(outputFile).run()

def shift(input, output, x_shift, y_shift):
    {
        ffmpeg.input(input)
        .filter('crop', out_w=f'iw-{abs(x_shift)}', out_h=f'ih-{abs(y_shift)}', x=abs(min(0, x_shift)), y=abs(min(0, y_shift)))
        .filter('pad', width=f'iw+{abs(x_shift)}', height=f'ih+{abs(y_shift)}', x=max(0, x_shift), y=max(0, y_shift))
        .output(output).run()
    }

def resize_shift(input, output, x_shift, y_shift):
    {
        ffmpeg.input(input)
        .filter('crop', out_w=f'iw-{abs(x_shift)}', out_h=f'ih-{abs(y_shift)}', x=abs(min(0, x_shift)), y=abs(min(0, y_shift)))
        .output(output).run()
    }