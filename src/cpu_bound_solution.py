import numpy as np
import scipy
import cv2
import multiprocessing
import pathlib

def cpu_compute_task(image, channel=2):
    channel = image[:, :, channel]
    transformed = scipy.signal.convolve2d(
        channel, np.random.randint(-2, 2, 4).reshape(2, 2)
    )
    scipy.fft.ifftn(channel)
    
def load_images(scale=2):
    images = [p for p in pathlib.Path("./assets/").iterdir() if p.suffix == ".png"]
    return [cv2.imread(str(p)) for p in images] * scale

def answer_cpu_bound_problem():
    """
    there are a few solutions here - this is the `most flat` one

    I don't think it's the fastest (?)
    """
    images = load_images()
    channels = []
    for image in images:
        for channel in range(len(image.shape)):
            channels.append((image, channel))

    with multiprocessing.Pool(4) as pool:
        pool.starmap(cpu_compute_task, channels)


if __name__ == "__main__":
    answer_cpu_bound_problem()