import time

import cv2
import matplotlib.pyplot as plt

from mean_shift_tracker import MSParams, MeanShiftTracker
from sequence_utils import VOTSequence

# set the path to directory where you have the sequences
dataset_path = 'vot2014'

# visualization and setup parameters
win_name = 'Tracking window'
reinitialize = True
show_gt = True
video_delay = 15
font = cv2.FONT_HERSHEY_PLAIN


def run_tracker(sequence_name, parameters=MSParams(), show_visualization=True):
    # create sequence object
    sequence = VOTSequence(dataset_path, sequence_name)
    init_frame = 0
    n_failures = 0
    tracker = MeanShiftTracker(parameters)
    time_all = 0
    # initialize visualization window
    sequence.initialize_window(win_name)
    # tracking loop - goes over all frames in the video sequence
    frame_idx = 0
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        # initialize or track
        if frame_idx == init_frame:
            # initialize tracker (at the beginning of the sequence or after tracking failure)
            t_ = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
            time_all += time.time() - t_
            predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
        else:
            # track on current frame - predict bounding box
            t_ = time.time()
            predicted_bbox = tracker.track(img)
            time_all += time.time() - t_

        # calculate overlap (needed to determine failure of a tracker)
        gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
        o = sequence.overlap(predicted_bbox, gt_bb)

        if show_visualization:
            # draw ground-truth and predicted bounding boxes, frame numbers and show image
            if show_gt:
                sequence.draw_region(img, gt_bb, (0, 255, 0), 1)

            sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
            sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
            sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
            sequence.show_image(img, video_delay)

        if o > 0 or not reinitialize:
            # increase frame counter by 1
            frame_idx += 1
        else:
            # increase frame counter by 5 and set re-initialization to the next frame
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1

    return tracker, sequence, time_all, n_failures


def run_sequences():
    # choose the sequences you want to test
    sequences = ['basketball', 'bolt', 'hand2', 'polarbear', 'jogging', 'fernando']
    # sequences = ['bolt']
    for sequence_name in sequences:
        tracker, sequence, time_all, n_failures = run_tracker(sequence_name, show_visualization=True)
        print(f'{sequence_name} & {sequence.length()} & {n_failures} \\\\\n\\hline')


def create_plot(parameter_, failures_, fps_, parameter_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.plot(parameter_, failures_, marker='s', markerfacecolor='none')
    ax2.plot(parameter_, fps_, marker='s', markerfacecolor='none', color='green')
    ax1.set(xlabel=parameter_name, ylabel='#failures', title=f'#failures / {parameter_name}')
    ax2.set(xlabel=parameter_name, ylabel='FPS', title=f'FPS / {parameter_name}')
    ax1.grid()
    ax2.grid()
    
    plt.tight_layout()
    plt.savefig(f'plots/{parameter_name}_failures.png', bbox_inches='tight')
    plt.show()


def run_parameters_comparison(seq='bolt'):
    params = MSParams()
    bins = [4, 6, 8, 12, 16]
    failures = []
    fps = []
    for bin_ in bins:
        params.bins = bin_
        tracker, sequence, time_all, n_failures = run_tracker(seq, params, show_visualization=False)
        failures.append(n_failures)
        fps.append(sequence.length() / time_all)

    create_plot(bins, failures, fps, 'bins')

    params = MSParams()
    alphas = [0, 0.1, 0.2, 0.4, 0.5, 0.6]
    failures = []
    fps = []
    for alpha_ in alphas:
        params.alpha = alpha_
        tracker, sequence, time_all, n_failures = run_tracker(seq, params, show_visualization=False)
        failures.append(n_failures)
        fps.append(sequence.length() / time_all)

    create_plot(alphas, failures, fps, 'alpha')

    params = MSParams()
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    failures = []
    fps = []
    for sigma_ in sigmas:
        params.sigma = sigma_
        tracker, sequence, time_all, n_failures = run_tracker(seq, params, show_visualization=False)
        failures.append(n_failures)
        fps.append(sequence.length() / time_all)

    create_plot(sigmas, failures, fps, 'sigma')

    params = MSParams()
    epsilons = [0.5, 1.0, 1.5, 2.0, 2.5]
    failures = []
    fps = []
    for epsilon_ in epsilons:
        params.epsilon = epsilon_
        tracker, sequence, time_all, n_failures = run_tracker(seq, params, show_visualization=False)
        failures.append(n_failures)
        fps.append(sequence.length() / time_all)

    create_plot(epsilons, failures, fps, 'epsilon')


if __name__ == '__main__':
    # run_parameters_comparison('bolt')
    run_sequences()

    # print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
    # print('Tracker failed %d times' % n_failures)
    # print(f'{sequence_name} & {sequence.length()} & {n_failures} \\\\\n\\hline')
    # print(f'{tracker.parameters.bins} & {sequence.length() / time_all:.1f} & {n_failures} \\\\\n\\hline')
