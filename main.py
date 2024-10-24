import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


def initialize_model():
    """
    Initialize the YOLO model for segmentation.

    :return: YOLO model
    """
    return YOLO("yolo11n-seg.pt")


def initialize_video_capture(video_path):
    """
    Initialize the video capture object.

    :param video_path: Path to the video file
    :return: VideoCapture object, frame width, frame height, and frames per second
    """
    cap = cv2.VideoCapture(video_path)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    return cap, w, h, fps


def initialize_video_writer(output_path, fps, frame_size):
    """
    Initialize the video writer object.

    :param output_path: Path to the output video file
    :param fps: Frames per second
    :param frame_size: Frame size (width, height)
    :return: VideoWriter object
    """
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, frame_size)


def process_frame(model, frame, annotator):
    """
    Process a single frame using the YOLO model.

    :param model: YOLO model
    :param frame: Input frame
    :param annotator: Annotator object
    :return: Annotated frame
    """
    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()
        summarys = results[0].summary()

        for mask, track_id, summary in zip(masks, track_ids, summarys):
            color = colors(int(track_id), True)
            txt_color = annotator.get_txt_color(color)
            annotator.seg_bbox(mask=mask, mask_color=color,
                               label="Class: " + summary["name"] + ", Conf: " + str(summary["confidence"]),
                               txt_color=txt_color)

    return frame


def main():
    """
    The main entry point of the application.

    This function initializes the application, sets up necessary configurations,
    and starts the main logic of the program.

    :return: None
    """
    model = initialize_model()
    cap, w, h, fps = initialize_video_capture("crowd.mp4")
    out = initialize_video_writer("instance-segmentation-object-tracking.avi", fps, (w, h))

    while True:
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        annotator = Annotator(im0, line_width=2)
        im0 = process_frame(model, im0, annotator)

        out.write(im0)
        cv2.imshow("instance-segmentation-object-tracking", im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
