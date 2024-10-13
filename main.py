import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


def main():
    """
    The main entry point of the application.

    This function initializes the application, sets up necessary configurations,
    and starts the main logic of the program.

    :return: None
    """

    model = YOLO("yolo11n-seg.pt")  # segmentation model
    cap = cv2.VideoCapture("crowd.mp4")
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter("instance-segmentation-object-tracking.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    while True:
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        annotator = Annotator(im0, line_width=2)
        results = model.track(im0, persist=True, verbose=False)

        if results[0].boxes.id is not None and results[0].masks is not None:
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()
            summarys = results[0].summary()

            for mask, track_id, summary in zip(masks, track_ids, summarys):
                color = colors(int(track_id), True)
                txt_color = annotator.get_txt_color(color)
                annotator.seg_bbox(mask=mask, mask_color=color,
                                   label="Class: " + summary["name"] + ", ID: " + str(track_id) + ", Conf: " +
                                         str(summary["confidence"]), txt_color=txt_color)

        out.write(im0)
        cv2.imshow("instance-segmentation-object-tracking", im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()
