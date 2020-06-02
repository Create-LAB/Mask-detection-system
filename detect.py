import argparse
import time
import threading
import face_recognition as fr
from sys import platform
from models import *
from project.datasets import *
from project.utils import *
from pydub import AudioSegment
from pydub.playback import play


def show(weights, im0):
    """
    Show image.
    """
    cv2.imshow(weights, im0)


def playsound():
    """
    play the audio
    """
    song = AudioSegment.from_wav('./voice/mask_on.wav')
    play(song)


def detect(
        cfg,
        data_cfg,
        weights,
        face_recognition,
        images='data/samples',  # input folder
        output='output',  # output folder
        fourcc='mp4v',
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
        save_txt=False,
        save_images=True,
        webcam=True):
    """
    Loading the trained model to detect whether a individual is wearing a mask.
    If not,a voice alert will be given.

    Parameters
    ----------
    cfg: str
        The path of the model cfg.
    data_cfg: str
        The path of the data cfg.
        eg. data/mask.data
    weights: str
        The path of the weights file.
    images: str,'data/samples'
        The path of the images for detect.
    output: str
        If you want to save the detected result, you should set it.
    fourcc: str
        Four character code,use in cv2.VideoWriter_fourcc
    img_size: int
        The size of the image loaded.
    conf_thres: int
    nms_thres: int
    save_txt: bool
        If ypu want to write bounding boxes and labels of detections into txt.
    save_images: bool
        If you want to save image.
    webcam: bool
        If you want to open the camera.

    """
    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder
    # ONNX:https://ptorch.com/docs/1/torch.onnx
    if ONNX_EXPORT:
        s = (320, 192)
        model = Darknet(cfg, s)
    else:
        model = Darknet(cfg, img_size)
    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(
            torch.load(
                weights,
                map_location=device)['model'])
    else:
        # darknet format
        _ = load_darknet_weights(model, weights)
    # Fuse Conv2d + BatchNorm2d layers
    model.fuse()
    # Eval mode
    model.to(device).eval()
    if ONNX_EXPORT:
        img = torch.zeros((1, 3, s[0], s[1]))
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return
    
    # init faces library
    known_face_encodings = []
    known_face_names = []
    print('INITIALIZING FACES LIBRARY...')
    faces_lib = os.listdir('data/faces_lib')
    for f in faces_lib:
        if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png'):
            current_image = fr.load_image_file(os.path.join('data/faces_lib', f))
            current_face_encoding = fr.face_encodings(current_image)[0]
            known_face_encodings.append(current_face_encoding)
            known_face_names.append(f.split('.')[0])
    print("{} known faces loaded.".format(len(faces_lib)))
    
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)
    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(classes))]
    res = ""
    times = 0
    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        t = time.time()
        # covert bgr to rgb
#         small_img = cv2.resize(im0, (0, 0), fx=0.25, fy=0.25)
#         fr_image = small_img[:, :, ::-1]
#         face_locations = fr.face_locations(fr_image)
#         print(face_locations)
        # face_encodings = fr.face_encodings(rgb_small_frame, face_locations)
        # print(face_locations)
        
        save_path = str(Path(output) / Path(path).name)
        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred, _ = model(img)
        det = non_max_suppression(pred, conf_thres, nms_thres)[0]
        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()
            # Print results to screen
            print('%gx%g ' % img.shape[2:], end='')  # print image size
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')
            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in det:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                # Add bbox to the image
                small_img = cv2.resize(im0, (0, 0), fx=0.25, fy=0.25)
                fr_image = small_img[:, :, ::-1]
                bbox = [int(x.item()//4) for x in xyxy]
                # face_locations = fr.face_locations(fr_image)
                print((bbox[0], bbox[3], bbox[2], bbox[1]))
                face_encodings = fr.face_encodings(fr_image, [(bbox[0], bbox[3], bbox[2], bbox[1])])
                # print(face_encoding)
                matches = fr.compare_faces(known_face_encodings, face_encodings[0])
                name = "Unknown Person"
                
                face_distances = fr.face_distance(known_face_encodings, face_encodings[0])
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                                
                label = '%s %s  %.2f' % (name, classes[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                
                res = classes[int(cls)]
        print('Done. (%.3fs)' % (time.time() - t))
        if webcam:
            show(weights, im0)
            if res == "no_mask" and (times % 50) == 0:
                threading.Thread(target=playsound).start()
        if save_images:  # Save image with detections
            if dataloader.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(
                        save_path, cv2.VideoWriter_fourcc(
                            *fourcc), fps, (width, height))
                vid_writer.write(im0)
        times += 1
    if save_images:
        print('Results saved to %s' % os.getcwd() + os.sep + output)
        if platform == 'darwin':  # macos
            os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg',
        type=str,
        default='cfg/yolov3-spp.cfg',
        help='cfg file path')
    parser.add_argument(
        '--data-cfg',
        type=str,
        default='data/coco.data',
        help='coco.data file path')
    parser.add_argument(
        '--weights',
        type=str,
        default='weights/yolov3-spp.weights',
        help='path to weights file')
    parser.add_argument(
        '--images',
        type=str,
        default='data/samples',
        help='path to images')
    parser.add_argument(
        '--img-size',
        type=int,
        default=416,
        help='inference size (pixels)')
    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.5,
        help='object confidence threshold')
    parser.add_argument(
        '--nms-thres',
        type=float,
        default=0.5,
        help='iou threshold for non-maximum suppression')
    parser.add_argument(
        '--fourcc',
        type=str,
        default='mp4v',
        help='specifies the fourcc code for output video encoding (make sure ffmpeg supports specified fourcc codec)')
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='specifies the output path for images and videos')
    parser.add_argument(
        '--face_recognition',
        type=bool,
        default=True,
        help='get names of faces in one image')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            images=opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            fourcc=opt.fourcc,
            output=opt.output,
            face_recognition=opt.face_recognition
        )
