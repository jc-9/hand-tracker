"""On-device hand tracking for Luxonis OAK-D (Myriad X).

All ML runs on the device (depthai v2 classic API):
  ColorCamera preview -> ImageManip(128) -> palm-detection NN
      -> post-proc NN (on-device NMS) -> manager Script node
      -> ImageManip(rotated crop 224) -> hand-landmark NN -> manager Script
      -> marshal'd dict to host ("manager_out") + BGR frames ("cam_out").

Pipeline logic adapted from geaxgx/depthai_hand_tracker HandTrackerEdge
(MIT License, Copyright (c) 2021 geax), solo mode, no depth/gesture/world
landmarks. Host only renders + trails.

Needs oak_models/palm_detection_sh4.blob,
      oak_models/hand_landmark_lite_sh4.blob,
      oak_models/pd_postproc_top2_sh1.blob,
      oak_template_script_solo.py (device script template).
"""

import marshal
import os
import re
from string import Template

import depthai as dai
import numpy as np

import oak_mpu as mpu

HERE = os.path.dirname(os.path.abspath(__file__))
PALM_BLOB = os.path.join(HERE, "oak_models", "palm_detection_sh4.blob")
LANDMARK_BLOB = os.path.join(HERE, "oak_models", "hand_landmark_lite_sh4.blob")
POSTPROC_BLOB = os.path.join(HERE, "oak_models", "pd_postproc_top2_sh1.blob")
SCRIPT_TEMPLATE = os.path.join(HERE, "oak_template_script_solo.py")


def hands_from_result(res, frame_size, pad_h=0, pad_w=0):
    """Convert a manager-script result dict (or unmarshalled bytes) into a
    list of HandRegion with landmarks in source-frame pixels."""
    if isinstance(res, (bytes, bytearray)):
        res = marshal.loads(bytes(res))
    hands = []
    for i in range(len(res.get("lm_score", []))):
        hand = mpu.HandRegion()
        hand.rect_x_center_a = res["rect_center_x"][i] * frame_size
        hand.rect_y_center_a = res["rect_center_y"][i] * frame_size
        hand.rect_w_a = hand.rect_h_a = res["rect_size"][i] * frame_size
        hand.rotation = res["rotation"][i]
        hand.rect_points = mpu.rotated_rect_to_points(
            hand.rect_x_center_a, hand.rect_y_center_a,
            hand.rect_w_a, hand.rect_h_a, hand.rotation)
        hand.lm_score = res["lm_score"][i]
        hand.handedness = res["handedness"][i]
        hand.label = "right" if hand.handedness > 0.5 else "left"
        hand.norm_landmarks = np.array(res["rrn_lms"][i]).reshape(-1, 3)
        hand.landmarks = (np.array(res["sqn_lms"][i]) * frame_size
                          ).reshape(-1, 2).astype(np.int32)
        if pad_h > 0:
            hand.landmarks[:, 1] -= pad_h
            for p in hand.rect_points:
                p[1] -= pad_h
        if pad_w > 0:
            hand.landmarks[:, 0] -= pad_w
            for p in hand.rect_points:
                p[0] -= pad_w
        hands.append(hand)
    return hands


def hand_to_det(hand, frame_w, frame_h):
    """HandRegion -> (x1,y1,x2,y2,cx,cy,pts,handed) tuple used by app.py."""
    pts = [(int(x), int(y)) for x, y in hand.landmarks[:, :2]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx, cy = pts[9]  # palm center (landmark 9)
    return (max(0, min(xs)), max(0, min(ys)),
            min(frame_w - 1, max(xs)), min(frame_h - 1, max(ys)),
            cx, cy, pts, hand.label)


class OakHandTracker:
    """Solo edge hand tracker. Raises RuntimeError if no OAK device found."""

    def __init__(self, pd_score_thresh=0.5, lm_score_thresh=0.5,
                 internal_frame_height=640):
        for p in (PALM_BLOB, LANDMARK_BLOB, POSTPROC_BLOB, SCRIPT_TEMPLATE):
            if not os.path.exists(p):
                raise RuntimeError(f"missing OAK asset: {p}")
        self.pd_score_thresh = pd_score_thresh
        self.lm_score_thresh = lm_score_thresh
        try:
            self.device = dai.Device()
        except Exception as e:
            raise RuntimeError(
                "No OAK-D found. Plug it in (USB 03e7:2485) and retry. "
                f"Underlying error: {e}")
        print("[oak] device:", self.device.getDeviceName(),
              "MxId:", self.device.getMxId())
        self.resolution = (1920, 1080)
        width, self.scale_nd = mpu.find_isp_scale_params(
            internal_frame_height * self.resolution[0] / self.resolution[1],
            self.resolution, is_height=False)
        self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
        self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
        self.pad_h = (self.img_w - self.img_h) // 2
        self.pad_w = 0
        self.frame_size = self.img_w
        self.crop_w = 0
        print(f"[oak] frame {self.img_w}x{self.img_h} pad_h={self.pad_h}")
        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"[oak] pipeline started - USB: {str(usb_speed).split('.')[-1]}")
        self.q_video = self.device.getOutputQueue(
            name="cam_out", maxSize=1, blocking=False)
        self.q_manager_out = self.device.getOutputQueue(
            name="manager_out", maxSize=1, blocking=False)

    def build_manager_script(self):
        with open(SCRIPT_TEMPLATE) as f:
            template = Template(f.read())
        code = template.substitute(
            _TRACE1="#", _TRACE2="#",
            _pd_score_thresh=self.pd_score_thresh,
            _lm_score_thresh=self.lm_score_thresh,
            _pad_h=self.pad_h, _img_h=self.img_h, _img_w=self.img_w,
            _frame_size=self.frame_size, _crop_w=self.crop_w,
            _IF_XYZ='"""', _IF_USE_HANDEDNESS_AVERAGE="",
            _single_hand_tolerance_thresh=10,
            _IF_USE_SAME_IMAGE='"""', _IF_USE_WORLD_LANDMARKS='"""',
        )
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'\n\s*\n', '\n', code)
        return code

    def create_pipeline(self):
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)
        cam = pipeline.createColorCamera()
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setInterleaved(False)
        cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
        cam.setFps(30)
        cam.setVideoSize(self.img_w, self.img_h)
        cam.setPreviewSize(self.img_w, self.img_h)
        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam_out.input.setQueueSize(1)
        cam_out.input.setBlocking(False)
        cam.video.link(cam_out.input)

        manager_script = pipeline.create(dai.node.Script)
        manager_script.setScript(self.build_manager_script())

        pre_pd_manip = pipeline.create(dai.node.ImageManip)
        pre_pd_manip.setMaxOutputFrameSize(128 * 128 * 3)
        pre_pd_manip.setWaitForConfigInput(True)
        pre_pd_manip.inputImage.setQueueSize(1)
        pre_pd_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_pd_manip.inputImage)
        manager_script.outputs["pre_pd_manip_cfg"].link(pre_pd_manip.inputConfig)

        pd_nn = pipeline.create(dai.node.NeuralNetwork)
        pd_nn.setBlobPath(PALM_BLOB)
        pre_pd_manip.out.link(pd_nn.input)

        post_pd_nn = pipeline.create(dai.node.NeuralNetwork)
        post_pd_nn.setBlobPath(POSTPROC_BLOB)
        pd_nn.out.link(post_pd_nn.input)
        post_pd_nn.out.link(manager_script.inputs["from_post_pd_nn"])

        manager_out = pipeline.create(dai.node.XLinkOut)
        manager_out.setStreamName("manager_out")
        manager_script.outputs["host"].link(manager_out.input)

        pre_lm_manip = pipeline.create(dai.node.ImageManip)
        pre_lm_manip.setMaxOutputFrameSize(224 * 224 * 3)
        pre_lm_manip.setWaitForConfigInput(True)
        pre_lm_manip.inputImage.setQueueSize(1)
        pre_lm_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_lm_manip.inputImage)
        manager_script.outputs["pre_lm_manip_cfg"].link(pre_lm_manip.inputConfig)

        lm_nn = pipeline.create(dai.node.NeuralNetwork)
        lm_nn.setBlobPath(LANDMARK_BLOB)
        lm_nn.setNumInferenceThreads(1)  # solo mode
        pre_lm_manip.out.link(lm_nn.input)
        lm_nn.out.link(manager_script.inputs["from_lm_nn"])
        return pipeline

    def next_frame(self):
        """Returns (bgr_frame, [HandRegion]). Blocks for the next synced pair."""
        in_video = self.q_video.get()
        frame = in_video.getCvFrame()
        res = marshal.loads(self.q_manager_out.get().getData())
        hands = hands_from_result(res, self.frame_size, self.pad_h, self.pad_w)
        return frame, hands

    def close(self):
        try:
            self.device.close()
        except Exception:
            pass
