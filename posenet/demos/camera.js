/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as posenet from '@tensorflow-models/posenet';
import dat from 'dat.gui';
import Stats from 'stats.js';
import {
  drawKeypoints,
  drawSkeleton,
  drawBoundingBox,
} from './demo_util';


const videoWidth = document.body.clientWidth;
const videoHeight = document.body.clientHeight;
const stats = new Stats();

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

const guiState = {
  algorithm: 'multi-pose',
  input: {
    mobileNetArchitecture: isMobile() ? '0.50' : '0.75',
    outputStride: 16,
    imageScaleFactor: 0.5,
  },
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    showBoundingBox: false,
  },
  net: null,
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras, net) {
  guiState.net = net;

  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const gui = new dat.GUI({
    width: 300,
  });

  // The single-pose algorithm is faster and simpler but requires only one
  // person to be in the frame or results will be innaccurate. Multi-pose works
  // for more than 1 person
  const algorithmController =
    gui.add(guiState, 'algorithm', ['single-pose', 'multi-pose']);

  // The input parameters have the most effect on accuracy and speed of the
  // network
  let input = gui.addFolder('Input');
  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  const architectureController = input.add(
    guiState.input, 'mobileNetArchitecture',
    ['1.01', '1.00', '0.75', '0.50']);
  // Output stride:  Internally, this parameter affects the height and width of
  // the layers in the neural network. The lower the value of the output stride
  // the higher the accuracy but slower the speed, the higher the value the
  // faster the speed but lower the accuracy.
  input.add(guiState.input, 'outputStride', [8, 16, 32]);
  // Image scale factor: What to scale the image by before feeding it through
  // the network.
  input.add(guiState.input, 'imageScaleFactor').min(0.2).max(1.0);
  input.open();

  // Pose confidence: the overall confidence in the estimation of a person's
  // pose (i.e. a person detected in a frame)
  // Min part confidence: the confidence that a particular estimated keypoint
  // position is accurate (i.e. the elbow's position)
  let single = gui.addFolder('Single Pose Detection');
  single.add(guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0);
  single.add(guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0);

  let multi = gui.addFolder('Multi Pose Detection');
  multi.add(guiState.multiPoseDetection, 'maxPoseDetections')
    .min(1)
    .max(20)
    .step(1);
  multi.add(guiState.multiPoseDetection, 'minPoseConfidence', 0.0, 1.0);
  multi.add(guiState.multiPoseDetection, 'minPartConfidence', 0.0, 1.0);
  // nms Radius: controls the minimum distance between poses that are returned
  // defaults to 20, which is probably fine for most use cases
  multi.add(guiState.multiPoseDetection, 'nmsRadius').min(0.0).max(40.0);
  multi.open();

  let output = gui.addFolder('Output');
  output.add(guiState.output, 'showVideo');
  output.add(guiState.output, 'showSkeleton');
  output.add(guiState.output, 'showPoints');
  output.add(guiState.output, 'showBoundingBox');
  output.open();


  architectureController.onChange(function (architecture) {
    guiState.changeToArchitecture = architecture;
  });

  algorithmController.onChange(function (value) {
    switch (guiState.algorithm) {
      case 'single-pose':
        multi.close();
        single.open();
        break;
      case 'multi-pose':
        single.close();
        multi.open();
        break;
    }
  });

  gui.close();
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);
}

let holdCounter = 0;
let downCount = 0;

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');
  // since images are being fed from a webcam
  const flipHorizontal = true;

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  const boxSize = 200;
  const leftBoxColor = 'olive';
  const rightBoxColor = 'green';

  const drawStaticBoxBuilder = function (x, y, w, h) {
    return function (color, lineWidth = 2) {
      const storedColor = ctx.strokeStyle;
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
      ctx.strokeStyle = storedColor;
    };
  };

  const drawLeftBox = drawStaticBoxBuilder(videoWidth * 0.1, videoHeight * 0.05, boxSize, boxSize);
  const drawRightBox = drawStaticBoxBuilder(videoWidth * 0.9 - boxSize, videoHeight * 0.05, boxSize, boxSize);

  async function poseDetectionFrame() {
    if (guiState.changeToArchitecture) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();

      // Load the PoseNet model weights for either the 0.50, 0.75, 1.00, or 1.01
      // version
      guiState.net = await posenet.load(+guiState.changeToArchitecture);

      guiState.changeToArchitecture = null;
    }

    // Begin monitoring code for frames per second
    stats.begin();

    // Scale an image down to a certain factor. Too large of an image will slow
    // down the GPU
    const imageScaleFactor = guiState.input.imageScaleFactor;
    const outputStride = +guiState.input.outputStride;

    let poses = [];
    let minPoseConfidence;
    let minPartConfidence;
    switch (guiState.algorithm) {
      case 'single-pose':
        const pose = await guiState.net.estimateSinglePose(
          video, imageScaleFactor, flipHorizontal, outputStride);
        poses.push(pose);

        minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
        break;
      case 'multi-pose':
        poses = await guiState.net.estimateMultiplePoses(
          video, imageScaleFactor, flipHorizontal, outputStride,
          guiState.multiPoseDetection.maxPoseDetections,
          guiState.multiPoseDetection.minPartConfidence,
          guiState.multiPoseDetection.nmsRadius);

        minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
        break;
    }

    ctx.clearRect(0, 0, videoWidth, videoHeight);

    if (guiState.output.showVideo) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-videoWidth, 0);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      ctx.restore();
    }

    // For each pose (i.e. person) detected in an image, loop through the poses
    // and draw the resulting skeleton and keypoints if over certain confidence
    // scores
    drawLeftBox(leftBoxColor);
    drawRightBox(rightBoxColor);

    const checkRegionBuilder = function (x1, y1, x2, y2) {
      return function (x, y) {
        return x > x1 && x < x2 && y > y1 && y < y2;
      };
    };

    const checkIfLayOnGround = function (kpList) {
      const flatThreshold = 100;
      const meanY = kpList.reduce((acc, kp) => kp.position.y + acc, 0) / kpList.length;
      const stdY = Math.sqrt(kpList.reduce((acc, kp) => {
        const diff = kp.position.y - meanY;
        return diff * diff;
      }, meanY));
      console.log("std",stdY);
      return stdY < flatThreshold;
    };

    const inLeftRegion = checkRegionBuilder(videoWidth * 0.1, videoHeight * 0.05, videoWidth * 0.1 + boxSize, videoHeight * 0.05 + boxSize);
    const inRightRegion = checkRegionBuilder(videoWidth * 0.9 - boxSize, videoHeight * 0.05, videoWidth * 0.9 - boxSize + boxSize, videoHeight * 0.05 + boxSize);

    poses.forEach(({
      score,
      keypoints,
    }) => {
      if (score >= minPoseConfidence) {
        if (guiState.output.showPoints) {
          drawKeypoints(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showSkeleton) {
          drawSkeleton(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showBoundingBox) {
          drawBoundingBox(keypoints, ctx);
        }

        const token = localStorage.getItem('token') || 'f376dfd5fcfa44a4a303e9547353a50a';
        const tokenList = {
          'IRtv1': '07ebe52bf99b42b0a8afe1714343c00e',
          'AC_1': 'bea6ed1a6e254974835a770e48222e99',
          'AC_2': '9d57901983264434be254166f6f7ba3d',
          'SmartSwitch1': '5288a4f94c0d4c32bee03bfa4c4968c1',
          'IRtv2': '5320fda062474dabba17a23ccd5ee861',
          'SmartSwitch2': '49e46f7fe07540f4b10e5b33f21cbabd',
          'SmartPlug1': '8678b16d252440418264c46b1eaffffd',
          'SmartPlug2': 'f376dfd5fcfa44a4a303e9547353a50a',
          'SmartSwitch3': '5a1383e214f34823bb9ebe22ff8d5d16',
          'SmartPlug3': '3de9abcd4f8b41dd91ebffc6e5ac9008',
        };

        keypoints.forEach((kp) => {
          let flag = 0;
          if (inRightRegion(kp.position.x, kp.position.y)) {
            switch (kp.part) {
              case 'leftWrist':
                fetch(`http://blynk-cloud.com/${token}/update/V2?value=1`);
                console.log(`Left Detected: Command On to ${token}`);
                drawRightBox('lime', 10);
                flag += 1;
                break;
              case 'rightWrist':
                fetch(`http://blynk-cloud.com/${tokenList['SmartPlug1']}/update/V2?value=1`);
                fetch(`http://blynk-cloud.com/${tokenList['SmartPlug3']}/update/V2?value=1`);
                console.log(`Right Detected: Command On to ${tokenList['SmartPlug1']}`);
                drawRightBox('lime', 10);

                break;
            }
          }

          if (inLeftRegion(kp.position.x, kp.position.y)) {
            switch (kp.part) {
              case 'rightWrist':
                if (holdCounter > 10) {
                  drawLeftBox('red', 10);
                  console.log(`!!!`);
                } else {
                  drawLeftBox('yellow', 10);
                  fetch(`http://blynk-cloud.com/${token}/update/V2?value=0`);
                  console.log(`Left Detected: Command Off to ${token}`);
                  flag += 1;
                }
                break;
              case 'leftWrist':
                if (holdCounter > 10) {
                  drawLeftBox('red', 10);
                  console.log(`!!!`);
                } else {
                  drawLeftBox('yellow', 10);
                  fetch(`http://blynk-cloud.com/${tokenList['SmartPlug1']}/update/V2?value=0`);
                  fetch(`http://blynk-cloud.com/${tokenList['SmartPlug3']}/update/V2?value=0`);
                  console.log(`Right Detected: Command Off to ${tokenList['SmartPlug1']}`);
                }
                break;
            }
          }

          if (flag == 2) {
            holdCounter += 1;
          }
        });

        if (checkIfLayOnGround(keypoints)) {
          const storedColor = ctx.strokeStyle;
          ctx.strokeStyle = 'red';
          ctx.lineWidth = 10;
          drawBoundingBox(keypoints, ctx);
          ctx.stroke();
          ctx.strokeStyle = storedColor;
          downCount++;
          if (downCount  > 20 && x.ended){
            console.log("warning!!! man down man down!!!");
            document.getElementById('xyz').play();
            downCount = 0;
          }
        }
      }
    });

    // End monitoring code for frames per second
    stats.end();

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}


/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  // Load the PoseNet model weights with architecture 0.75
  const net = await posenet.load(0.75);

  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';

  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
      'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  setupGui([], net);
  setupFPS();
  detectPoseInRealTime(video, net);
}

navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
