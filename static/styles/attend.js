// Global variables
let faceDetectionInterval;
let isAutoCaptureEnabled = true;

// Start webcam
const video = document.getElementById('webcam');
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      const guide = document.getElementById('positionGuide');
      guide.style.display = 'block';

      // Set initial color
      guide.style.setProperty('--corner-color', 'red');
    };
    startFaceDetection();
  })
  .catch(err => {
    console.error("Error accessing webcam: ", err);
  });
function updatePositionGuide() {
  const guide = document.getElementById('positionGuide');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const hasFace = imageData.some(channel => channel !== 0);

  if (hasFace) {
    // Change to bright green when face is properly positioned
    guide.style.setProperty('--corner-color', '#00ff00');
    guide.querySelector('.position-text').textContent = 'HOLD POSITION';
    guide.querySelector('.position-text').style.color = '#00ff00';
  } else {
    // Bold red when searching for face
    guide.style.setProperty('--corner-color', 'red');
    guide.querySelector('.position-text').textContent = 'POSITION FACE HERE';
    guide.querySelector('.position-text').style.color = 'red';
  }
}

// detectFace function:
function detectFace() {
  const canvas = document.getElementById('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  //  position guide
  updatePositionGuide();

  // detection logic...
  const imageData = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
  const hasContent = imageData.some(channel => channel !== 0);

  if (hasContent) {
    document.getElementById('faceDetected').style.display = 'block';
    setTimeout(() => {
      captureAndSend();
    }, 2000);
  } else {
    document.getElementById('faceDetected').style.display = 'none';
  }
}
// Start face detection
function startFaceDetection() {
  faceDetectionInterval = setInterval(() => {
    if (isAutoCaptureEnabled) {
      detectFace();
    }
  }, 1000); // Check every second
}

// Detect face using MediaPipe (simulated here - actual detection would be server-side)
function detectFace() {
  const canvas = document.getElementById('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);


  // simulate face detection by checking if the video has content
  const imageData = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
  const hasContent = imageData.some(channel => channel !== 0);

  if (hasContent) {
    document.getElementById('faceDetected').style.display = 'block';
    setTimeout(() => {
      captureAndSend();
    }, 2000); // Auto-capture after 2 seconds of face detection
  } else {
    document.getElementById('faceDetected').style.display = 'none';
  }
}

// Capture frame and send to server
function captureAndSend() {
  if (!isAutoCaptureEnabled) return;

  isAutoCaptureEnabled = false; // Prevent multiple captures
  document.getElementById('loading').style.display = 'block';
  document.getElementById('result').innerText = '';
  document.getElementById('recordingIndicator').style.display = 'block';
  document.getElementById('manualCaptureBtn').disabled = true;

  const canvas = document.getElementById('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    const formData = new FormData();
    formData.append('image', blob, 'capture.jpg');

    fetch('/mark_attendance', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      document.getElementById('loading').style.display = 'none';
      document.getElementById('recordingIndicator').style.display = 'none';
      document.getElementById('faceDetected').style.display = 'none';

      document.getElementById('result').innerText = data.message;

      if (data.status === 'success') {
        document.getElementById('recognizedImage').style.display = 'block';
        document.getElementById('recognizedImage').src = "/static/faces/marked.jpg?" + new Date().getTime();
      }

      // Re-enable auto capture after 4 seconds
      setTimeout(() => {
        isAutoCaptureEnabled = true;
        document.getElementById('manualCaptureBtn').disabled = false;
      }, 4000);
    })
    .catch(error => {
      document.getElementById('loading').style.display = 'none';
      document.getElementById('recordingIndicator').style.display = 'none';
      document.getElementById('faceDetected').style.display = 'none';

      document.getElementById('result').innerText = 'Error capturing face. Please try again.';
      console.error(error);

      // Re-enable auto capture after error
      isAutoCaptureEnabled = true;
      document.getElementById('manualCaptureBtn').disabled = false;
    });
  }, 'image/jpeg');
}

// Edit Toggle
let editing = false;
function toggleEdit() {
  editing = !editing;
  if (editing) {
    // Show inputs
    document.getElementById('logoInput').style.display = 'block';
    document.getElementById('aimInput').style.display = 'block';
    document.getElementById('orgInput').style.display = 'block';

    // Hide static text
    document.getElementById('aimText').style.display = 'none';
    document.getElementById('orgName').style.display = 'none';

    // Fill inputs with current text
    document.getElementById('aimInput').value = document.getElementById('aimText').innerText.trim();
    document.getElementById('orgInput').value = document.getElementById('orgName').innerText.trim();
  } else {
    // Save and hide inputs
    document.getElementById('aimText').innerText = document.getElementById('aimInput').value;
    document.getElementById('orgName').innerText = document.getElementById('orgInput').value;

    document.getElementById('logoInput').style.display = 'none';
    document.getElementById('aimInput').style.display = 'none';
    document.getElementById('orgInput').style.display = 'none';

    document.getElementById('aimText').style.display = 'block';
    document.getElementById('orgName').style.display = 'block';
  }
}

// Logo Preview
function changeLogo() {
  const input = document.getElementById('logoInput');
  const logo = document.getElementById('logoImage');
  if (input.files && input.files[0]) {
    const reader = new FileReader();
    reader.onload = function (e) {
      logo.src = e.target.result;
    };
    reader.readAsDataURL(input.files[0]);
  }
}

// Date functions
function updateDateInfo() {
    const now = new Date();

    // Month name
    const monthNames = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ];
    document.getElementById("month").textContent = monthNames[now.getMonth()];

    // Year
    document.getElementById("year").textContent = now.getFullYear();

    // Week number (ISO week number)
    function getWeekNumber(d) {
        d = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()));
        d.setUTCDate(d.getUTCDate() + 4 - (d.getUTCDay() || 7));
        const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
        const weekNo = Math.ceil((((d - yearStart) / 86400000) + 1) / 7);
        return weekNo;
    }
    document.getElementById("week").textContent = "Week " + getWeekNumber(now);
}

// Run it when page loads
document.addEventListener('DOMContentLoaded', updateDateInfo);