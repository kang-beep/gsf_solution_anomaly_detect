document.addEventListener('DOMContentLoaded', function() {
    // State
    let isSaving = false;
    let isToolActive = false;

    // 전역 변수로 선언
    let ACTIVE_SERVER_URL;

    // 페이지 로드 시 실행되는 함수
    async function initializeServer() {
        try {
            ACTIVE_SERVER_URL = await checkServerStatus();
            alert(`카메라 서버가 연결되었습니다: ${ACTIVE_SERVER_URL}`);
        } catch (error) {
            alert("모든 카메라 서버가 응답하지 않습니다.");
        }
    }

    async function checkServerStatus() {
        const ACTIVE_SERVER_URL = 'localhost:8888';
        const FALLBACK_URL = '100.75.8.21:8888';
        
        try {
            await Promise.race([
                fetch(`http://${ACTIVE_SERVER_URL}/api/health/health_check`),
                new Promise((_, reject) => setTimeout(reject, 5000))
            ]);
            return ACTIVE_SERVER_URL;
        } catch {
            try {
                await Promise.race([
                    fetch(`http://${FALLBACK_URL}/api/health/health_check`),
                    new Promise((_, reject) => setTimeout(reject, 5000))
                ]);
                return FALLBACK_URL;
            } catch {
                throw new Error("No servers available");
            }
        }
    }
    

    // DOM Elements
    const elements = {
        videoFeed: document.getElementById('videoFeed'),
        cameraSelect: document.getElementById('cameraSelect'),
        selectCameraIndex: document.getElementById('selectCameraIndex'),
        resolutionSelect: document.getElementById('resolutionSelect'),

        toggleButton: document.getElementById('toggleButton'),
        statusIndicator: document.getElementById('statusIndicator'),
        hsvDisplay: document.getElementById('hsvDisplay'),
        configHsvDisplay: document.getElementById('configHsvDisplay'),
        hsvRange: document.getElementById('hsvRange'),
        hsvRangeValue: document.getElementById('hsvRangeValue'),
        rowSelect: document.getElementById('rowSelect'),
        autoInputBtn: document.getElementById('autoInputBtn'),
        configTableBody: document.getElementById('configTableBody'),
        videoSection: document.getElementById('videoSection'),
        processedImageSection: document.getElementById('processedImageSection'),
        processedImage: document.getElementById('processedImage'),
        viewModeSelect: document.getElementById('viewModeSelect'),
        savingStatus: document.getElementById('savingStatus'),

        // 저장상태 부분
        saveSelect: document.getElementById('saveSelect'),
        saveStartBtn: document.getElementById('saveStartBtn'),
        saveStopBtn: document.getElementById('saveStopBtn'),
        savingStatus: document.getElementById('savingStatus'),

        // 모드 선택 부분
        detectionUI : document.getElementById('detectionUI'),
        brightnessUI : document.getElementById('brightnessUI'),

    };

    // 선택된 카메라 정보를 저장할 전역 변수
    let selectedCamera = {
        index: null,
        resolutions: [],
    }

    // 카메라 선택 버튼 이벤트 리스너
    document.getElementById('camSelect').addEventListener('click', async function() {
        const selectedValue = elements.cameraSelect.value;
        if (!selectedValue) {
            alert("카메라를 선택해주세요.");
            return;
        }

        selectedCamera.index = selectedValue;
        elements.selectCameraIndex.textContent = `선택한 카메라 번호 ${selectedValue}`;

        // 서버에 선택된 카메라 정보 전송 및 해상도 목록 요청
        try {
            const response = await fetch(`http://${ACTIVE_SERVER_URL}/api/camera/add_camera?camera_index=${selectedValue}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    camera_index: selectedValue
                })
            });

            if (!response.ok) {
                throw new Error('서버 응답 오류');
            }

            const data = await response.json();
            selectedCamera.resolutions = data.resolutions;

            // 해상도 선택 옵션 업데이트
            const resolutionSelect = document.getElementById('resolutionSelect');
            resolutionSelect.innerHTML = '';
            data.resolutions.forEach(resolution => {
                const option = document.createElement('option');
                option.value = resolution;
                option.text = resolution;
                resolutionSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Error:', error);
            alert('카메라 해상도 정보를 가져오는데 실패했습니다.');
        }
    });

    // Initialize
    init();

    function init() {
        initializeServer();
        elements.autoInputBtn.disabled = true;
        elements.hsvDisplay.hidden = true;
        elements.saveStartBtn.disabled = true; // 초기 버튼 비활성화

        // 모드 선택 초기
        elements.detectionUI.style.display = 'block';
        elements.brightnessUI.style.display = 'none';

        attachEventListeners();
        updateRowSelect();
    }

    function attachEventListeners() {
        elements.toggleButton.addEventListener('click', toggleTool);
        elements.hsvRange.addEventListener('input', updateHsvRangeValue);
        elements.rowSelect.addEventListener('change', handleRowSelect);
        elements.autoInputBtn.addEventListener('click', handleAutoInput);
        elements.configTableBody.addEventListener('click', handleRowRemove);
        elements.viewModeSelect.addEventListener('change', handleViewModeChange);
        elements.saveSelect.addEventListener('change', handleSaveOptionChange);


        // 모드 선택 이벤트 리스너 추가
        const modeSelection = document.getElementById('modeSelection');
        if (modeSelection) {
            const radioButtons = modeSelection.querySelectorAll('input[type="radio"]');
            radioButtons.forEach(radio => {
                radio.addEventListener('change', handleModeChange);
            });
        }


        document.getElementById('refreshBtn').addEventListener('click', refreshCameraList);
        document.getElementById('camOpenBtn').addEventListener('click', startStreaming);
        document.getElementById('addRowBtn').addEventListener('click', addRow);
        document.getElementById('detectBtn').addEventListener('click', requestProcessing);
        document.getElementById('saveStartBtn').addEventListener('click', saveStart);
        document.getElementById('saveStopBtn').addEventListener('click', saveStop);
    }

    // Camera Functions
    function refreshCameraList() {
        if (isSaving) {
            alert("이미지 저장중이므로 다른 동작을 수행하실 수 없습니다.");
            return;
        }

        fetch(`http://${ACTIVE_SERVER_URL}/api/camera/available`)
            .then(response => response.json())
            .then(data => {
                elements.cameraSelect.innerHTML = '<option value="">Select a camera</option>';
                data.forEach(cameraIndex => {   // data.available_cameras 대신 data 사용
                    const option = document.createElement('option');
                    option.value = cameraIndex;
                    option.text = `Camera ${cameraIndex}`;
                    elements.cameraSelect.appendChild(option);
                });
            });
        elements.videoFeed.src = "/static/imgs/ImageIcon.png";
    }


    function handleModeChange() {
        const detectionUI = document.getElementById('detectionUI');
        const brightnessUI = document.getElementById('brightnessUI');
        
        if (this.value === 'detection') {
            detectionUI.style.display = 'block';
            brightnessUI.style.display = 'none';
        } else {
            detectionUI.style.display = 'none';
            brightnessUI.style.display = 'block';
        }
    }

    async function startStreaming() {
        if (isSaving) {
            alert("이미지 저장중이므로 다른 동작을 수행하실 수 없습니다.");
            return;
        }
    
        const selectedCamera = elements.cameraSelect.value;
        const selectedResolution = elements.resolutionSelect.value;
    
        if (!selectedCamera) {
            alert("Please select a camera.");
            return;
        }
    
        if (!selectedResolution) {
            alert("Please select a resolution.");
            return;
        }
    
        try {
            // 해상도 값을 width와 height로 분리
            const [width, height] = selectedResolution.split('x').map(Number);
    
            // FormData 생성 및 데이터 추가
            const formData = new FormData();
            formData.append('camera_index', selectedCamera);
            formData.append('width', width);
            formData.append('height', height);
    
            // 서버에 카메라 설정 요청
            const response = await fetch(`http://${ACTIVE_SERVER_URL}/api/camera/open_camera`, {
                method: 'POST',
                body: formData
            });
    
            if (!response.ok) {
                throw new Error('서버 응답 오류');
            }
    
            // 성공하면 기존 비디오 피드 로직 실행
            elements.videoFeed.src = `http://${ACTIVE_SERVER_URL}/api/video/feed?camera_index=${selectedCamera}`;
            elements.videoFeed.onerror = () => {
                alert("Failed to load video stream. Please check the camera.");
            };
        } catch (error) {
            console.error('Error:', error);
            alert('카메라 열기에 실패했습니다.');
        }
    }
    

    function stopStreamingAndShowFrame() {
        const selectedCamera = elements.cameraSelect.value;
        const stream = elements.videoFeed.srcObject;
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            elements.videoFeed.srcObject = null;
        }

        fetch(`http://${ACTIVE_SERVER_URL}/api/camera/image_feed?camera_index=${selectedCamera}`)
            .then(response => response.blob())
            .then(blob => {
                const imageURL = URL.createObjectURL(blob);
                elements.videoFeed.src = imageURL;
                elements.videoFeed.onload = enableHsvExtraction;
                elements.videoFeed.onerror = () => console.error("Failed to load the image.");
            })
            .catch(error => console.error("Error fetching image from server:", error));

        alert("펜 모드 활성화");
    }

    // HSV Extraction
    function enableHsvExtraction() {
        elements.videoFeed.removeEventListener('click', extractHsvOnClick);
        elements.videoFeed.addEventListener('click', extractHsvOnClick);
    }

    function extractHsvOnClick(event) {
        const rect = elements.videoFeed.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
    
        const canvas = document.createElement('canvas');
        canvas.width = elements.videoFeed.width;
        canvas.height = elements.videoFeed.height;
        const ctx = canvas.getContext('2d');
        
        const img = new Image();
        img.crossOrigin = "anonymous";
        
        img.onload = () => {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            const pixelData = ctx.getImageData(x, y, 1, 1).data;
            const [r, g, b] = [pixelData[0], pixelData[1], pixelData[2]];
    
            fetch(`http://${ACTIVE_SERVER_URL}/api/image/rgb2hsv`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ r, g, b })
            })
            .then(response => response.json())
            .then(hsv => {
                elements.hsvDisplay.textContent = `HSV: H(${hsv.h}), S(${hsv.s}), V(${hsv.v})`;
                updateHsvDisplay(hsv.h, hsv.s, hsv.v);
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error occurred while converting RGB to HSV.');
            });
        };
        
        img.src = elements.videoFeed.src;
    }

    // UI Functions
    function toggleTool() {
        isToolActive = !isToolActive;
        elements.toggleButton.textContent = isToolActive ? 'Stop' : 'Live';
        elements.toggleButton.classList.toggle('btn-danger', isToolActive);
        elements.toggleButton.classList.toggle('btn-primary', !isToolActive);
        elements.statusIndicator.style.color = isToolActive ? 'red' : 'green';
        elements.statusIndicator.textContent = isToolActive ? 'Stopped' : 'Live';
        elements.hsvDisplay.hidden = !isToolActive;

        if (isToolActive) {
            stopStreamingAndShowFrame();
        } else {
            startStreaming();
        }
    }

    function updateHsvRangeValue() {
        elements.hsvRangeValue.textContent = elements.hsvRange.value;
    }

    function updateHsvDisplay(h, s, v) {
        elements.configHsvDisplay.textContent = `HSV: H(${h}), S(${s}), V(${v})`;
        updateRowSelect();
    }

    function updateRowSelect() {
        elements.rowSelect.innerHTML = '<option value="">Select</option>';
        const rows = elements.configTableBody.querySelectorAll('tr');
        rows.forEach((row, index) => {
            const option = document.createElement('option');
            option.value = index + 1;
            option.textContent = `Row ${index + 1}`;
            elements.rowSelect.appendChild(option);
        });
    }

    function handleRowSelect() {
        elements.autoInputBtn.disabled = !this.value;
    }

    function handleAutoInput() {
        const selectedRow = parseInt(elements.rowSelect.value) - 1;
        const range = parseInt(elements.hsvRange.value);
        const hsv = elements.configHsvDisplay.textContent.match(/H\((\d+)\), S\((\d+)\), V\((\d+)\)/);
        
        if (hsv && selectedRow >= 0) {
            const [h, s, v] = [parseInt(hsv[1]), parseInt(hsv[2]), parseInt(hsv[3])];
            const row = elements.configTableBody.querySelectorAll('tr')[selectedRow];
            
            row.querySelector('input[name="lower_h"]').value = Math.max(0, h - range);
            row.querySelector('input[name="lower_s"]').value = Math.max(0, s - range);
            row.querySelector('input[name="lower_v"]').value = Math.max(0, v - range);
            
            row.querySelector('input[name="upper_h"]').value = Math.min(179, h + range);
            row.querySelector('input[name="upper_s"]').value = Math.min(255, s + range);
            row.querySelector('input[name="upper_v"]').value = Math.min(255, v + range);
        }
    }

    function addRow() {
        const rowCount = elements.configTableBody.rows.length + 1;
        const newRow = document.createElement('tr');
        newRow.innerHTML = `
            <td>${rowCount}</td>
            <td><input type="number" class="form-control" name="margin" placeholder="Enter Margin"></td>
            <td>
                <div style="display: flex; gap: 5px;">
                    <input type="number" class="form-control" name="lower_h" placeholder="H" style="width: 100px;">
                    <input type="number" class="form-control" name="lower_s" placeholder="S" style="width: 100px;">
                    <input type="number" class="form-control" name="lower_v" placeholder="V" style="width: 100px;">
                </div>
            </td>
            <td>
                <div style="display: flex; gap: 5px;">
                    <input type="number" class="form-control" name="upper_h" placeholder="H" style="width: 100px;">
                    <input type="number" class="form-control" name="upper_s" placeholder="S" style="width: 100px;">
                    <input type="number" class="form-control" name="upper_v" placeholder="V" style="width: 100px;">
                </div>
            </td>
            <td><button type="button" class="btn btn-outline-danger btn-sm remove-row">삭제</button></td>
        `;
        elements.configTableBody.appendChild(newRow);
        updateRowSelect();
    }

    function handleRowRemove(e) {
        if (e.target.classList.contains('remove-row')) {
            e.target.closest('tr').remove();
            updateRowSelect();
        }
    }

    function handleViewModeChange() {
        const selectedMode = this.value;
        if (selectedMode === '1|2') {
            resetSections();
            showBothSections();
        } else if (selectedMode === '1') {
            resetSections();
            showOnlyVideoSection();
        } else if (selectedMode === '2') {
            resetSections();
            showOnlyProcessedSection();
        }
    }

    function resetSections() {
        [elements.videoSection, elements.processedImageSection].forEach(section => {
            if (section) {
                section.style.width = '';
                section.style.maxWidth = '';
                section.classList.remove('d-none');
                section.classList.remove('col-md-12');
                section.classList.add('col-md-6');
            }
        });
    
        [elements.videoFeed, elements.processedImage].forEach(image => {
            if (image) {
                image.classList.remove('fullscreen');
            }
        });
    }

    
    function showBothSections() {
        elements.videoSection.classList.remove('d-none');
        elements.processedImageSection.classList.remove('d-none');
    }

    function showOnlyVideoSection() {
        elements.videoSection.classList.remove('col-md-6');
        elements.videoSection.classList.add('col-md-12');
        elements.processedImageSection.classList.add('d-none');
        applyFullscreen(elements.videoSection, elements.videoFeed);
    }
    
    function showOnlyProcessedSection() {
        elements.processedImageSection.classList.remove('col-md-6');
        elements.processedImageSection.classList.add('col-md-12');
        elements.videoSection.classList.add('d-none');
        applyFullscreen(elements.processedImageSection, elements.processedImage);
    }

    function applyFullscreen(section, image, maxWidth) {
        image.classList.add('fullscreen');
        section.style.width = "100%";
        section.style.maxWidth = maxWidth;
    }

    // Data Processing
    function collectTableData() {
        const rows = elements.configTableBody.querySelectorAll('tr');
        return Array.from(rows).map((row, index) => ({
            rowIndex: index + 1,
            margin: parseFloat(row.querySelector('input[name="margin"]').value),
            lower_bound: {
                h: parseFloat(row.querySelector('input[name="lower_h"]').value),
                s: parseFloat(row.querySelector('input[name="lower_s"]').value),
                v: parseFloat(row.querySelector('input[name="lower_v"]').value)
            },
            upper_bound: {
                h: parseFloat(row.querySelector('input[name="upper_h"]').value),
                s: parseFloat(row.querySelector('input[name="upper_s"]').value),
                v: parseFloat(row.querySelector('input[name="upper_v"]').value)
            }
        }));
    }

    function requestProcessing() {
        if (isSaving) {
            alert("이미지 저장중이므로 다른 동작을 수행하실 수 없습니다.");
            return;
        }
        
        if (!isToolActive) {
            alert('error : 실시간 영상이 stop 상태인지 확인해주세요');
            return;
        }

        const tableData = collectTableData();
    
        fetch(`http://${ACTIVE_SERVER_URL}/api/image/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(tableData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success' || data.status === 'partial_success') {
                processedImages(data.image_paths);
                if (data.alert_message) alert(data.alert_message);
                if (data.failed_detections.length > 0) {
                    console.log('Failed detections for rows:', data.failed_detections);
                }
                alert(`Images processed. Successful: ${data.successful_detections}, Failed: ${data.failed_detections.length}`);
            } else {
                throw new Error('Image processing failed');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('error : 카메라가 Stop 상태인지 확인해 주시고, 값을 정확하게 입력했는지 확인해주세요.');
        });
    }

    function processedImages(imagePaths) {
        const select = document.getElementById('processedImageSelect');
        select.innerHTML = '<option value="default">Select Image</option>';
        select.add(new Option('FULL', 'FULL'));
    
        imagePaths.forEach((path, index) => {
            select.add(new Option(`Image ${index + 1}`, path));
        });
    
        function updateImage(src) {
            const timestamp = new Date().getTime();
            elements.processedImage.src = `${src}?t=${timestamp}`;
        }
    
        if (imagePaths.length > 0) {
            updateImage(`http://${ACTIVE_SERVER_URL}${imagePaths[0].replace(/^\./, '')}`);
            select.value = imagePaths[0];
        } else {
            elements.processedImage.src = "/static/imgs/ImageIcon.png";
            select.value = "default";
        }
    
        select.onchange = function() {
            if (this.value === "FULL") {
                updateImage(`http://${ACTIVE_SERVER_URL}/temp/output/image/all_crop.png`);
            } else if (this.value !== "default") {
                updateImage(`http://${ACTIVE_SERVER_URL}${this.value.replace(/^\./, '')}`);
            } else {
                elements.processedImage.src = "/static/imgs/ImageIcon.png";
            }
        };
    }
    
    function handleSaveOptionChange(){
        elements.saveStartBtn.disabled = !this.value;
    }
    function saveStart() {
        if (isSaving) {
            alert("이미 저장 중입니다.");
            return;
        }
        
        const selectedOption = elements.saveSelect.value;
        let endpoint = `http://${ACTIVE_SERVER_URL}/api/`;

        switch(selectedOption) {
            case 'image_sender':
                endpoint += 'sender/start_sending';
                break;
            case 'detection_image':
                endpoint += 'image/save/start';
                break;
            case 'full_video':
                endpoint += 'video/full/start';
                break;
            case 'detection_video':
                endpoint += 'video/detection/start';
                break;
            default:
                alert('저장 옵션을 선택해주세요.');
                return;
        }

        fetch(endpoint)
        .then(response => {
            if(!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            isSaving = true;
            console.log('Save started:', data);
            alert('저장이 시작되었습니다.');
            
            isToolActive = false;
            updateToolState();

            elements.savingStatus.textContent = "현재 저장 중입니다. 페이지를 끄거나 변경하지 말아주세요.";
            elements.savingStatus.style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error occurred while starting image save.');
        });
    }

    function saveStop() {
        fetch(`http://${ACTIVE_SERVER_URL}/api/video/stop/`)
            .then(response => response.json())
            .then(data => {
                console.log('Save stopped:', data);
                alert('Image saving stopped.');
                
                elements.savingStatus.style.display = 'none';
                isSaving = false; 
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error occurred while stopping image save.');
            });
    }

    
    function updateToolState() {
        elements.toggleButton.classList.toggle('active', isToolActive);
        elements.toggleButton.textContent = isToolActive ? 'Stop' : 'Start';
        elements.toggleButton.classList.toggle('btn-danger', isToolActive);
        elements.toggleButton.classList.toggle('btn-primary', !isToolActive);
        elements.statusIndicator.style.color = isToolActive ? 'red' : 'green';
        elements.statusIndicator.textContent = isToolActive ? 'Stopped' : 'Live';
        elements.hsvDisplay.hidden = !isToolActive;
    }
});