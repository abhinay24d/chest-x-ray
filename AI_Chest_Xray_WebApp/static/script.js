/* ==========================================================================
   AI Chest X-Ray Disease Detection System - Application Logic
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {

    // 1. Navigation & Scroll Interactions
    const navbar = document.getElementById('navbar');
    const navToggle = document.getElementById('navToggle');
    const navMenu = document.getElementById('navMenu');
    const navLinks = document.querySelectorAll('.nav-link');

    // Solid navbar background on scroll
    window.addEventListener('scroll', () => {
        if (window.scrollY > 20) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // Mobile navigation drawer toggle
    navToggle.addEventListener('click', () => {
        navMenu.classList.toggle('active');
        navToggle.innerHTML = navMenu.classList.contains('active') ? '&#10005;' : '&#9776;';
    });

    // Close menu when links are clicked on mobile
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            navMenu.classList.remove('active');
            navToggle.innerHTML = '&#9776;';
        });
    });

    // 2. Upload Elements & State Management
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const uploadPrompt = document.getElementById('uploadPrompt');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const uploadedInfo = document.getElementById('uploadedInfo');
    const fileNameDisplay = document.getElementById('fileName');
    const fileSizeDisplay = document.getElementById('fileSize');
    const changeFileBtn = document.getElementById('changeFileBtn');
    
    const analyzeBtn = document.getElementById('analyzeBtn');
    const analyzeBtnText = document.getElementById('analyzeBtnText');
    const analyzeBtnIcon = document.getElementById('analyzeBtnIcon');
    const progressLog = document.getElementById('progressLog');
    const analysisScan = document.getElementById('analysisScan');

    const resultsWaiting = document.getElementById('resultsWaiting');
    const resultsCard = document.getElementById('resultsCard');
    const severityBadge = document.getElementById('severityBadge');
    const detectedDisease = document.getElementById('detectedDisease');
    const diseaseDisplay = document.getElementById('diseaseDisplay');

    const pctNormal = document.getElementById('pctNormal');
    const pctPneumonia = document.getElementById('pctPneumonia');
    const pctTB = document.getElementById('pctTB');
    const barNormal = document.getElementById('barNormal');
    const barPneumonia = document.getElementById('barPneumonia');
    const barTB = document.getElementById('barTB');

    let uploadedFile = null; // Stores file object

    // Trigger input browse click on dropzone click
    dropzone.addEventListener('click', (e) => {
        // Prevent click if clicking preview or buttons
        if (e.target.closest('#previewContainer') || e.target.closest('.change-file-btn') || uploadedFile) {
            return;
        }
        fileInput.click();
    });

    // File input selection listener
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    // Drag and Drop visual overrides
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        handleFiles(dt.files);
    });

    // Process file uploads
    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        // Validation: Support JPG, JPEG, PNG
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            showErrorMessage('Invalid file format. Please upload a chest X-Ray in PNG or JPG format.');
            fileInput.value = '';
            return;
        }

        uploadedFile = file;
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = () => {
            displayPreview(reader.result, file.name, formatBytes(file.size));
        };
    }

    // Helper format file sizes
    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    // Reset workflow
    function resetUpload() {
        uploadedFile = null;
        fileInput.value = '';
        
        // Update states
        previewImage.src = '';
        previewContainer.style.display = 'none';
        uploadPrompt.style.display = 'block';
        uploadedInfo.style.display = 'none';
        
        analyzeBtn.disabled = true;
        analysisScan.style.display = 'none';
        
        progressLog.innerHTML = '<span>Ready to ingest clinical scan data...</span>';
        progressLog.classList.remove('visible');

        resultsCard.style.display = 'none';
        resultsWaiting.style.display = 'flex';
    }

    changeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    // Populate UI with preview details
    function displayPreview(src, name, sizeString) {
        previewImage.src = src;
        uploadPrompt.style.display = 'none';
        previewContainer.style.display = 'flex';
        
        fileNameDisplay.textContent = name;
        fileSizeDisplay.textContent = sizeString;
        uploadedInfo.style.display = 'flex';

        // Reset scanning/heatmap states
        analysisScan.style.display = 'none';
        
        analyzeBtn.disabled = false;
        progressLog.classList.add('visible');
        progressLog.innerHTML = '<span>Scan loaded. Ready to run Random Forest AI classification.</span>';

        resultsCard.style.display = 'none';
        resultsWaiting.style.display = 'flex';
    }

    // Helper to log UI error messages
    function showErrorMessage(msg) {
        progressLog.classList.add('visible');
        progressLog.innerHTML = `<span style="color: var(--danger);">[ERROR] ${msg}</span>`;
    }



    // 4. Model Ingest & Real Prediction
    analyzeBtn.addEventListener('click', performPrediction);

    function performPrediction() {
        if (!uploadedFile) {
            showErrorMessage('No image file selected for analysis.');
            return;
        }

        // Disable UI controls
        setInteractiveState(false);
        resultsWaiting.style.display = 'flex';
        resultsCard.style.display = 'none';

        // Run scanner animation on image
        analysisScan.style.display = 'block';

        // Trigger step-by-step progress tickers
        const logSteps = [
            { time: 0, text: '<span>[SYSTEM]</span> Ingesting chest X-ray image...' },
            { time: 600, text: '<span>[SYSTEM]</span> Dispatching scan payload to Flask backend...' },
            { time: 1200, text: '<span>[BACKEND]</span> Running PIL grayscale conversion & 64x64 rescaling...' },
            { time: 1800, text: '<span>[MODEL]</span> Flattening features to (1, 4096) and running Random Forest trees...' }
        ];

        logSteps.forEach(step => {
            setTimeout(() => {
                progressLog.innerHTML = step.text;
                progressLog.scrollTop = progressLog.scrollHeight;
            }, step.time);
        });

        // Initialize FormData payload
        const formData = new FormData();
        formData.append('file', uploadedFile);

        // Fetch predict API
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(res => {
            if (!res.ok) {
                // Parse server error
                return res.json().then(errData => {
                    throw new Error(errData.error || `Server responded with status ${res.status}`);
                }).catch(() => {
                    throw new Error(`HTTP network error. Status: ${res.status}`);
                });
            }
            return res.json();
        })
        .then(data => {
            // Wait for visual delay (matching log timings) to ensure smooth transition
            const delay = Math.max(0, 2400 - (Date.now() - startTime));
            
            setTimeout(() => {
                analysisScan.style.display = 'none';
                setInteractiveState(true);
                
                if (data.success) {
                    progressLog.innerHTML = '<span>[SUCCESS]</span> Machine learning classification completed successfully.';
                    renderResults(data);
                } else {
                    showErrorMessage(data.error || 'Unknown error occurred during prediction.');
                }
            }, 500);
        })
        .catch(err => {
            analysisScan.style.display = 'none';
            setInteractiveState(true);
            showErrorMessage(err.message || 'Network connection failed.');
        });

        const startTime = Date.now();
    }

    // Lock/Unlock buttons during analysis
    function setInteractiveState(enabled) {
        if (enabled) {
            analyzeBtn.disabled = false;
            analyzeBtnText.textContent = 'Analyze X-Ray';
            analyzeBtnIcon.style.display = 'block';
            changeFileBtn.disabled = false;
        } else {
            analyzeBtn.disabled = true;
            analyzeBtnText.textContent = 'Predicting Pathology...';
            analyzeBtnIcon.style.display = 'none';
            changeFileBtn.disabled = true;
        }
    }

    // 5. Render Prediction Report UI
    function renderResults(data) {
        resultsWaiting.style.display = 'none';
        resultsCard.style.display = 'flex';

        // Set text
        detectedDisease.textContent = data.disease;
        
        // Remove old style classes from badge & display block
        severityBadge.className = 'badge';
        diseaseDisplay.className = 'detected-disease-display';

        if (data.disease === 'NORMAL') {
            severityBadge.textContent = 'Healthy';
            severityBadge.classList.add('badge-success');
            diseaseDisplay.classList.add('class-normal');
        } else if (data.disease === 'PNEUMONIA') {
            severityBadge.textContent = 'Attention Required';
            severityBadge.classList.add('badge-warning');
            diseaseDisplay.classList.add('class-pneumonia');
        } else if (data.disease === 'TUBERCULOSIS') {
            severityBadge.textContent = 'Consult Doctor';
            severityBadge.classList.add('badge-danger');
            diseaseDisplay.classList.add('class-tb');
        }

        // Animate count-up for numbers and width expansion for tracks
        animateMetric(pctNormal, barNormal, data.confidences.NORMAL);
        animateMetric(pctPneumonia, barPneumonia, data.confidences.PNEUMONIA);
        animateMetric(pctTB, barTB, data.confidences.TUBERCULOSIS);
    }

    // Custom metric ticker animation
    function animateMetric(textEl, barEl, targetVal) {
        let currentVal = 0;
        barEl.style.width = '0%';
        textEl.textContent = '0%';
        
        if (targetVal === 0) return;

        const duration = 1200; // Match CSS transition duration
        const stepTime = Math.max(Math.floor(duration / targetVal), 15);
        
        // Animate width immediately using CSS transition
        setTimeout(() => {
            barEl.style.width = targetVal + '%';
        }, 50);

        // Incremental text update ticker
        const timer = setInterval(() => {
            currentVal += 1;
            textEl.textContent = currentVal + '%';
            if (currentVal >= targetVal) {
                clearInterval(timer);
                textEl.textContent = targetVal + '%'; // lock exact value
            }
        }, stepTime);
    }

});
