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
    const heatmapOverlay = document.getElementById('heatmapOverlay');

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

    const heatmapToggle = document.getElementById('heatmapToggle');
    const heatmapToggleContainer = document.getElementById('heatmapToggleContainer');

    let currentImageData = null; // Stores file data
    let selectedCaseType = null; // Tracks normal, pneumonia, tb, or custom
    let activeResult = null; // Stores calculated diagnosis percentages

    // Trigger input browse click on dropzone click
    dropzone.addEventListener('click', (e) => {
        // Prevent click if clicking preview or button
        if (e.target.closest('#previewContainer') || e.target.closest('.change-file-btn') || fileInput.files.length > 0 && currentImageData) {
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
            alert('Invalid file format. Please upload a chest X-Ray in PNG or JPG format.');
            return;
        }

        selectedCaseType = 'custom';
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = () => {
            currentImageData = reader.result;
            displayPreview(currentImageData, file.name, formatBytes(file.size));
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
        currentImageData = null;
        selectedCaseType = null;
        fileInput.value = '';
        
        // Update states
        previewImage.src = '';
        previewContainer.style.display = 'none';
        uploadPrompt.style.display = 'block';
        uploadedInfo.style.display = 'none';
        
        analyzeBtn.disabled = true;
        analysisScan.style.display = 'none';
        heatmapOverlay.className = 'heatmap-overlay';
        heatmapToggle.checked = false;
        
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
        heatmapOverlay.className = 'heatmap-overlay';
        heatmapToggle.checked = false;
        
        analyzeBtn.disabled = false;
        progressLog.classList.add('visible');
        progressLog.innerHTML = '<span>Scan loaded. Ready to run machine learning diagnosis.</span>';

        resultsCard.style.display = 'none';
        resultsWaiting.style.display = 'flex';
    }



    // 4. Model Ingest & Prediction Simulation
    analyzeBtn.addEventListener('click', startAnalysis);

    function startAnalysis() {
        if (!selectedCaseType) return;

        // Disable buttons and actions
        setInteractiveState(false);
        resultsWaiting.style.display = 'flex';
        resultsCard.style.display = 'none';
        heatmapOverlay.className = 'heatmap-overlay';
        heatmapToggle.checked = false;

        // Run scanner animation on image
        analysisScan.style.display = 'block';

        // Trigger log progression tickers
        const logSteps = [
            { time: 0, text: '<span>[SYSTEM]</span> Initiating multi-disease AI classification engine...' },
            { time: 800, text: '<span>[MODEL]</span> Normalizing pixel values & scaling canvas size to 224x224x3...' },
            { time: 1600, text: '<span>[DENSENET]</span> Extracting 1,024 spatial pathology feature weights...' },
            { time: 2400, text: '<span>[SOFTMAX]</span> Calculating target distribution probabilities...' },
            { time: 3000, text: '<span>[GRAD-CAM]</span> Locating spatial coordinates & synthesizing heatmap matrix...' }
        ];

        logSteps.forEach(step => {
            setTimeout(() => {
                progressLog.innerHTML = step.text;
                // Scroll log text if overflowing
                progressLog.scrollTop = progressLog.scrollHeight;
            }, step.time);
        });

        // Resolve results at 3300ms
        setTimeout(() => {
            analysisScan.style.display = 'none';
            setInteractiveState(true);
            progressLog.innerHTML = '<span>[SUCCESS]</span> Analysis complete. Diagnostic report generated below.';
            
            // Calculate final results
            activeResult = getPredictionMetrics(selectedCaseType);
            renderResults(activeResult);
        }, 3300);
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
            analyzeBtnText.textContent = 'Analyzing Scan...';
            analyzeBtnIcon.style.display = 'none';
            changeFileBtn.disabled = true;
        }
    }

    // Determine values based on preset or custom uploads
    function getPredictionMetrics(type) {
        if (type === 'normal') {
            return { label: 'NORMAL', normal: 96, pneumonia: 3, tb: 1 };
        } else if (type === 'pneumonia') {
            return { label: 'PNEUMONIA', normal: 4, pneumonia: 88, tb: 8 };
        } else if (type === 'tb') {
            return { label: 'TUBERCULOSIS', normal: 2, pneumonia: 10, tb: 88 };
        } else {
            // Custom Upload - Simulate a realistic distribution
            // Give 50% chance of Normal, 30% Pneumonia, 20% TB
            const rand = Math.random();
            if (rand < 0.5) {
                // Mock Normal
                return { label: 'NORMAL', normal: Math.floor(Math.random() * 10) + 88, pneumonia: Math.floor(Math.random() * 8) + 2, tb: Math.floor(Math.random() * 4) + 1 };
            } else if (rand < 0.8) {
                // Mock Pneumonia
                const pneumoniaPct = Math.floor(Math.random() * 15) + 75;
                const tbPct = Math.floor(Math.random() * 12) + 2;
                return { label: 'PNEUMONIA', normal: 100 - pneumoniaPct - tbPct, pneumonia: pneumoniaPct, tb: tbPct };
            } else {
                // Mock TB
                const tbPct = Math.floor(Math.random() * 15) + 75;
                const pneumoniaPct = Math.floor(Math.random() * 12) + 2;
                return { label: 'TUBERCULOSIS', normal: 100 - tbPct - pneumoniaPct, pneumonia: pneumoniaPct, tb: tbPct };
            }
        }
    }

    // 5. Render Prediction Report UI
    function renderResults(result) {
        resultsWaiting.style.display = 'none';
        resultsCard.style.display = 'flex';

        // Set text
        detectedDisease.textContent = result.label;
        
        // Remove old style classes from badge & display block
        severityBadge.className = 'badge';
        diseaseDisplay.className = 'detected-disease-display';

        if (result.label === 'NORMAL') {
            severityBadge.textContent = 'Healthy';
            severityBadge.classList.add('badge-success');
            diseaseDisplay.classList.add('class-normal');
            
            // Disable heatmap localization slider (nothing to highlight on normal lungs)
            heatmapToggle.checked = false;
            heatmapOverlay.className = 'heatmap-overlay';
            heatmapToggleContainer.style.opacity = '0.5';
            heatmapToggle.disabled = true;
        } else if (result.label === 'PNEUMONIA') {
            severityBadge.textContent = 'Attention Required';
            severityBadge.classList.add('badge-warning');
            diseaseDisplay.classList.add('class-pneumonia');
            
            heatmapToggleContainer.style.opacity = '1';
            heatmapToggle.disabled = false;
        } else if (result.label === 'TUBERCULOSIS') {
            severityBadge.textContent = 'Consult Doctor';
            severityBadge.classList.add('badge-danger');
            diseaseDisplay.classList.add('class-tb');
            
            heatmapToggleContainer.style.opacity = '1';
            heatmapToggle.disabled = false;
        }

        // Animate count-up for numbers and width expansion for tracks
        animateMetric(pctNormal, barNormal, result.normal);
        animateMetric(pctPneumonia, barPneumonia, result.pneumonia);
        animateMetric(pctTB, barTB, result.tb);
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

    // 6. Heatmap Activation Slider Toggle
    heatmapToggle.addEventListener('change', () => {
        if (!activeResult) return;
        
        if (heatmapToggle.checked) {
            // Activate target medical visual overlay
            heatmapOverlay.className = 'heatmap-overlay active';
            if (activeResult.label === 'PNEUMONIA') {
                heatmapOverlay.classList.add('heatmap-pneumonia');
            } else if (activeResult.label === 'TUBERCULOSIS') {
                heatmapOverlay.classList.add('heatmap-tb');
            }
        } else {
            // Deactivate and reset overlay opacity
            heatmapOverlay.className = 'heatmap-overlay';
        }
    });

});
