let currentFile = null;

function previewImage(event) {
    const file = event.target.files[0];
    if (!file) return;

    currentFile = file;
    document.getElementById('fileMeta').textContent = `${file.name.substring(0, 16)}... (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;

    const reader = new FileReader();
    reader.onload = function(e) {
        const preview = document.getElementById('imagePreview');
        preview.src = e.target.result;
        document.getElementById('dropPrompt').classList.add('hidden');
        document.getElementById('previewContainer').classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

function resetForm() {
    currentFile = null;
    document.getElementById('imageInput').value = '';
    document.getElementById('fileMeta').textContent = 'Awaiting specimen';
    document.getElementById('dropPrompt').classList.remove('hidden');
    document.getElementById('previewContainer').classList.add('hidden');
    document.getElementById('scannerOverlay').classList.add('hidden');

    document.getElementById('idleState').classList.remove('hidden');
    document.getElementById('activeResultState').classList.add('hidden');
    document.getElementById('inferenceLatency').textContent = 'Latency: -- ms';
}

async function analyzeImage() {
    if (!currentFile) {
        alert("Please upload a facial portrait first!");
        return;
    }

    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnSpinner = document.getElementById('btnSpinner');
    const btnText = document.getElementById('btnText');
    const scanner = document.getElementById('scannerOverlay');

    // Activate UI Loading States
    analyzeBtn.disabled = true;
    btnSpinner.classList.remove('hidden');
    btnText.textContent = "Processing Tensor...";
    scanner.classList.remove('hidden');

    const startTime = performance.now();
    const formData = new FormData();
    formData.append("file", currentFile);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        const latency = Math.round(performance.now() - startTime);
        document.getElementById('inferenceLatency').textContent = `Latency: ${latency} ms`;

        displayResults(data);

    } catch (err) {
        console.error(err);
        alert("Inference request failed. Please verify that your FastAPI backend is running on http://127.0.0.1:8000.");
    } finally {
        analyzeBtn.disabled = false;
        btnSpinner.classList.add('hidden');
        btnText.textContent = "Execute Inference";
        scanner.classList.add('hidden');
    }
}

function displayResults(data) {
    const isReal = (data.prediction.toLowerCase() === "real");
    const confidence = parseFloat(data.confidence);

    document.getElementById('idleState').classList.add('hidden');
    document.getElementById('activeResultState').classList.remove('hidden');

    const verdictBadge = document.getElementById('verdictBadge');
    const verdictTitle = document.getElementById('verdictTitle');
    const verdictIcon = document.getElementById('verdictIcon');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceValue = document.getElementById('confidenceValue');
    const probReal = document.getElementById('probReal');
    const probFake = document.getElementById('probFake');

    if (isReal) {
        verdictBadge.className = "p-4 rounded-xl border border-emerald-500/50 bg-emerald-950/30 text-emerald-300 flex items-center justify-between";
        verdictTitle.textContent = "Authentic Portrait (Real)";
        verdictIcon.className = "w-10 h-10 rounded-full flex items-center justify-center font-bold text-lg bg-emerald-500/20 text-emerald-400 border border-emerald-500/40";
        verdictIcon.textContent = "✓";
        
        confidenceBar.className = "h-full rounded-full transition-all duration-700 ease-out bg-emerald-400 shadow-[0_0_12px_#34d399]";
        
        probReal.textContent = `${confidence.toFixed(2)}%`;
        probFake.textContent = `${(100 - confidence).toFixed(2)}%`;
    } else {
        verdictBadge.className = "p-4 rounded-xl border border-rose-500/50 bg-rose-950/30 text-rose-300 flex items-center justify-between";
        verdictTitle.textContent = "Synthetic Face (Fake/AI)";
        verdictIcon.className = "w-10 h-10 rounded-full flex items-center justify-center font-bold text-lg bg-rose-500/20 text-rose-400 border border-rose-500/40";
        verdictIcon.textContent = "⚠";

        confidenceBar.className = "h-full rounded-full transition-all duration-700 ease-out bg-rose-500 shadow-[0_0_12px_#f43f5e]";

        probFake.textContent = `${confidence.toFixed(2)}%`;
        probReal.textContent = `${(100 - confidence).toFixed(2)}%`;
    }

    confidenceValue.textContent = `${confidence.toFixed(2)}%`;
    
    // Animate bar fill
    confidenceBar.style.width = '0%';
    setTimeout(() => {
        confidenceBar.style.width = `${confidence}%`;
    }, 50);
}
