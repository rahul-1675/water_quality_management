document.addEventListener('DOMContentLoaded', () => {
    // Single Prediction Form
    const predictionForm = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const gaugeFill = document.getElementById('gauge-fill');
    const wqiValue = document.getElementById('wqi-value');
    const wqiAdvice = document.getElementById('wqi-advice');

    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(predictionForm);
        const data = Object.fromEntries(formData.entries());
        
        // Show loading state
        const submitBtn = predictionForm.querySelector('button');
        const originalText = submitBtn.textContent;
        submitBtn.textContent = 'Predicting...';
        submitBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            
            if (response.ok) {
                displayResult(result.wqi, result.advice);
            } else {
                alert(`Error: ${result.error}`);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while making the prediction.');
        } finally {
            submitBtn.textContent = originalText;
            submitBtn.disabled = false;
        }
    });

    function displayResult(wqi, advice) {
        resultContainer.classList.remove('hidden');
        
        // Animate counter
        const target = Math.round(wqi * 10) / 10;
        let current = 0;
        const inc = target / 50; // 50 frames
        
        const interval = setInterval(() => {
            current += inc;
            if (current >= target) {
                current = target;
                clearInterval(interval);
            }
            wqiValue.textContent = current.toFixed(1);
        }, 30);
        
        wqiAdvice.textContent = advice;
        
        // Color coding and gauge update
        let color = 'var(--success)';
        if (wqi < 50) color = 'var(--danger)';
        else if (wqi < 70) color = 'var(--warning)';
        
        // gauge max is 100, dasharray is 125.6 (Pi * r where r=40)
        // dashoffset goes from 125.6 (empty) to 0 (full)
        const percentage = Math.min(Math.max(wqi, 0), 100) / 100;
        const dashOffset = 125.6 - (percentage * 125.6);
        
        gaugeFill.style.stroke = color;
        wqiValue.style.color = color;
        setTimeout(() => {
            gaugeFill.style.strokeDashoffset = dashOffset;
        }, 100);
    }

    // Batch Prediction
    const fileInput = document.getElementById('csv-file');
    const fileNameDisplay = document.getElementById('file-name');
    const batchBtn = document.getElementById('batch-btn');
    const uploadArea = document.getElementById('upload-area');

    // Drag and drop handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.remove('dragover');
        }, false);
    });

    uploadArea.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        let files = dt.files;
        fileInput.files = files;
        updateFileName();
    });

    fileInput.addEventListener('change', updateFileName);

    function updateFileName() {
        if (fileInput.files.length > 0) {
            fileNameDisplay.textContent = fileInput.files[0].name;
            batchBtn.disabled = false;
        } else {
            fileNameDisplay.textContent = 'Click to browse or drag and drop';
            batchBtn.disabled = true;
        }
    }

    batchBtn.addEventListener('click', async () => {
        if (fileInput.files.length === 0) return;
        
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);
        
        const originalText = batchBtn.textContent;
        batchBtn.textContent = 'Processing...';
        batchBtn.disabled = true;

        try {
            const response = await fetch('/predict_batch', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                // Handle file download
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'predictions_' + file.name;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                
                alert('Batch processing complete! Downloading results.');
            } else {
                const result = await response.json();
                alert(`Error: ${result.error}`);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during batch processing.');
        } finally {
            batchBtn.textContent = originalText;
            batchBtn.disabled = false;
            fileInput.value = '';
            updateFileName();
        }
    });

    // Load Metrics Chart
    fetch('/metrics')
        .then(response => response.json())
        .then(data => {
            const results = data.results;
            const models = Object.keys(results);
            const mse = models.map(m => results[m].MSE);
            const r2 = models.map(m => results[m].R2);

            const ctx = document.getElementById('metricsChart').getContext('2d');
            
            // Set Chart.js defaults for dark theme
            Chart.defaults.color = 'rgba(255, 255, 255, 0.7)';
            Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
            
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: models,
                    datasets: [
                        {
                            label: 'R² Score (Higher is better)',
                            data: r2,
                            backgroundColor: 'rgba(0, 210, 255, 0.6)',
                            borderColor: 'rgba(0, 210, 255, 1)',
                            borderWidth: 1,
                            yAxisID: 'y'
                        },
                        {
                            label: 'MSE (Lower is better)',
                            data: mse,
                            backgroundColor: 'rgba(255, 51, 102, 0.6)',
                            borderColor: 'rgba(255, 51, 102, 1)',
                            borderWidth: 1,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'R² Score'
                            },
                            min: 0,
                            max: 1
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Mean Squared Error'
                            },
                            grid: {
                                drawOnChartArea: false,
                            },
                        },
                    },
                    plugins: {
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: '#fff',
                            bodyColor: '#fff',
                            padding: 10,
                            cornerRadius: 4
                        }
                    }
                }
            });
        })
        .catch(error => console.error('Error loading metrics:', error));
});
