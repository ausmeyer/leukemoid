// --- Global Variables ---
let ortSession = null; // To hold the ONNX Runtime session
let plotData = null; // To hold data from lgbm_plot_data.json
let featureOrder = null; // To hold the exact order of features from lgb_feature_order.txt

// --- DOM Element References ---
const computeButton = document.getElementById('computeButton');
const predictionScoreDiv = document.getElementById('predictionScore');
const diagnosisResultDiv = document.getElementById('diagnosisResult');
const predictionPlotDiv = document.getElementById('predictionPlot');
const keyFindingsDiv = document.getElementById('keyFindings');

// List of all input fields
const inputFields = [
    'age', 'ldh', 'initial_wbc_50', 'initial_wbc_hosp', 'hgb', 'mcv', 'platelets',
    'neuts', 'bands', 'lymphs', 'monos', 'eos', 'baso'
];

// Reference ranges for parameters
const referenceRanges = {
    age: { min: 0, max: 120, normal_min: 0, normal_max: 120, unit: "years" },
    ldh: { min: 0, max: 1000, normal_min: 94, normal_max: 250, unit: "IU/L" },
    initial_wbc_50: { min: 50, max: 200, normal_min: 4, normal_max: 10, unit: "k/μL" },
    initial_wbc_hosp: { min: 0, max: 200, normal_min: 4, normal_max: 10, unit: "k/μL" },
    hgb: { min: 0, max: 20, normal_min: 11, normal_max: 16, unit: "g/dL" },
    mcv: { min: 60, max: 120, normal_min: 82, normal_max: 98, unit: "fL" },
    platelets: { min: 0, max: 1000, normal_min: 150, normal_max: 400, unit: "k/μL" },
    neuts: { min: 0, max: 100, normal_min: 34, normal_max: 71, unit: "%" },
    bands: { min: 0, max: 30, normal_min: 0, normal_max: 5, unit: "%" },
    lymphs: { min: 0, max: 100, normal_min: 19, normal_max: 53, unit: "%" },
    monos: { min: 0, max: 30, normal_min: 5, normal_max: 13, unit: "%" },
    eos: { min: 0, max: 20, normal_min: 1, normal_max: 7, unit: "%" },
    baso: { min: 0, max: 5, normal_min: 0, normal_max: 1, unit: "%" }
};

// Function to validate the WBC >50k field
function validateWbc50Field() {
    const wbc50Input = document.getElementById('initial_wbc_50');
    if (!wbc50Input) return true; // Field doesn't exist, no validation needed
    
    const value = parseFloat(wbc50Input.value);
    if (isNaN(value)) return false; // Not a number
    
    if (value < 50) {
        // Value is less than 50, show error
        wbc50Input.style.border = '2px solid var(--danger-color)';
        
        // Create or update error message
        let errorMsg = wbc50Input.parentNode.querySelector('.error-message');
        if (!errorMsg) {
            errorMsg = document.createElement('div');
            errorMsg.className = 'error-message';
            
            // Find the unit element to position the error message after it
            const unitElement = wbc50Input.parentNode.querySelector('.unit');
            if (unitElement) {
                unitElement.parentNode.insertBefore(errorMsg, unitElement.nextSibling);
            } else {
                wbc50Input.parentNode.appendChild(errorMsg);
            }
        }
        
        errorMsg.textContent = 'WBC count must be ≥ 50k for accurate results';
        errorMsg.style.color = 'var(--danger-color)';
        errorMsg.style.fontSize = '0.8rem';
        errorMsg.style.marginLeft = '15px'; // Add more space from the units
        errorMsg.style.display = 'inline-block'; // Make it inline-block
        errorMsg.style.verticalAlign = 'middle'; // Align vertically in the middle
        errorMsg.style.position = 'relative';
        errorMsg.style.top = '-1px'; // Fine-tune vertical alignment
        
        return false;
    } else {
        // Value is valid, clear any error
        wbc50Input.style.border = '';
        
        // Remove error message if it exists
        const errorMsg = wbc50Input.parentNode.querySelector('.error-message');
        if (errorMsg) {
            errorMsg.remove();
        }
        
        return true;
    }
}

// Function to set up WBC >50k field validation
function setupWbc50Validation() {
    const wbc50Input = document.getElementById('initial_wbc_50');
    if (!wbc50Input) return;
    
    // Remove any existing listeners to prevent duplicates
    const newInput = wbc50Input.cloneNode(true);
    wbc50Input.parentNode.replaceChild(newInput, wbc50Input);
    
    // Add the event listeners to the new element
    newInput.addEventListener('input', validateWbc50Field);
    newInput.addEventListener('change', validateWbc50Field);
    
    // Initial validation
    validateWbc50Field();
}

// Function to remove range slider from initial_wbc_hosp if it exists
function removeWbcRangeSlider() {
    const wbcInputGroup = document.getElementById('initial_wbc_hosp').closest('.input-group');
    if (wbcInputGroup) {
        const rangeSlider = wbcInputGroup.querySelector('.range-slider-container');
        if (rangeSlider) {
            rangeSlider.remove();
        }
    }
}

// Add CSS for error messages
function addErrorMessageStyles() {
    const styleElement = document.createElement('style');
    styleElement.textContent = `
        .error-message {
            color: var(--danger-color);
            font-size: 0.8rem;
            margin-left: 15px;
            display: inline-block;
            vertical-align: middle;
            animation: fadeIn 0.3s;
            line-height: 1.2;
            position: relative;
            top: -1px; /* Fine-tune vertical alignment */
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .input-field {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .unit {
            padding-left: 8px;
            color: var(--dark-gray);
            font-size: 0.85rem;
            white-space: nowrap;
        }
    `;
    document.head.appendChild(styleElement);
}

// Function to adjust the results layout
function adjustResultsLayout() {
    // Get the results summary container
    const resultSummary = document.querySelector('.result-summary');
    if (!resultSummary) return;
    
    // Remove the model confidence div if it exists
    const confidenceItem = document.getElementById('modelConfidence');
    if (confidenceItem) {
        const confidenceContainer = confidenceItem.closest('.result-item');
        if (confidenceContainer) {
            confidenceContainer.remove();
        }
    }
    
    // Adjust the layout of the remaining items to be side by side
    resultSummary.style.display = 'flex';
    resultSummary.style.flexDirection = 'row';
    resultSummary.style.justifyContent = 'space-between';
    resultSummary.style.alignItems = 'flex-start';
    resultSummary.style.gap = '40px'; // Add some space between columns
    
    // Make each item take up equal width
    const remainingItems = resultSummary.querySelectorAll('.result-item');
    remainingItems.forEach(item => {
        item.style.flex = '1 0 calc(50% - 20px)'; // Equal width with gap consideration
        item.style.maxWidth = 'calc(50% - 20px)';
        item.style.margin = '0'; // Reset any existing margins
    });
}

// Additional CSS adjustments for result items
function addResultItemStyles() {
    // Add a style element to the head
    const styleElement = document.createElement('style');
    styleElement.textContent = `
        .result-summary {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: flex-start;
            gap: 40px;
            margin-bottom: 30px;
        }
        
        .result-item {
            flex: 1 0 calc(50% - 20px);
            max-width: calc(50% - 20px);
            margin: 0;
            text-align: center;
        }
        
        .result-item-title {
            font-size: 1rem;
            margin-bottom: 10px;
        }
        
        .result-item-value {
            font-size: 1.4rem;
        }
    `;
    document.head.appendChild(styleElement);
}

// --- Initialization ---
async function initializeApp() {
    console.log("Initializing application...");
    try {
        // 1. Fetch the feature order
        console.log("Fetching feature order...");
        const featureResponse = await fetch('lgb_feature_order.txt');
        if (!featureResponse.ok) {
            throw new Error(`HTTP error! status: ${featureResponse.status} while fetching feature order.`);
        }
        const featureText = await featureResponse.text();
        featureOrder = featureText.split('\n').map(f => f.trim()).filter(f => f.length > 0);
        console.log(`Feature order loaded (${featureOrder.length} features):`, featureOrder);
        
        // 2. Fetch the plot data and cutoff
        console.log("Fetching plot data...");
        const plotDataResponse = await fetch('lgbm_plot_data.json');
        if (!plotDataResponse.ok) {
            throw new Error(`HTTP error! status: ${plotDataResponse.status} while fetching plot data.`);
        }
        plotData = await plotDataResponse.json();
        console.log("Plot data loaded:", plotData);
        if (!plotData || typeof plotData.cutoff === 'undefined' || !plotData.background_predictions || !plotData.background_diagnoses) {
             throw new Error("Plot data JSON is missing required fields (cutoff, background_predictions, background_diagnoses).");
        }
        console.log("Cutoff:", plotData.cutoff);

        // 3. Create the ONNX inference session - USE THE nozipmap MODEL FILE
        console.log("Creating ONNX session (using nozipmap model)...");
        // *** Make sure this filename matches the model created with zipmap=False ***
        ortSession = await ort.InferenceSession.create('./lightgbm_model_nozipmap.onnx', { executionProviders: ['wasm'] });
        console.log("ONNX session created successfully.");

        // 4. Enable the button
        computeButton.disabled = false;
        computeButton.textContent = 'Compute Prediction';
        console.log("Initialization complete. Ready for prediction.");

        // 5. Add range sliders to CBC and LDH fields
        addRangeSlidersToFields();

    } catch (error) {
        console.error("Initialization failed:", error);
        diagnosisResultDiv.textContent = `Initialization Error: ${error.message}. Please check file paths (using nozipmap model?) and network connection.`;
        diagnosisResultDiv.className = 'prediction-score score-high';
        computeButton.textContent = 'Initialization Failed';
        computeButton.disabled = true;
    }
}

// Function to add range sliders to CBC and LDH fields
function addRangeSlidersToFields() {
    // Fields that need range sliders - removed initial_wbc_hosp
    const cbcLdhFields = ['ldh', 'hgb', 'mcv', 'platelets'];
    
    cbcLdhFields.forEach(field => {
        const inputGroup = document.getElementById(field).closest('.input-group');
        if (!inputGroup) return;
        
        // Create range slider container
        const rangeSliderContainer = document.createElement('div');
        rangeSliderContainer.className = 'range-slider-container';
        
        // Create min value span
        const rangeMin = document.createElement('span');
        rangeMin.className = 'range-min';
        rangeMin.textContent = referenceRanges[field].min;
        
        // Create slider div
        const rangeSlider = document.createElement('div');
        rangeSlider.className = 'range-slider';
        
        // Create normal range indicator
        const rangeNormal = document.createElement('div');
        rangeNormal.className = 'range-normal';
        
        // Calculate normal range position as percentage
        const totalRange = referenceRanges[field].max - referenceRanges[field].min;
        const normalMinPercent = ((referenceRanges[field].normal_min - referenceRanges[field].min) / totalRange) * 100;
        const normalMaxPercent = ((referenceRanges[field].normal_max - referenceRanges[field].min) / totalRange) * 100;
        
        rangeNormal.style.left = `${normalMinPercent}%`;
        rangeNormal.style.right = `${100 - normalMaxPercent}%`;
        
        // Create marker
        const rangeMarker = document.createElement('div');
        rangeMarker.className = `range-marker ${field}-marker`;
        
        // Calculate initial marker position
        const inputValue = parseFloat(document.getElementById(field).value);
        if (!isNaN(inputValue)) {
            const percentPosition = ((inputValue - referenceRanges[field].min) / totalRange) * 100;
            rangeMarker.style.left = `${percentPosition}%`;
        }
        
        // Create max value span
        const rangeMax = document.createElement('span');
        rangeMax.className = 'range-max';
        rangeMax.textContent = referenceRanges[field].max;
        
        // Assemble the components
        rangeSlider.appendChild(rangeNormal);
        rangeSlider.appendChild(rangeMarker);
        
        rangeSliderContainer.appendChild(rangeMin);
        rangeSliderContainer.appendChild(rangeSlider);
        rangeSliderContainer.appendChild(rangeMax);
        
        // Add to DOM
        inputGroup.appendChild(rangeSliderContainer);
    });
}

// --- Prediction Handling (Expecting Simplified Tensor Output) ---
async function handleCompute() {
    console.log("Compute button clicked.");

    if (!ortSession || !plotData || !featureOrder) {
        console.error("App not initialized.");
        diagnosisResultDiv.textContent = "Error: Application not initialized.";
        diagnosisResultDiv.className = 'prediction-score score-high';
        return;
    }
    
    // Validate WBC >50k field before proceeding
    if (!validateWbc50Field()) {
        alert("Error: WBC count must be greater than or equal to 50k for accurate results.");
        return;
    }

    computeButton.disabled = true;
    computeButton.textContent = 'Computing...';
    predictionScoreDiv.textContent = '-';
    diagnosisResultDiv.textContent = '-';

    // Clear previous plot
    try {
        Plotly.purge(predictionPlotDiv);
    } catch (e) { 
        console.warn("Plotly purge failed (maybe no plot yet):", e);
    }
    predictionPlotDiv.innerHTML = '';

    try {
        // 1. Get input values
        const inputValues = [];
        let invalidInput = false;
        
        for (const featureName of featureOrder) {
            const inputElement = document.getElementById(featureName);
            if (!inputElement) throw new Error(`Input element missing: ${featureName}`);
            
            const value = parseFloat(inputElement.value);
            if (isNaN(value)) {
                console.error(`Invalid input for ${featureName}: '${inputElement.value}'`);
                inputElement.style.border = '2px solid var(--danger-color)';
                invalidInput = true;
            } else {
                inputElement.style.border = '';
                inputValues.push(value);
            }
        }
        
        if (invalidInput) throw new Error("Invalid input detected.");
        console.log("Input values collected:", inputValues);

        // 2. Prepare the ONNX tensor
        const inputTensor = new ort.Tensor('float32', new Float32Array(inputValues), [1, featureOrder.length]);
        console.log("Input tensor created:", inputTensor);

        // 3. Run inference
        console.log("Running inference...");
        const feeds = { 'float_input': inputTensor }; // Input name from conversion
        const results = await ortSession.run(feeds);
        console.log("Raw Inference Results:", results); // Log raw results

        // 4. Extract the prediction score (Expecting Simple Tensor named 'probabilities')
        let prediction = NaN;
        const probabilityOutputName = 'probabilities';

        if (!results[probabilityOutputName]) {
            let availableKeys = results ? Object.keys(results).join(', ') : 'undefined';
            throw new Error(`Output '${probabilityOutputName}' not found in results. Available: ${availableKeys}`);
        }

        const probabilitiesTensor = results[probabilityOutputName];
        console.log(`Value of results['${probabilityOutputName}']:`, probabilitiesTensor);

        if (!probabilitiesTensor || typeof probabilitiesTensor !== 'object' || !probabilitiesTensor.dims || !probabilitiesTensor.data) {
             throw new Error(`Output '${probabilityOutputName}' is not a valid ONNX Tensor object.`);
        }
        console.log(`Is Float32Array?`, probabilitiesTensor.data instanceof Float32Array);
        console.log(`Dims:`, probabilitiesTensor.dims);

        // Expecting shape [1, 2]
        if (!(probabilitiesTensor.data instanceof Float32Array) || probabilitiesTensor.dims.length !== 2 || probabilitiesTensor.dims[0] !== 1 || probabilitiesTensor.dims[1] !== 2) {
            console.error("Output structure mismatch. Expected tensor<float32>[1, 2]. Found Dims:", probabilitiesTensor.dims, "Data Type:", typeof probabilitiesTensor.data);
            throw new Error(`Output '${probabilityOutputName}' does not have the expected structure (tensor<float32>[1, 2]).`);
        }

        // Data is [prob_class_0, prob_class_1]. We want class 1 (index 1).
        prediction = probabilitiesTensor.data[1];
        console.log("Successfully extracted prediction score (Prob Class 1):", prediction);

        // 5. Display results
        if (isNaN(prediction)) {
            throw new Error("Prediction score extraction resulted in NaN.");
        }
        predictionScoreDiv.textContent = prediction.toFixed(3);
        
        // Determine color class based on cutoff
        const cutoff = plotData.cutoff;
        let colorClass;
        
        if (prediction >= cutoff) {
            // Above cutoff - Myeloid Malignancy (red)
            colorClass = 'prediction-score score-high';
            diagnosisResultDiv.textContent = "Myeloid Malignancy";
        } else {
            // Below cutoff - Leukemoid Reaction (green)
            colorClass = 'prediction-score score-low';
            diagnosisResultDiv.textContent = "Leukemoid Reaction";
        }
        
        // Apply the same color class to both elements
        predictionScoreDiv.className = colorClass;
        diagnosisResultDiv.className = colorClass;

        // Switch to results tab
        showTab('results-tab');

        // Generate plot
        generatePlot(prediction, plotData.background_predictions, plotData.background_diagnoses, plotData.cutoff);

    } catch (error) {
        console.error("Prediction failed:", error);
        predictionScoreDiv.textContent = "Error";
        diagnosisResultDiv.textContent = error.message;
        diagnosisResultDiv.className = 'prediction-score score-high';
    } finally {
        computeButton.disabled = false;
        computeButton.textContent = 'Compute Prediction';
    }
}

// Get list of abnormal values
function getAbnormalValues() {
    const abnormal = [];
    Object.keys(referenceRanges).forEach(param => {
        const input = document.getElementById(param);
        if (!input) return;
        
        const value = parseFloat(input.value);
        if (isNaN(value)) return;
        
        const range = referenceRanges[param];
        
        if (value < range.normal_min || value > range.normal_max) {
            abnormal.push({
                name: param,
                value: value,
                unit: range.unit,
                normal_min: range.normal_min,
                normal_max: range.normal_max
            });
        }
    });
    return abnormal;
}

// --- Plotting Function ---
function generatePlot(oosPredicrion, backgroundPredictions, backgroundDiagnoses, cutoff) {
    // --- Data processing ---
    let plotPoints = backgroundPredictions.map((pred, index) => ({
        prediction: pred,
        diagnosis: backgroundDiagnoses[index],
        type: backgroundDiagnoses[index] === 1 ? 'Myeloid Malignancy' : 'Leukemoid Reaction'
    }));
    
    plotPoints.push({ 
        prediction: oosPredicrion, 
        diagnosis: null, 
        type: 'Prediction' 
    });
    
    plotPoints.sort((a, b) => a.prediction - b.prediction);
    plotPoints.forEach((point, index) => { point.x = index + 1; });

    const traceLR = { 
        x: [], 
        y: [], 
        type: 'scatter', 
        mode: 'markers', 
        name: 'Leukemoid Reaction', 
        marker: { 
            color: '#27ae60', 
            size: 8, 
            opacity: 0.7 
        } 
    };
    
    const traceMM = { 
        x: [], 
        y: [], 
        type: 'scatter', 
        mode: 'markers', 
        name: 'Myeloid Malignancy', 
        marker: { 
            color: '#e74c3c', 
            size: 8, 
            opacity: 0.7 
        } 
    };
    
    const tracePred = { 
        x: [], 
        y: [], 
        type: 'scatter', 
        mode: 'markers', 
        name: 'Your Patient', 
        marker: { 
            color: '#3498db', 
            size: 14, 
            symbol: 'diamond', 
            opacity: 1.0,
            line: {
                color: '#2c3e50',
                width: 2
            }
        } 
    };
    
    // Create annotation with position based on prediction value
    const annotation = { 
        x: null, 
        y: null, 
        text: 'Your Patient', 
        showarrow: true, 
        arrowhead: 7,
        ax: 0,  // Center the annotation on the x-axis of the point
        font: {
            size: 14,
            color: '#2c3e50'
        },
        bgcolor: 'rgba(255, 255, 255, 0.8)',
        bordercolor: '#3498db',
        borderwidth: 1,
        borderpad: 4,
        xanchor: 'center'  // Center the annotation text on the x-coordinate
    };

    plotPoints.forEach(point => {
        if (point.type === 'Leukemoid Reaction') { 
            traceLR.x.push(point.x); 
            traceLR.y.push(point.prediction); 
        }
        else if (point.type === 'Myeloid Malignancy') { 
            traceMM.x.push(point.x); 
            traceMM.y.push(point.prediction); 
        }
        else if (point.type === 'Prediction') {
            tracePred.x.push(point.x); 
            tracePred.y.push(point.prediction);
            annotation.x = point.x; 
            annotation.y = point.prediction;
            
            // Position annotation above or below the point based on y-value
            if (point.prediction > 0.5) {
                annotation.ay = 40;  // Below the point
                annotation.yanchor = 'top';
            } else {
                annotation.ay = -40; // Above the point
                annotation.yanchor = 'bottom';
            }
        }
    });

    // --- Layout definition with Horizontal Top-Right Legend ---
    const plotLayout = {
        title: {
            text: 'Prediction Visualization',
            font: {
                family: 'Roboto, sans-serif',
                size: 20,
                color: '#2c3e50'
            },
            x: 0.5,
            xanchor: 'center'
        },
        xaxis: { 
            title: {
                text: 'Patients Ordered by Model Score',
                font: {
                    family: 'Roboto, sans-serif',
                    size: 16,
                    color: '#2c3e50'
                }
            },
            zeroline: false,
            gridcolor: '#e9ecef',
            tickfont: {
                family: 'Roboto, sans-serif',
                size: 14
            }
        },
        yaxis: { 
            title: {
                text: 'Probability of Malignancy',
                font: {
                    family: 'Roboto, sans-serif',
                    size: 16,
                    color: '#2c3e50'
                }
            },
            range: [-0.05, 1.05], 
            zeroline: false,
            gridcolor: '#e9ecef',
            tickfont: {
                family: 'Roboto, sans-serif',
                size: 14
            }
        },
        hovermode: 'closest',
        showlegend: true,
        legend: {
            orientation: "h",
            yanchor: "top",
            y: 1.08,
            xanchor: "center",
            x: 0.5,
            bgcolor: 'rgba(255,255,255,0)',
            bordercolor: 'rgba(0,0,0,0)',
            borderwidth: 0,
            font: {
                family: 'Roboto, sans-serif',
                size: 14
            }
        },
        height: 550,
        margin: { t: 80, b: 80, l: 80, r: 40 },
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        shapes: [
            { // Cutoff line
                type: 'line',
                x0: 0,
                y0: cutoff,
                x1: (plotPoints.length || 1) + 1,
                y1: cutoff,
                line: { color: '#f39c12', width: 2, dash: 'dash' }
            }
        ],
        annotations: [
            { // Cutoff label
                x: 0.05,
                y: cutoff,
                xref: 'paper',
                yref: 'y',
                text: 'Diagnostic Cutoff',
                showarrow: false,
                yanchor: 'bottom',
                xanchor: 'left',
                font: { 
                    size: 14,
                    color: '#f39c12', 
                    family: 'Roboto, sans-serif' 
                }
            },
            // Add the annotation for 'Your Patient' ONLY if coordinates were found
            (annotation.x !== null && annotation.y !== null) ? annotation : {}
        ]
    };

    const plotTraces = [traceLR, traceMM, tracePred];
    Plotly.newPlot(predictionPlotDiv, plotTraces, plotLayout, {responsive: true});
}

// --- Tab switching function ---
function showTab(tabId) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all tab buttons
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.classList.remove('active');
    });
    
    // Show the selected tab content
    document.getElementById(tabId).classList.add('active');
    
    // Add active class to the clicked button
    const activeButton = document.querySelector(`.tab-button[onclick="showTab('${tabId}')"]`);
    if (activeButton) {
        activeButton.classList.add('active');
    }

    // If results tab is selected and there's a prediction, resize the plot
    if (tabId === 'results-tab' && document.getElementById('predictionPlot').innerHTML) {
        setTimeout(() => {
            if (typeof Plotly !== 'undefined') {
                Plotly.Plots.resize(document.getElementById('predictionPlot'));
            }
        }, 100);
    }
}

// --- Update range markers ---
function updateRangeMarkers() {
    // Update all parameters with range markers - removed initial_wbc_hosp
    const parameters = [
        'neuts', 'bands', 'lymphs', 'monos', 'eos', 'baso',
        'ldh', 'hgb', 'mcv', 'platelets'
    ];
    
    parameters.forEach(param => {
        const input = document.getElementById(param);
        if (!input) return;
        
        const value = parseFloat(input.value);
        if (isNaN(value)) return;
        
        const range = referenceRanges[param];
        const total = range.max - range.min;
        const percentPosition = ((value - range.min) / total) * 100;
        
        const marker = document.querySelector(`.${param}-marker`);
        if (marker) {
            marker.style.left = `${percentPosition}%`;
        }
        
        // Mark abnormal values
        if (value < range.normal_min || value > range.normal_max) {
            input.classList.add('abnormal');
        } else {
            input.classList.remove('abnormal');
        }
    });
}

// Function to check and mark abnormal values
function checkAbnormalValues() {
    Object.keys(referenceRanges).forEach(param => {
        const input = document.getElementById(param);
        if (!input) return;
        
        const value = parseFloat(input.value);
        if (isNaN(value)) return;
        
        const range = referenceRanges[param];
        
        if (value < range.normal_min || value > range.normal_max) {
            input.classList.add('abnormal');
        } else {
            input.classList.remove('abnormal');
        }
    });
}

// Add this function to ensure all event listeners are properly set up
function setupInputListeners() {
    // Add input event listeners to all parameters with range markers - removed initial_wbc_hosp
    const parameters = [
        'neuts', 'bands', 'lymphs', 'monos', 'eos', 'baso',
        'ldh', 'hgb', 'mcv', 'platelets'
    ];
    
    parameters.forEach(param => {
        const input = document.getElementById(param);
        if (input) {
            // Remove any existing listeners to prevent duplicates
            const newInput = input.cloneNode(true);
            input.parentNode.replaceChild(newInput, input);
            
            // Add the event listener to the new element
            newInput.addEventListener('input', function() {
                updateRangeMarkers();
                checkAbnormalValues();
            });
            
            // Also add change event for when users use arrow keys or blur
            newInput.addEventListener('change', function() {
                updateRangeMarkers();
                checkAbnormalValues();
            });
        }
    });
    
    // Still need to add event listener for initial_wbc_hosp for abnormal value highlighting
    const wbcInput = document.getElementById('initial_wbc_hosp');
    if (wbcInput) {
        const newWbcInput = wbcInput.cloneNode(true);
        wbcInput.parentNode.replaceChild(newWbcInput, wbcInput);
        
        newWbcInput.addEventListener('input', function() {
            checkAbnormalValues();
        });
        
        newWbcInput.addEventListener('change', function() {
            checkAbnormalValues();
        });
    }
}

// Debounce function for resize
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Resize handler for plots
const debouncedResizeHandler = debounce(() => {
    if (predictionPlotDiv && typeof Plotly !== 'undefined') {
        console.log("Window resized, calling Plotly.Plots.resize...");
        try {
            Plotly.Plots.resize(predictionPlotDiv);
        } catch (error) {
            console.error("Error during Plotly resize:", error);
        }
    }
}, 250);

window.addEventListener('resize', debouncedResizeHandler);

document.addEventListener('DOMContentLoaded', function() {
    // Add event listener to compute button
    computeButton.addEventListener('click', handleCompute);
    
    // Set up input listeners properly
    setupInputListeners();
    
    // Set up WBC >50k validation
    setupWbc50Validation();
    
    // Initialize range markers
    updateRangeMarkers();
    
    // Remove WBC range slider if it exists
    removeWbcRangeSlider();
    
    // Adjust the results layout
    adjustResultsLayout();
    
    // Add additional styles for result items
    addResultItemStyles();
    
    // Add error message styles
    addErrorMessageStyles();
    
    // Start app initialization
    initializeApp();
});