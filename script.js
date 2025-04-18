// Myeloid Malignancy Predictor - script.js
// VERSION FOR ONNX MODEL CONVERTED WITH zipmap=False (e.g., lightgbm_model_nozipmap.onnx)
// Includes Horizontal Top Legend and Plot Resizing

// --- Global Variables ---
let ortSession = null; // To hold the ONNX Runtime session
let plotData = null; // To hold data from lgbm_plot_data.json
let featureOrder = null; // To hold the exact order of features from lgb_feature_order.txt

// --- DOM Element References ---
const computeButton = document.getElementById('computeButton');
const predictionScoreDiv = document.getElementById('predictionScore');
const diagnosisResultDiv = document.getElementById('diagnosisResult');
const predictionPlotDiv = document.getElementById('predictionPlot'); // For plotting
const inputSection = document.querySelector('.card'); // Updated selector

// --- Initialization ---
// Disable button until resources are loaded
computeButton.disabled = true;
computeButton.textContent = 'Loading Model...';

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
        if (featureOrder.length !== 13) {
            console.warn("Warning: Expected 13 features, but found", featureOrder.length);
        }

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

    } catch (error) {
        console.error("Initialization failed:", error);
        diagnosisResultDiv.textContent = `Initialization Error: ${error.message}. Please check file paths (using nozipmap model?) and network connection.`;
        diagnosisResultDiv.className = 'diagnosis-output';
        diagnosisResultDiv.style.backgroundColor = 'red';
        computeButton.textContent = 'Initialization Failed';
        computeButton.disabled = true;
    }
}

// --- Prediction Handling (Expecting Simplified Tensor Output) ---
async function handleCompute() {
    console.log("Compute button clicked.");

    if (!ortSession || !plotData || !featureOrder) {
        console.error("App not initialized.");
        diagnosisResultDiv.textContent = "Error: Application not initialized.";
        diagnosisResultDiv.className = 'diagnosis-output';
        diagnosisResultDiv.style.backgroundColor = 'red';
        return;
    }

    computeButton.disabled = true;
    computeButton.textContent = 'Computing...';
    predictionScoreDiv.textContent = 'Prediction Score: -';
    diagnosisResultDiv.textContent = 'Diagnosis: -';
    diagnosisResultDiv.className = 'diagnosis-output';
    diagnosisResultDiv.style.backgroundColor = '#6c757d';

    // Clear previous plot
    try {
        Plotly.purge(predictionPlotDiv); // Clear plotly plot if exists
    } catch (e) { console.warn("Plotly purge failed (maybe no plot yet):", e)}
    predictionPlotDiv.innerHTML = ''; // Clear any other content


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
        if (inputValues.length !== featureOrder.length) throw new Error("Input count mismatch.");
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
        predictionScoreDiv.textContent = `Prediction Score: ${prediction.toFixed(4)}`;

        const cutoff = plotData.cutoff;
        if (prediction >= cutoff) {
            diagnosisResultDiv.textContent = "Diagnosis: Myeloid Malignancy";
            diagnosisResultDiv.className = 'diagnosis-output diagnosis-mm';
        } else {
            diagnosisResultDiv.textContent = "Diagnosis: Leukemoid Reaction";
            diagnosisResultDiv.className = 'diagnosis-output diagnosis-lr';
        }
        diagnosisResultDiv.style.backgroundColor = '';

        // --- Call Plotting Function ---
        console.log("Generating plot...");
        generatePlot(prediction, plotData.background_predictions, plotData.background_diagnoses, plotData.cutoff);
        console.log("Plot generation initiated.");


    } catch (error) {
        console.error("Prediction failed:", error);
        diagnosisResultDiv.textContent = `Prediction Error: ${error.message}`;
        diagnosisResultDiv.className = 'diagnosis-output';
        diagnosisResultDiv.style.backgroundColor = 'red';
    } finally {
        computeButton.disabled = false;
        computeButton.textContent = 'Compute Prediction';
    }
}

// --- Plotting Function ---
function generatePlot(oosPrediction, backgroundPredictions, backgroundDiagnoses, cutoff) {
    console.log("Plotting inputs:", {oosPrediction, numBgPreds: backgroundPredictions.length, numBgDiags: backgroundDiagnoses.length, cutoff});

    // --- Data processing ---
    let plotPoints = backgroundPredictions.map((pred, index) => ({
        prediction: pred,
        diagnosis: backgroundDiagnoses[index],
        type: backgroundDiagnoses[index] === 1 ? 'Myeloid Malignancy' : 'Leukemoid Reaction'
    }));
    plotPoints.push({ prediction: oosPrediction, diagnosis: null, type: 'Prediction' });
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
        if (point.type === 'Leukemoid Reaction') { traceLR.x.push(point.x); traceLR.y.push(point.prediction); }
        else if (point.type === 'Myeloid Malignancy') { traceMM.x.push(point.x); traceMM.y.push(point.prediction); }
        else if (point.type === 'Prediction') {
             tracePred.x.push(point.x); tracePred.y.push(point.prediction);
             annotation.x = point.x; annotation.y = point.prediction;
             
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
    // --- End Data processing ---


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
            y: 1.08,  // Moved down slightly from 1.12
            xanchor: "center",
            x: 0.5,
            bgcolor: 'rgba(255,255,255,0)',  // Transparent background
            bordercolor: 'rgba(0,0,0,0)',    // No border
            borderwidth: 0,                  // No border width
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
    // --- End plotLayout definition ---

    const plotTraces = [traceLR, traceMM, tracePred];
    Plotly.newPlot(predictionPlotDiv, plotTraces, plotLayout, {responsive: true});

} // --- End generatePlot function ---


// --- Debounce Utility Function ---
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
};

// --- Plot Resizing Logic ---
const plotContainer = document.getElementById('predictionPlot');

const debouncedResizeHandler = debounce(() => {
    if (plotContainer && typeof Plotly !== 'undefined') {
         console.log("Window resized, calling Plotly.Plots.resize...");
         try {
             Plotly.Plots.resize(plotContainer);
         } catch (error) {
             console.error("Error during Plotly resize:", error);
         }
    }
}, 250);

window.addEventListener('resize', debouncedResizeHandler);

// --- Attach Event Listener for Compute Button ---
computeButton.addEventListener('click', handleCompute);

// --- Start Initialization ---
initializeApp();