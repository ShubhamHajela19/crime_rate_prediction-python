const statusSummary = document.getElementById("statusSummary");
const statusJson = document.getElementById("statusJson");
const trainSummary = document.getElementById("trainSummary");
const trainStatus = document.getElementById("trainStatus");
const resultBox = document.getElementById("resultBox");
const resultCards = document.getElementById("resultCards");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const predictionJson = document.getElementById("predictionJson");
const probabilityText = document.getElementById("probabilityText");
const probabilityFill = document.getElementById("probabilityFill");
const trainBtn = document.getElementById("trainBtn");
const healthBtn = document.getElementById("healthBtn");
const predictBtn = document.getElementById("predictBtn");
const predictForm = document.getElementById("predictForm");
const datasetPathInput = document.getElementById("datasetPathInput");
const maxRowsInput = document.getElementById("maxRowsInput");
const statusLoader = document.getElementById("statusLoader");
const trainLoader = document.getElementById("trainLoader");

const heroServiceStatus = document.getElementById("heroServiceStatus");
const heroModelVersion = document.getElementById("heroModelVersion");
const heroModelsReady = document.getElementById("heroModelsReady");
const heroDatasetState = document.getElementById("heroDatasetState");
const heroSelectedK = document.getElementById("heroSelectedK");
const themeToggle = document.getElementById("themeToggle");
const themeToggleLabel = document.getElementById("themeToggleLabel");

let crimeDistributionChart;
let clusterChart;
let currentOverview = null;

const THEME_STORAGE_KEY = "crime-dashboard-theme";


function getSavedTheme() {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    if (savedTheme === "dark" || savedTheme === "light") {
        return savedTheme;
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}


function updateThemeToggle(theme) {
    if (!themeToggle || !themeToggleLabel) return;
    const isDark = theme === "dark";
    themeToggle.setAttribute("aria-pressed", String(isDark));
    themeToggle.setAttribute(
        "aria-label",
        isDark ? "Switch to light mode" : "Switch to dark mode"
    );
    themeToggleLabel.textContent = isDark ? "Light Mode" : "Dark Mode";
}


function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_STORAGE_KEY, theme);
    updateThemeToggle(theme);
    if (currentOverview) {
        updateCharts(currentOverview);
    }
}


function toggleTheme() {
    const nextTheme =
        document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(nextTheme);
}


function getThemePalette() {
    const styles = getComputedStyle(document.documentElement);
    return {
        text: styles.getPropertyValue("--text").trim() || "#1e2933",
        muted: styles.getPropertyValue("--muted").trim() || "#61707a",
        border: styles.getPropertyValue("--chart-grid").trim() || "rgba(35, 57, 74, 0.1)",
        primary: styles.getPropertyValue("--primary").trim() || "#0e7490",
        accent: styles.getPropertyValue("--accent").trim() || "#d97706",
        success: styles.getPropertyValue("--success").trim() || "#15803d",
        danger: styles.getPropertyValue("--danger").trim() || "#dc2626",
    };
}


function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}


function setButtonLoading(button, isLoading, defaultText, loadingText) {
    if (!button) return;
    button.disabled = isLoading;
    button.classList.toggle("is-loading", isLoading);
    button.textContent = isLoading ? loadingText : defaultText;
}


function toggleLoader(loader, isVisible) {
    if (!loader) return;
    loader.classList.toggle("hidden", !isVisible);
}


async function safeJsonParse(response) {
    const text = await response.text();
    if (!text) return {};
    try {
        return JSON.parse(text);
    } catch (error) {
        return { status: "error", message: text };
    }
}


function formatJson(data) {
    return JSON.stringify(data, null, 2);
}


function formatPercent(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "N/A";
    }
    return `${(Number(value) * 100).toFixed(1)}%`;
}


function titleCase(value) {
    return String(value || "")
        .split(/[\s_-]+/)
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
        .join(" ");
}


function renderSummaryCards(container, cards) {
    if (!container) return;
    container.innerHTML = cards.map((card) => `
        <article class="summary-card">
            <span class="label">${escapeHtml(card.label)}</span>
            <span class="value ${card.compact ? "small" : ""}">${escapeHtml(card.value)}</span>
        </article>
    `).join("");
}


function normalizeOverview(payload) {
    const source = payload?.details || payload?.model_overview || payload || {};
    const kmeansSummary = source.kmeans_summary || {};
    return {
        modelVersion: source.model_version || "Not trained",
        trainedAt: source.trained_at || "N/A",
        rows: source.rows_used_for_training || source.rows_used || 0,
        metrics: source.random_forest_metrics || source.metrics || {},
        selectedK: kmeansSummary.selected_k ?? source.selected_k ?? "-",
        zoneDistribution: kmeansSummary.zone_distribution || source.zone_distribution || {},
        clusterDistribution: kmeansSummary.cluster_distribution || source.cluster_distribution || {},
        featureImportance: source.feature_importance || [],
    };
}


function updateHero(statusPayload) {
    const overview = normalizeOverview(statusPayload.model_overview || {});
    const modelsReady = Boolean(statusPayload.models_ready);
    const datasetExists = Boolean(statusPayload.dataset?.exists);

    heroServiceStatus.textContent = modelsReady ? "Online and Ready" : "Online, Waiting for Model";
    heroModelVersion.textContent = `Latest model version: ${overview.modelVersion || "Not trained"}`;
    heroModelsReady.textContent = modelsReady ? "Yes" : "No";
    heroDatasetState.textContent = datasetExists ? "Available" : "Missing";
    heroSelectedK.textContent = overview.selectedK ?? "-";
}


function updateCharts(summary) {
    if (typeof Chart === "undefined") {
        return;
    }
    currentOverview = summary;

    const palette = getThemePalette();

    const zoneDistribution = summary.zoneDistribution || {};
    const clusterDistribution = summary.clusterDistribution || {};

    const zoneLabels = ["High", "Medium", "Low"];
    const zoneValues = [
        Number(zoneDistribution.high || 0),
        Number(zoneDistribution.medium || 0),
        Number(zoneDistribution.low || 0),
    ];

    const clusterLabels = Object.keys(clusterDistribution).map((clusterId) => `Cluster ${clusterId}`);
    const clusterValues = Object.values(clusterDistribution).map((value) => Number(value || 0));

    if (crimeDistributionChart) {
        crimeDistributionChart.destroy();
    }
    if (clusterChart) {
        clusterChart.destroy();
    }

    crimeDistributionChart = new Chart(document.getElementById("crimeDistributionChart"), {
        type: "doughnut",
        data: {
            labels: zoneLabels,
            datasets: [{
                data: zoneValues,
                backgroundColor: [palette.danger, palette.accent, palette.success],
                borderWidth: 0,
            }],
        },
        options: {
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        color: palette.text,
                    },
                },
            },
        },
    });

    clusterChart = new Chart(document.getElementById("clusterChart"), {
        type: "bar",
        data: {
            labels: clusterLabels.length ? clusterLabels : ["No Clusters"],
            datasets: [{
                label: "Records",
                data: clusterValues.length ? clusterValues : [0],
                backgroundColor: [
                    palette.primary,
                    "#0ea5b9",
                    "#14b8a6",
                    palette.accent,
                    "#fb7185",
                    "#8b5cf6",
                ],
                borderRadius: 12,
            }],
        },
        options: {
            maintainAspectRatio: false,
            scales: {
                x: {
                    ticks: {
                        color: palette.muted,
                    },
                    grid: {
                        color: palette.border,
                    },
                },
                y: {
                    beginAtZero: true,
                    ticks: {
                        precision: 0,
                        color: palette.muted,
                    },
                    grid: {
                        color: palette.border,
                    },
                },
            },
            plugins: {
                legend: {
                    display: false,
                },
            },
        },
    });
}


function renderStatus(statusPayload) {
    const overview = normalizeOverview(statusPayload.model_overview || {});
    const versionCount = Array.isArray(statusPayload.available_versions)
        ? statusPayload.available_versions.length
        : 0;

    renderSummaryCards(statusSummary, [
        { label: "API Status", value: statusPayload.status || "ok" },
        { label: "Models Ready", value: statusPayload.models_ready ? "Yes" : "No" },
        { label: "Dataset", value: statusPayload.dataset?.exists ? "Available" : "Missing" },
        { label: "Model Version", value: overview.modelVersion, compact: true },
        { label: "Saved Versions", value: versionCount },
        { label: "Selected K", value: overview.selectedK },
    ]);

    statusJson.textContent = formatJson(statusPayload);
    if (statusPayload.dataset?.path && datasetPathInput && !datasetPathInput.value.trim()) {
        datasetPathInput.value = statusPayload.dataset.path;
    }

    updateHero(statusPayload);
    updateCharts(overview);
}


function renderTraining(trainingPayload) {
    const overview = normalizeOverview(trainingPayload);
    const metrics = overview.metrics || {};
    const topFeature = overview.featureImportance?.[0]?.feature || "Not available";

    renderSummaryCards(trainSummary, [
        { label: "Model Version", value: overview.modelVersion, compact: true },
        { label: "Rows Used", value: overview.rows || 0 },
        { label: "Accuracy", value: formatPercent(metrics.accuracy) },
        { label: "F1 Score", value: formatPercent(metrics.f1_weighted) },
        { label: "Selected K", value: overview.selectedK },
        { label: "Top Feature", value: topFeature, compact: true },
    ]);

    trainStatus.textContent = formatJson(trainingPayload.details || trainingPayload);
    updateCharts(overview);
}


function renderPrediction(prediction) {
    const zoneClass = `zone-${String(prediction.crime_zone || "medium").toLowerCase()}`;
    const probability = Number(prediction.probability_score || 0);

    resultPlaceholder.classList.add("hidden");
    resultBox.classList.remove("hidden");

    resultCards.innerHTML = `
        <article class="result-card ${zoneClass}">
            <span class="label">Crime Zone</span>
            <span class="value">${escapeHtml(titleCase(prediction.crime_zone))}</span>
        </article>
        <article class="result-card">
            <span class="label">Case Closed Prediction</span>
            <span class="value">${escapeHtml(prediction.case_closed_prediction)}</span>
        </article>
        <article class="result-card">
            <span class="label">Cluster ID</span>
            <span class="value">${escapeHtml(prediction.cluster_id)}</span>
        </article>
        <article class="result-card">
            <span class="label">Model Version</span>
            <span class="value small">${escapeHtml(prediction.model_version || "legacy")}</span>
        </article>
    `;

    probabilityText.textContent = formatPercent(probability);
    probabilityFill.style.width = `${Math.max(6, Math.min(probability * 100, 100))}%`;
    predictionJson.textContent = formatJson(prediction);
}


async function checkStatus() {
    setButtonLoading(healthBtn, true, "Refresh Status", "Refreshing...");
    toggleLoader(statusLoader, true);

    try {
        const response = await fetch("/api/status");
        const data = await safeJsonParse(response);
        if (!response.ok) {
            throw new Error(data.message || "Unable to fetch API status.");
        }
        renderStatus(data);
    } catch (error) {
        statusJson.textContent = `Status error: ${error.message}`;
        renderSummaryCards(statusSummary, [
            { label: "API Status", value: "Error" },
            { label: "Models Ready", value: "No" },
            { label: "Dataset", value: "Unknown" },
        ]);
        heroServiceStatus.textContent = "Service Error";
    } finally {
        toggleLoader(statusLoader, false);
        setButtonLoading(healthBtn, false, "Refresh Status", "Refreshing...");
    }
}


async function trainModel() {
    setButtonLoading(trainBtn, true, "Train Pipeline", "Training...");
    toggleLoader(trainLoader, true);
    trainStatus.textContent = "Training is in progress. Large datasets may take some time.";

    const payload = {};
    if (datasetPathInput?.value.trim()) {
        payload.dataset_path = datasetPathInput.value.trim();
    }
    if (maxRowsInput?.value.trim()) {
        payload.max_rows = Number(maxRowsInput.value);
    }

    try {
        const response = await fetch("/api/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await safeJsonParse(response);
        if (!response.ok || data.status !== "success") {
            throw new Error(data.message || "Training failed.");
        }

        renderTraining(data);
        await checkStatus();
    } catch (error) {
        trainStatus.textContent = `Training error: ${error.message}`;
    } finally {
        toggleLoader(trainLoader, false);
        setButtonLoading(trainBtn, false, "Train Pipeline", "Training...");
    }
}


async function handlePredict(event) {
    event.preventDefault();
    setButtonLoading(predictBtn, true, "Predict Crime Case", "Predicting...");

    const formData = new FormData(predictForm);
    const payload = Object.fromEntries(formData.entries());
    const numericFields = [
        "victim_age",
        "police_deployed",
        "occurrence_year",
        "occurrence_month",
        "occurrence_hour",
        "report_delay_hours",
    ];

    numericFields.forEach((field) => {
        payload[field] = Number(payload[field]);
    });

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await safeJsonParse(response);
        if (!response.ok || data.status !== "success") {
            throw new Error(data.message || "Prediction failed.");
        }
        renderPrediction(data.prediction);
    } catch (error) {
        resultPlaceholder.classList.add("hidden");
        resultBox.classList.remove("hidden");
        resultCards.innerHTML = `
            <article class="result-card zone-high">
                <span class="label">Prediction Error</span>
                <span class="value">${escapeHtml(error.message)}</span>
            </article>
        `;
        probabilityText.textContent = "0%";
        probabilityFill.style.width = "0%";
        predictionJson.textContent = `Prediction error: ${error.message}`;
    } finally {
        setButtonLoading(predictBtn, false, "Predict Crime Case", "Predicting...");
    }
}


healthBtn.addEventListener("click", checkStatus);
trainBtn.addEventListener("click", trainModel);
predictForm.addEventListener("submit", handlePredict);
themeToggle.addEventListener("click", toggleTheme);

applyTheme(getSavedTheme());
checkStatus();
