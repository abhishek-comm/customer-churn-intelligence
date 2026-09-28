const form = document.querySelector("#profile-form");
const button = document.querySelector("#predict-button");
const errorBox = document.querySelector("#form-error");
const emptyState = document.querySelector("#empty-state");
const resultState = document.querySelector("#result-state");

function payloadFromForm() {
  const data = new FormData(form);
  return Object.fromEntries(data.entries());
}

function showResult(result) {
  const probability = Math.round(result.churn_probability * 100);
  const risk = result.risk_level.toLowerCase();
  const riskCopy = {
    low: {
      headline: "Low churn risk",
      interpretation: "Customers with this profile show a lower predicted churn risk.",
    },
    medium: {
      headline: "Medium churn risk",
      interpretation: "Customers with this profile show a moderate predicted churn risk.",
    },
    high: {
      headline: "High churn risk",
      interpretation: "Customers with this profile show an elevated predicted churn risk.",
    },
  }[risk];
  if (!riskCopy) throw new Error("Unknown risk level returned by API.");

  emptyState.classList.add("hidden");
  resultState.classList.remove("hidden");
  document.querySelector("#prediction-label").textContent = riskCopy.headline;
  document.querySelector("#probability").textContent = `${probability}%`;
  document.querySelector("#risk-level").textContent = result.risk_level.toUpperCase();
  document.querySelector("#risk-ring").className = `risk-ring ${risk}`;
  document.querySelector("#meter-fill").style.width = `${probability}%`;
  document.querySelector("#meter-fill").className = risk;
  document.querySelector("#model-name").textContent = result.model;
  document.querySelector("#insight-text").textContent = riskCopy.interpretation;
  const signalList = document.querySelector("#signal-list");
  signalList.replaceChildren();
  result.feature_signals.forEach((signal) => {
    const item = document.createElement("li");
    item.textContent = signal.feature;
    signalList.appendChild(item);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.textContent = "";
  button.disabled = true;
  button.querySelector("span:first-child").textContent = "Analyzing profile…";
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadFromForm()),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Prediction failed.");
    showResult(result);
  } catch (error) {
    errorBox.textContent = error.message || "The API is unavailable. Please try again.";
  } finally {
    button.disabled = false;
    button.querySelector("span:first-child").textContent = "Analyze churn risk";
  }
});

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const health = await response.json();
    document.querySelector("#status-text").textContent = health.model_loaded ? "API Online" : "Model unavailable";
    if (health.model_loaded) document.querySelector("#status-dot").classList.add("online");
  } catch {
    document.querySelector("#status-text").textContent = "API Offline";
  }
}

checkHealth();
