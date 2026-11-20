// 3D viewers (<model-viewer> elements) in the docs pages.
// model-viewer is loaded only on pages that have a viewer. A viewer with reveal="manual" shows
// its poster, and the script and the mesh are downloaded only when the viewer is clicked.

const MODEL_VIEWER_URL =
  "https://cdn.jsdelivr.net/npm/@google/model-viewer@4.4.0/dist/model-viewer.min.js";

let modelViewerReady = null;

function loadModelViewer() {
  modelViewerReady ??= import(MODEL_VIEWER_URL).then(() =>
    customElements.whenDefined("model-viewer")
  );
  return modelViewerReady;
}

// the same look as the paper's project page
function setColor(viewer) {
  const material = viewer.model?.materials[0];
  if (!material) return;
  material.pbrMetallicRoughness.setBaseColorFactor("#e09568");
  material.pbrMetallicRoughness.setMetallicFactor(1);
  material.pbrMetallicRoughness.setRoughnessFactor(1);
}

const viewers = document.querySelectorAll("model-viewer");

for (const viewer of viewers) {
  viewer.addEventListener("load", () => setColor(viewer));
  // in case the model finished loading before the listener was added
  if (viewer.loaded) setColor(viewer);

  if (viewer.getAttribute("reveal") === "manual") {
    viewer.addEventListener(
      "click",
      () => loadModelViewer().then(() => viewer.dismissPoster()),
      { once: true }
    );
  } else {
    loadModelViewer();
  }
}
