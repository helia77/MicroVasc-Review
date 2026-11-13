// 3D viewers on the Binarization and Skeletonization pages.
// model-viewer is loaded only on pages that have a <model-viewer> element.

const MODEL_VIEWER_URL =
  "https://cdn.jsdelivr.net/npm/@google/model-viewer@4.4.0/dist/model-viewer.min.js";

const VIEWER_IDS = ["viewer-segmentation", "viewer-centerline"];

// the same look as the paper's project page
function setColor(viewer) {
  const material = viewer.model?.materials[0];
  if (!material) return;
  material.pbrMetallicRoughness.setBaseColorFactor("#e09568");
  material.pbrMetallicRoughness.setMetallicFactor(1);
  material.pbrMetallicRoughness.setRoughnessFactor(1);
}

if (document.querySelector("model-viewer")) {
  for (const id of VIEWER_IDS) {
    const viewer = document.getElementById(id);
    if (!viewer) continue;
    viewer.addEventListener("load", () => setColor(viewer));
    // in case the model finished loading before the listener was added
    if (viewer.loaded) setColor(viewer);
  }
  import(MODEL_VIEWER_URL);
}
