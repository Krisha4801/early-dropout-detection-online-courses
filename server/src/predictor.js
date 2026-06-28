import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, "..", "..");

export function runPrediction(payload) {
  const pythonBin = process.env.PYTHON_BIN || "python";
  const scriptPath = path.join(PROJECT_ROOT, "scripts", "predict_manual.py");
  const modelPath = path.join(PROJECT_ROOT, "models", "dropout_model.joblib");

  return new Promise((resolve, reject) => {
    const child = spawn(pythonBin, [scriptPath, "--model-path", modelPath], {
      cwd: PROJECT_ROOT,
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      reject(new Error(`Could not start Python predictor: ${error.message}`));
    });

    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `Python predictor exited with code ${code}`));
        return;
      }

      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(new Error(`Python predictor returned invalid JSON: ${error.message}`));
      }
    });

    child.stdin.write(JSON.stringify(payload));
    child.stdin.end();
  });
}
