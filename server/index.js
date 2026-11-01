import cors from "cors";
import express from "express";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const app = express();
const port = Number(process.env.PORT || 3001);
const root = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(root, "..");
const python = process.env.PYTHON || "C:/Users/Saurabh Kumar/AppData/Local/Programs/Python/Python312/python.exe";

app.use(cors());
app.use(express.json());

app.get("/api/health", (_req, res) => res.json({ status: "ok", service: "diabetes-risk-api" }));

app.post("/api/auth/login", (req, res) => {
  const { email, password } = req.body ?? {};
  if (!email || !password) {
    return res.status(400).json({ message: "Email and password are required." });
  }
  if (password.length < 6) {
    return res.status(401).json({ message: "Use a password with at least 6 characters." });
  }
  return res.json({
    user: { name: email.split("@")[0].replace(/[._-]/g, " "), email },
    token: `demo-${Buffer.from(email).toString("base64url")}`
  });
});

app.post("/api/predict", (req, res) => {
  const input = req.body;
  if (!input || !["screening", "full"].includes(input.featureSet)) {
    return res.status(400).json({ message: "featureSet must be screening or full." });
  }

  const child = spawn(python, [path.join(projectRoot, "predict.py")], {
    cwd: projectRoot,
    stdio: ["pipe", "pipe", "pipe"]
  });
  let output = "";
  let errors = "";
  child.stdout.on("data", (chunk) => { output += chunk.toString(); });
  child.stderr.on("data", (chunk) => { errors += chunk.toString(); });
  child.on("error", (error) => res.status(500).json({ message: `Python worker failed: ${error.message}` }));
  child.on("close", (code) => {
    if (code !== 0) {
      return res.status(500).json({ message: "Prediction failed.", detail: errors.slice(-1000) });
    }
    try {
      return res.json(JSON.parse(output));
    } catch {
      return res.status(500).json({ message: "Prediction returned an invalid response.", detail: errors.slice(-1000) });
    }
  });
  child.stdin.end(JSON.stringify(input));
});

app.listen(port, () => {
  console.log(`Diabetes Risk API listening on http://localhost:${port}`);
});
