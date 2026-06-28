import "dotenv/config";
import cors from "cors";
import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";

import predictionRoutes from "./routes/predictions.js";

const app = express();
const port = Number(process.env.PORT || 5000);
const clientOrigin = process.env.CLIENT_ORIGIN || "http://localhost:5173";
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const clientDist = path.resolve(__dirname, "..", "..", "client", "dist");

app.use(
  cors({
    origin: clientOrigin,
  }),
);
app.use(express.json({ limit: "1mb" }));

app.use("/api", predictionRoutes);
app.use(express.static(clientDist));

app.get("*", (req, res, next) => {
  if (req.path.startsWith("/api")) {
    next();
    return;
  }

  res.sendFile(path.join(clientDist, "index.html"), (error) => {
    if (error) {
      next();
    }
  });
});

app.use((error, _req, res, _next) => {
  console.error(error);
  res.status(500).json({
    message: error.message || "Something went wrong.",
  });
});

const server = app.listen(port, () => {
  console.log(`Dropout detection API running on http://localhost:${port}`);
});

server.on("error", (error) => {
  if (error.code === "EADDRINUSE") {
    console.error(
      `Port ${port} is already in use. Stop the old server or set PORT to another value.`,
    );
    process.exit(1);
  }

  throw error;
});
