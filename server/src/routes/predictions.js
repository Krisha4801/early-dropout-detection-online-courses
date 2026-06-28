import express from "express";
import { randomUUID } from "node:crypto";
import { z } from "zod";

import { runPrediction } from "../predictor.js";

const router = express.Router();

const numericInput = z.union([z.number(), z.string()]).optional();
const payloadSchema = z.object({
  studentReference: z.string().max(80).optional(),
  code_module: z.string().min(1).default("FFF"),
  code_presentation: z.string().min(1).default("2014J"),
  gender: z.string().min(1).default("M"),
  region: z.string().min(1).default("East Anglian Region"),
  highest_education: z.string().min(1).default("A Level or Equivalent"),
  imd_band: z.string().min(1).default("50-60%"),
  age_band: z.string().min(1).default("0-35"),
  disability: z.string().min(1).default("N"),
  num_of_prev_attempts: numericInput,
  studied_credits: numericInput,
  module_presentation_length: numericInput,
  days_registered_before_start: numericInput,
  total_clicks: numericInput,
  active_days: numericInput,
  unique_sites_visited: numericInput,
  max_clicks_in_day: numericInput,
  std_clicks_per_active_day: numericInput,
  clicks_pre_course: numericInput,
  clicks_days_00_07: numericInput,
  clicks_days_08_14: numericInput,
  clicks_days_15_cutoff: numericInput,
  resource_clicks: numericInput,
  forum_clicks: numericInput,
  homepage_clicks: numericInput,
  content_clicks: numericInput,
  subpage_clicks: numericInput,
  url_clicks: numericInput,
  quiz_clicks: numericInput,
  first_activity_day: numericInput,
  last_activity_day: numericInput,
  days_since_last_activity: numericInput,
  due_assessments: numericInput,
  due_weight: numericInput,
  assessment_submissions: numericInput,
  submitted_due_assessments: numericInput,
  late_submissions: numericInput,
  on_time_submissions: numericInput,
  banked_assessments: numericInput,
  mean_score: numericInput,
  min_score: numericInput,
  max_score: numericInput,
  submitted_weight: numericInput,
  weighted_score: numericInput,
});

router.get("/health", (_req, res) => {
  res.json({
    ok: true,
    storage: "stateless",
  });
});

router.get("/predictions", (_req, res) => {
  res.json({ predictions: [] });
});

router.post("/predict", async (req, res, next) => {
  try {
    const parsed = payloadSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({
        message: "Invalid prediction input.",
        issues: parsed.error.issues,
      });
      return;
    }

    const input = parsed.data;
    const result = await runPrediction(input);
    const record = {
      _id: randomUUID(),
      studentReference: input.studentReference || "",
      input,
      result,
      createdAt: new Date().toISOString(),
    };

    res.status(201).json({ prediction: record });
  } catch (error) {
    next(error);
  }
});

export default router;
