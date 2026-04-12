// models/detection.js
const mongoose = require("mongoose");

const detectionSchema = new mongoose.Schema(
  {
    user: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },

    // logged-in user cha username
    username: { type: String, required: true },

    label: { type: String, required: true },
    confidence: { type: Number, required: true },
    isCompliant: { type: Boolean, required: true },
    source: { type: String, enum: ["upload", "camera"], default: "upload" },
  },
  { timestamps: true } // createdAt, updatedAt
);

module.exports = mongoose.model("Detection", detectionSchema);
