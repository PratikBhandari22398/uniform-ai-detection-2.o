const mongoose = require("mongoose");

const userSchema = new mongoose.Schema(
  {
    username: {
      type: String,
      required: true,
      unique: true,
      trim: true,
    },

    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true,
    },

    password: {
      type: String,
      required: true,
    },

    department: {
      type: String,
      required: true,
    },

    // ✅ FIXED HERE
    year: {
      type: Number,
      required: true,
      enum: [1, 2, 3], // First, Second, Third
    },

    division: {
      type: String,
      required: true,
      enum: ["A", "B", "C"],
    },

    role: {
      type: String,
      default: "student",
    },
  },
  { timestamps: true }
);

module.exports = mongoose.model("User", userSchema);
