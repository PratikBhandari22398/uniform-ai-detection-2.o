require("dotenv").config();
const express = require("express");
const path = require("path");
const mongoose = require("mongoose");
const session = require("express-session");
const MongoStore = require("connect-mongo");
const multer = require("multer");
const fs = require("fs");
const tf = require("@tensorflow/tfjs-node"); // ✅ IMPORTANT

const app = express();

/* ===================== MODELS ===================== */
const User = require("./models/user");
const Detection = require("./models/detection");

/* ===================== CONFIG ===================== */
const PORT = process.env.PORT || 3000;
const MONGODB_URI =
  process.env.MONGO_URI || "mongodb://localhost:27017/uniformpro";

const TEACHER_ID = process.env.TEACHER_ID || "teacher123";
const TEACHER_PASSWORD = process.env.TEACHER_PASSWORD || "teacher@999";

/* ===================== VIEW ENGINE ===================== */
app.set("views", path.join(__dirname, "views"));
app.set("view engine", "ejs");

/* ===================== STATIC ===================== */
app.use(express.static(path.join(__dirname, "public")));

/* ===================== BODY PARSER ===================== */
app.use(express.urlencoded({ extended: true }));
app.use(express.json({ limit: "10mb" }));

/* ===================== DATABASE ===================== */
mongoose
  .connect(MONGODB_URI)
  .then(() => console.log("🟢 MongoDB Connected"))
  .catch((err) => console.error("🔴 Mongo Error:", err));

/* ===================== SESSION ===================== */
app.use(
  session({
    secret: process.env.SESSION_SECRET || "uniform-secret",
    resave: false,
    saveUninitialized: false,
    store: MongoStore.create({ mongoUrl: MONGODB_URI }),
    cookie: { maxAge: 1000 * 60 * 60 * 24 * 7 },
  })
);

/* ===================== GLOBAL USER ===================== */
app.use((req, res, next) => {
  res.locals.currentUser = req.session.user || null;
  res.locals.isTeacher = req.session.isTeacher || false;
  next();
});

/* ===================== AUTH HELPERS ===================== */
function requireLogin(req, res, next) {
  if (!req.session.user) return res.redirect("/login");
  next();
}

function requireTeacher(req, res, next) {
  if (!req.session.isTeacher) return res.redirect("/teacher-login");
  next();
}

/* ===================== ROUTES ===================== */

app.get("/", (req, res) => res.render("home"));

app.get("/signup", (req, res) => {
  if (req.session.user) return res.redirect("/detect");
  res.render("users/signup");
});

app.post("/signup", async (req, res) => {
  try {
    const user = await User.create({ ...req.body, role: "student" });
    req.session.user = user;
    res.redirect("/detect");
  } catch (err) {
    res.render("users/signup", { error: "User already exists" });
  }
});

app.get("/login", (req, res) => {
  if (req.session.user) return res.redirect("/detect");
  res.render("users/login");
});

app.post("/login", async (req, res) => {
  const user = await User.findOne(req.body);
  if (!user)
    return res.render("users/login", { error: "Invalid credentials" });
  req.session.user = user;
  res.redirect("/detect");
});

app.post("/logout", (req, res) => {
  req.session.destroy(() => res.redirect("/"));
});

app.get("/detect", requireLogin, async (req, res) => {
  const history = await Detection.find({ user: req.session.user._id })
    .sort({ createdAt: -1 })
    .limit(10);
  res.render("detect", { history });
});

/* ===================== TEACHER ===================== */

app.get("/teacher-login", (req, res) =>
  res.render("users/teacher-login")
);

app.post("/teacher-login", (req, res) => {
  if (
    req.body.teacherId === TEACHER_ID &&
    req.body.password === TEACHER_PASSWORD
  ) {
    req.session.isTeacher = true;
    return res.redirect("/teacher/students");
  }
  res.render("users/teacher-login", { error: "Invalid credentials" });
});

app.get("/teacher/students", requireTeacher, async (req, res) => {
  const students = await User.find({ role: "student" });
  const detections = await Detection.find().sort({ createdAt: -1 });

  const uniformMap = {};
  detections.forEach((d) => {
    if (!uniformMap[d.user]) uniformMap[d.user] = d;
  });

  res.render("users/teacher-students", {
    students,
    uniformMap,
    query: req.query,
  });
});

/* ===================== IMAGE UPLOAD ===================== */
const upload = multer({
  dest: path.join(__dirname, "public/uploads"),
});

/* ===================== LOAD AI MODEL ===================== */
let model = null;

(async () => {
  try {
    const modelPath =
      "file://" + path.join(__dirname, "tfjs_model/model.json");
    model = await tf.loadLayersModel(modelPath);
    console.log("✅ AI Model Loaded");
  } catch (err) {
    console.error("❌ Model Load Error:", err.message);
  }
})();

/* ===================== DETECTION ===================== */
async function handleDetection(req, res) {
  if (!model) return res.status(500).json({ error: "Model not ready" });
  if (!req.file) return res.status(400).json({ error: "No image" });

  try {
    const buffer = fs.readFileSync(req.file.path);

    const tensor = tf.node
      .decodeImage(buffer, 3)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat()
      .div(255);

    const preds = model.predict(tensor);
    const scores = preds.dataSync();

    const LABELS = [
      "1st year",
      "2nd year",
      "3rd year",
      "without uniform and id",
    ];

    const maxIndex = scores.indexOf(Math.max(...scores));
    const label = LABELS[maxIndex];
    const confidence = scores[maxIndex];

    await Detection.create({
      user: req.session.user._id,
      username: req.session.user.username,
      label,
      confidence,
      isCompliant: label !== "without uniform and id",
    });

    tf.dispose([tensor, preds]);
    fs.unlinkSync(req.file.path);

    res.json({ label, confidence });
  } catch (err) {
    console.error("Detection Error:", err);
    res.status(500).json({ error: "Detection failed" });
  }
}

app.post("/detect-image", requireLogin, upload.single("image"), handleDetection);
app.post("/detect-frame", requireLogin, upload.single("image"), handleDetection);

/* ===================== START ===================== */
app.listen(PORT, () =>
  console.log(`🚀 Server running at http://localhost:${PORT}`)
);
