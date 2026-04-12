require("dotenv").config();
const express = require("express");
const path = require("path");
const mongoose = require("mongoose");
const session = require("express-session");
const MongoStore = require("connect-mongo");
const multer = require("multer");
const fs = require("fs");
const tf = require("@tensorflow/tfjs-node");

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

/* ===================== STATIC FILES ===================== */
app.use(express.static(path.join(__dirname, "public")));
app.use("/model", express.static(path.join(__dirname, "tfjs_model")));

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
  res.locals.userId = req.session.userId || null;
  res.locals.userRole = req.session.role || null;
  res.locals.isTeacher = req.session.role === "teacher";
  next();
});

/* ===================== AUTH HELPERS ===================== */
function requireLogin(req, res, next) {
  if (!req.session.user) return res.redirect("/users/auth");
  next();
}

function requireTeacher(req, res, next) {
  if (!req.session.userId || req.session.role !== "teacher") {
    return res.redirect("/users/teacher-login");
  }
  next();
}

function studentOnly(req, res, next) {
  if (req.session.role === "teacher") {
    return res.redirect("/teacher/dashboard");
  }
  next();
}

/* ===================== ROUTES ===================== */

app.get("/", (req, res) => res.render("home"));

app.get("/users/auth", (req, res) => {
  if (req.session.user) return res.redirect("/detect");
  res.render("users/auth", { error: null, tab: req.query.tab || 'login' });
});

app.get("/signup", (req, res) => {
  res.redirect("/users/auth?tab=signup");
});

app.post("/signup", async (req, res) => {
  try {
    const user = await User.create({ ...req.body, role: "student" });
    req.session.user = user;
    req.session.userId = user._id;
    req.session.role = "student";
    res.redirect("/detect");
  } catch (err) {
    res.render("users/auth", { error: "User already exists", tab: 'signup' });
  }
});

app.get("/login", (req, res) => {
  res.redirect("/users/auth?tab=login");
});

app.post("/login", async (req, res) => {
  const { username, password } = req.body;
  const user = await User.findOne({ username, password });
  if (!user)
    return res.render("users/auth", { error: "Invalid credentials", tab: 'login' });
  req.session.user = user;
  req.session.userId = user._id;
  req.session.role = user.role || "student";
  res.redirect("/detect");
});

app.post("/logout", (req, res) => {
  req.session.destroy(() => {
    // res.redirect works for both:
    // - Teacher: regular form POST → browser follows the redirect directly
    // - Student auto-logout: fetch() follows redirect silently,
    //   then .finally(() => window.location.href = '/') navigates the browser
    res.redirect("/");
  });
});

app.get("/detect", requireLogin, studentOnly, async (req, res) => {
  const history = await Detection.find({ user: req.session.user._id })
    .sort({ createdAt: -1 })
    .limit(10);
  res.render("detect", { history });
});

app.get("/student/profile", requireLogin, studentOnly, async (req, res) => {
  try {
    const user = await User.findById(req.session.userId);
    const detections = await Detection.find({ user: req.session.userId }).sort({ createdAt: -1 });

    const total = detections.length;
    const compliant = detections.filter(d => d.isCompliant).length;
    const nonCompliant = total - compliant;
    const percentage = total > 0 ? Math.round((compliant / total) * 100) : 0;
    const history = detections.slice(0, 5); // Last 5

    res.render("student/profile", {
      user,
      total,
      compliant,
      nonCompliant,
      percentage,
      history
    });
  } catch (err) {
    console.error("Profile error:", err);
    res.redirect("/");
  }
});

// Overwrite the previous /profile route to point to the new one
app.get("/profile", (req, res) => res.redirect("/student/profile"));

/* ===================== TEACHER ===================== */

app.get("/users/teacher-login", (req, res) => {
  // If already logged in as teacher, go straight to dashboard
  if (req.session.role === "teacher") return res.redirect("/teacher/dashboard");
  res.render("users/teacher-login", { error: null });
});

app.post("/users/teacher-login", (req, res) => {
  const TEACHER_ID = process.env.TEACHER_ID || "teacher123";
  const TEACHER_PASSWORD = process.env.TEACHER_PASSWORD || "teacher@999";

  if (
    req.body.teacherId === TEACHER_ID &&
    req.body.password === TEACHER_PASSWORD
  ) {
    req.session.userId = "teacher_admin";
    req.session.role = "teacher";
    // Set a display user so navbar shows correctly
    req.session.user = { username: TEACHER_ID };
    return res.redirect("/teacher/dashboard");
  }

  res.render("users/teacher-login", { error: "Invalid Teacher ID or Password" });
});

app.get("/teacher/dashboard", requireTeacher, async (req, res) => {
  try {
    const { date, department, year, division, compliance, viewMode = "history" } = req.query;

    // 1. Resolve User IDs that match the departmental/year filters
    const userFilter = { role: "student" };
    if (department) userFilter.department = department;
    if (year)       userFilter.year = Number(year);
    if (division)   userFilter.division = division;

    const matchedUsers = await User.find(userFilter).select("_id");
    const userIds = matchedUsers.map(u => u._id);

    // 2. Build Detection filter based on resolved users and query params
    const detectionFilter = { user: { $in: userIds } };
    
    if (date) {
      const start = new Date(date);
      const end = new Date(date);
      end.setUTCHours(23, 59, 59, 999);
      detectionFilter.createdAt = { $gte: start, $lte: end };
    }
    
    if (compliance) {
      detectionFilter.isCompliant = (compliance === "compliant");
    }

    // 3. Fetch detections with user details
    let detections = await Detection.find(detectionFilter)
      .populate("user")
      .sort({ createdAt: -1 })
      .limit(200);

    // 4. Handle "Latest Status" View Mode (Show only top record per student)
    if (viewMode === "latest") {
      const seen = new Set();
      detections = detections.filter(d => {
        if (!d.user) return false;
        const uid = String(d.user._id);
        if (seen.has(uid)) return false;
        seen.add(uid);
        return true;
      });
    }

    console.log(`[Dashboard] View: ${viewMode} | Found: ${detections.length} records`);

    res.render("users/teacher-dashboard", {
      detections,
      query: req.query,
      viewMode
    });
  } catch (err) {
    console.error("[Teacher Dashboard Error]", err);
    res.status(500).send("An error occurred loading the dashboard.");
  }
});

app.get("/teacher/dashboard/export", requireTeacher, async (req, res) => {
  try {
    const { date, department, year, division, compliance, viewMode = "latest" } = req.query;

    const userFilter = { role: "student" };
    if (department) userFilter.department = department;
    if (year)       userFilter.year = Number(year);
    if (division)   userFilter.division = division;

    const matchedUsers = await User.find(userFilter).select("_id");
    const userIds = matchedUsers.map(u => u._id);

    const detectionFilter = { user: { $in: userIds } };
    if (date) {
      const start = new Date(date);
      const end = new Date(date);
      end.setUTCHours(23, 59, 59, 999);
      detectionFilter.createdAt = { $gte: start, $lte: end };
    }
    if (compliance) {
      detectionFilter.isCompliant = (compliance === "compliant");
    }

    let records = await Detection.find(detectionFilter)
      .populate("user")
      .sort({ createdAt: -1 });

    if (viewMode === "latest") {
      const seen = new Set();
      records = records.filter(d => {
        if (!d.user) return false;
        const uid = String(d.user._id);
        if (seen.has(uid)) return false;
        seen.add(uid);
        return true;
      });
    }

    const rows = [["Username", "Department", "Year", "Division", "Status", "Date Time"]];
    records.forEach(d => {
      const u = d.user || {};
      const status = d.isCompliant ? "Uniform OK" : "Not in Uniform";
      const time = new Date(d.createdAt).toLocaleString("en-IN");
      rows.push([
        `"${u.username || "Unknown"}"`,
        `"${u.department || "-"}"`,
        `"${u.year || "-"}"`,
        `"${u.division || "-"}"`,
        `"${status}"`,
        `"${time}"`
      ]);
    });

    const csv = rows.map(r => r.join(",")).join("\n");
    res.setHeader("Content-Type", "text/csv");
    res.setHeader("Content-Disposition", `attachment; filename=detection_${viewMode}_report.csv`);
    res.status(200).send(csv);

  } catch (err) {
    console.error("[Export Error]", err);
    res.status(500).send("Export failed");
  }
});

/* ===================== TEACHER ANALYTICS ===================== */

app.get("/teacher/analytics", requireTeacher, async (req, res) => {
  try {
    const today = new Date();
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(today.getDate() - 7);
    sevenDaysAgo.setHours(0, 0, 0, 0);

    // 1️⃣ Daily Compliance Graph (Last 7 Days)
    const dailyCompliance = await Detection.aggregate([
      { 
        $match: { 
          confidence: { $gt: 0.8 }, 
          createdAt: { $gte: sevenDaysAgo } 
        } 
      },
      {
        $group: {
          _id: { $dateToString: { format: "%Y-%m-%d", date: "$createdAt" } },
          total: { $sum: 1 },
          compliantCount: { $sum: { $cond: ["$isCompliant", 1, 0] } }
        }
      },
      { $sort: { _id: 1 } },
      {
        $project: {
          date: "$_id",
          percentage: { 
            $cond: [
              { $eq: ["$total", 0] }, 
              0, 
              { $multiply: [{ $divide: ["$compliantCount", "$total"] }, 100] }
            ]
          }
        }
      }
    ]);

    // 2️⃣ Weekly Violations (By Day of Week from 7 days ago until now)
    const weeklyViolationsRaw = await Detection.aggregate([
      { 
        $match: { 
          confidence: { $gt: 0.8 }, 
          isCompliant: false,
          createdAt: { $gte: sevenDaysAgo } 
        } 
      },
      {
        $group: {
          _id: { $dayOfWeek: "$createdAt" }, // 1 (Sun) to 7 (Sat)
          count: { $sum: 1 }
        }
      },
      { $sort: { _id: 1 } }
    ]);

    const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    const weeklyViolations = days.map((day, index) => {
      // $dayOfWeek: 1=Sun, 2=Mon... 7=Sat
      const mongoDayIndex = (index + 2) > 7 ? (index + 2 - 7) : (index + 2);
      const match = weeklyViolationsRaw.find(v => v._id === mongoDayIndex);
      return match ? match.count : 0;
    });

    // 3️⃣ Department-wise Compliance
    const departmentStats = await Detection.aggregate([
      { $match: { confidence: { $gt: 0.8 } } },
      {
        $lookup: {
          from: "users",
          localField: "user",
          foreignField: "_id",
          as: "studentInfo"
        }
      },
      { $unwind: "$studentInfo" },
      {
        $group: {
          _id: "$studentInfo.department",
          total: { $sum: 1 },
          compliantCount: { $sum: { $cond: ["$isCompliant", 1, 0] } }
        }
      },
      {
        $project: {
          department: "$_id",
          percentage: { 
            $cond: [
              { $eq: ["$total", 0] }, 
              0, 
              { $multiply: [{ $divide: ["$compliantCount", "$total"] }, 100] }
            ]
          }
        }
      }
    ]);

    // 4️⃣ Top Compliant Class (Highest Compliance Percentage)
    const classStats = await Detection.aggregate([
      { $match: { confidence: { $gt: 0.8 } } },
      {
        $lookup: {
          from: "users",
          localField: "user",
          foreignField: "_id",
          as: "studentInfo"
        }
      },
      { $unwind: "$studentInfo" },
      {
        $group: {
          _id: {
            dept: "$studentInfo.department",
            year: "$studentInfo.year",
            div: "$studentInfo.division"
          },
          total: { $sum: 1 },
          compliantCount: { $sum: { $cond: ["$isCompliant", 1, 0] } }
        }
      },
      {
        $project: {
          className: { 
            $concat: [
              "$_id.dept", " - ", 
              { 
                $switch: { 
                  branches: [
                    { case: { $eq: ["$_id.year", 1] }, then: "1st Year" },
                    { case: { $eq: ["$_id.year", 2] }, then: "2nd Year" },
                    { case: { $eq: ["$_id.year", 3] }, then: "3rd Year" }
                  ],
                  default: "N/A"
                } 
              }, 
              " - ", "$_id.div", " Division"
            ] 
          },
          percentage: { 
            $cond: [
              { $eq: ["$total", 0] }, 
              0, 
              { $multiply: [{ $divide: ["$compliantCount", "$total"] }, 100] }
            ]
          }
        }
      },
      { $sort: { percentage: -1, total: -1 } },
      { $limit: 1 }
    ]);

    res.render("users/teacher-analytics", {
      dailyData: dailyCompliance,
      weeklyViolations,
      departmentStats,
      topClass: classStats[0] || null
    });
  } catch (err) {
    console.error("[Analytics Error]", err);
    res.status(500).send("Analytics Error: " + err.message);
  }
});

/* ===================== IMAGE UPLOAD (TEMP ONLY) ===================== */
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

/* ===================== DETECTION LOGIC ===================== */
/*
  IMPORTANT:
  - Student images are NOT stored
  - Image is used for prediction only
  - Image is deleted immediately after detection
*/
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

    const newDetection = new Detection({
      user: req.session.user._id,
      username: req.session.user.username,
      label,
      confidence,
      isCompliant: label !== "without uniform and id",
    });
    
    // Server-side check: only save if confidence > 0.80 and wait limit
    // Wait, the new logic expects frontend to handle the wait limit and saving JSON.
    await newDetection.save();

    tf.dispose([tensor, preds]);

    // ✅ delete image immediately (no storage)
    fs.unlink(req.file.path, () => {});

    res.json({ label, confidence });
  } catch (err) {
    console.error("Detection Error:", err);
    res.status(500).json({ error: "Detection failed" });
  }
}

app.post("/detect-image", requireLogin, studentOnly, upload.single("image"), handleDetection);
app.post("/detect-frame", requireLogin, studentOnly, upload.single("image"), handleDetection);

/* ===================== FRONTEND DETECTION SAVE ===================== */
app.post("/detect", requireLogin, studentOnly, async (req, res) => {
  try {
    const { isCompliant, confidence, timestamp, label } = req.body;
    
    const detection = await Detection.create({
      user: req.session.user._id,
      username: req.session.user.username,
      label: label || (isCompliant ? "Uniform OK" : "Not in Uniform"),
      confidence: Number(confidence) || 0,
      isCompliant: Boolean(isCompliant),
      createdAt: timestamp ? new Date(timestamp) : new Date(),
      source: "camera"
    });
    
    res.json({ success: true, detection });
  } catch (err) {
    console.error("Save Detection Error:", err);
    res.status(500).json({ error: "Failed to save" });
  }
});

/* ===================== START SERVER ===================== */
app.listen(PORT, () =>
  console.log(`🚀 Server running at http://localhost:${PORT}`)
);
