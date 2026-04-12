const mongoose = require("mongoose");
const User = require("./models/user");

mongoose.connect("mongodb://localhost:27017/uniformpro").then(async () => {
  await User.deleteOne({ username: "TCHR-12345" });
  await User.create({
    username: "TCHR-12345",
    password: "teacher@999", // Note: plain text because hashing wasn't implemented originally
    email: "teacher@example.com",
    department: "CS",
    year: 1,
    division: "A",
    role: "teacher"
  });
  console.log("Teacher seeded!");
  process.exit(0);
});
