# 🎓 Uniform AI Detection System

An AI-powered web application that detects whether a student is wearing a proper uniform using image-based analysis.  
This project is built for educational institutions to automate and simplify uniform compliance verification.

---

## 🚀 Project Overview

The **Uniform AI Detection System** allows students or teachers to upload an image of a student.  
The system analyzes the image using an AI model and determines whether the student is wearing the correct uniform.

This project demonstrates the integration of **Artificial Intelligence + Full Stack Web Development** in a real-world academic use case.

---

## ✨ Features

- 🧠 AI-based uniform detection  
- 👤 Role-based authentication (Student / Teacher)  
- 📤 Image upload and processing  
- 🎨 Clean and responsive UI using EJS  
- 🔐 Secure backend with MongoDB  
- 🧩 Modular MVC architecture  

---

## 🛠️ Tech Stack

### 🌐 Frontend
- HTML5  
- CSS3  
- EJS (Embedded JavaScript Templates)  

### ⚙️ Backend
- Node.js  
- Express.js  
- MongoDB  
- Mongoose  

### 🤖 AI / ML
- TensorFlow / TensorFlow.js  
- Custom-trained uniform detection model  

> ⚠️ AI model files are intentionally excluded from this repository.

---

## 📁 Project Structure

```text
finalproject/
│── app.js
│── package.json
│── package-lock.json
│── models/
│   ├── user.js
│   └── detection.js
│── views/
│   ├── home.ejs
│   ├── detect.ejs
│   ├── about.ejs
│   ├── benefits.ejs
│   ├── team.ejs
│   ├── workflow.ejs
│   └── users/
│       ├── login.ejs
│       ├── signup.ejs
│       ├── teacher-login.ejs
│       └── teacher-students.ejs
│── public/
│── .gitignore
│── README.md
````

---

## 🔐 Environment Variables

Create a `.env` file in the root directory and add the following:

```env
PORT=3000
MONGO_URI=your_mongodb_connection_string
```

> ⚠️ The `.env` file is ignored for security reasons and is not pushed to GitHub.

---

## ▶️ How to Run the Project Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/PratikBhandari22398/uniform-ai-detection.git
cd uniform-ai-detection
```

### 2️⃣ Install Dependencies

```bash
npm install
```

### 3️⃣ Start the Server

```bash
npm start
```

### 4️⃣ Open in Browser

```text
http://localhost:3000
```

---

## 🧠 AI Model Information

* AI model files (`.h5`, `tfjs_model/`) are **not included** in this repository.
* This is done intentionally to:

  * Reduce repository size
  * Protect trained AI assets
* The model can be loaded using:

  * Google Drive
  * Cloud storage
  * Git LFS (if required)

---

## 🎯 Use Cases

* Schools and colleges
* Smart campus systems
* AI-based discipline monitoring
* Academic AI demonstration project
* Resume and portfolio project

---

## 📚 Learning Outcomes

* Full-stack web development using Node.js
* Integration of AI models with backend services
* Secure authentication and MVC architecture
* Professional Git & GitHub workflow
* Real-world AI project implementation

---
