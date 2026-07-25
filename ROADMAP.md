# Group 8 Project Resources

## 0.  Git Workflow 

### Step 1: Accept the Invite & Clone the Repository
* In VS Code:
* Run: `git clone https://github.com/avatarmongacancode34/Intro_to_AI_project.git`
* Run: `cd Intro_to_AI_project` to move into the project directory.

### Step 2: Verify Your Local Identity
** Ensure your first name is set on this machine so our commit history is clear
* Run: `git config --global user.name "Your First Name"`

### Step 3: Create Your Role-Specific Branch
** Here are the suggested branch names Vanessa: `data-pipeline` Shaun: `cnn-model` Nadine: `streamlit-ui` Victoria `model-evaluation`.
* Run: `git checkout -b your-branch-name`


### Step 4: Stage Your Changes
* Run: `git add .` 

### Step 5: Commit Your Code
* Run: `git commit -m "Briefly describe what you added or fixed here"`

### Step 6: Push Your Branch to GitHub
Upload your local branch and its commits to the shared remote repository.
* Run: `git push origin your-branch-name`
* Go to the GitHub website. Click the green "Compare & pull request" button so the we can review your code before it merges into `main`.

---

## 1. Vanessa
**Focus:** Getting the raw Kaggle dataset cleaned, formatted, and loaded into the GPU memory.
[Kaggle Link](https://www.kaggle.com/datasets/anachaba/adinkra-symbols?select=Adinkra+symbol+dataset)

### Concepts to Learn:
* **Tensors:** data structure in PyTorch for representing images as matrices.
* **Dataset & DataLoader Classes:** Writing custom Python classes to map the Adinkra folder paths to integer labels (0 through 9) and handle batching.
* **Data Augmentation:** Using `torchvision.transforms` to rotate, crop, and normalize images during training


---

## 2. Shaun
**Focus:** Building the  neural network and teaching it to recognize the geometric shapes.

### Concepts to Learn:
* **CNN Architecture:** How `Conv2D`, `ReLU`, `MaxPool2D`, and `Linear` layers stack together to extract spatial features.
* **Forward vs. Backward Pass:** Making a prediction versus calculating the error gradient (backpropagation).
* **Optimization & Loss:** How Cross-Entropy Loss calculates the error, and how the Adam Optimizer adjusts the weights.
* **Regularization:** Using Dropout ($p=0.5$) and Early Stopping to prevent overfitting on the small dataset.


---

## 3. Nadine
**Focus:** Creating a user-friendly web application for the live demo.

### Concepts to Learn:
* **Streamlit Fundamentals:** Building the UI layout, handling file uploads (images), and managing `st.session_state`.
* **Inference Pipeline:** Taking the user's uploaded image, converting it to a tensor, and passing it through the model.
* **Inference Mode (`model.eval()`):** Turning off Dropout and gradient tracking to make real predictions on the local machine.

**Recommended Video:**
* [Streamlit Tutorial - The Basics](https://www.youtube.com/watch?v=MPtTcQ5xMIk) 

---

## 4. Victoria
**Focus:** Linking the AI predictions to cultural context, evaluating model success

### Concepts to Learn:
* **JSON Dictionaries:** Structuring a clean `.json` file that maps the model's output (e.g., Class 0) to the cultural metadata (Name, Meaning, Proverb). Get the information from Nadine
* **Confusion Matrices:** Understanding how to read and generate a $10 \times 10$ heatmap to see exactly which symbols the model is confusing.
* **Evaluation Metrics:** Understanding Top-1 Accuracy and Validation Loss curves to write the Methodology section of the final report.
