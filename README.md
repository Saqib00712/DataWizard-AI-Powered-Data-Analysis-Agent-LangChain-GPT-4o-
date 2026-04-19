# DataWizard-AI-Powered-Data-Analysis-Agent-LangChain-GPT-4o-
Built a conversational AI agent that enables non-technical users to perform data science through natural language. The agent uses LangChain tools to list datasets, load and cache CSVs, generate summaries, and automatically train classification or regression models using scikit-learn. Powered by a ReAct-style AgentExecutor that reasons step-by-step 

# Data Visualization Agent with LangChain
> A conversational AI agent that lets you analyze and visualize CSV data using plain English — no SQL, no manual plotting code required. Built with LangChain, IBM Granite LLM, and pandas.

---

## What This Project Does

Instead of writing complex queries or matplotlib code manually, you simply **ask questions in natural language** and the agent generates the analysis and charts automatically.

```
You:   "Generate a pie chart showing average weekend alcohol consumption by gender"
Agent: → writes pandas + matplotlib code → executes it → displays the chart
```

The agent uses `create_pandas_dataframe_agent` from LangChain to bridge natural language and data operations — making data analysis accessible to everyone, not just engineers.

---

## Demo

**Dataset:** UCI Student Alcohol Consumption (`student-mat.csv`) — 395 Portuguese secondary school students with 33 features including grades, demographics, and lifestyle habits.

| Query | Output |
|-------|--------|
| *"How many rows of data are in this file?"* | `395` |
| *"Plot the gender count with bars"* | Bar chart of M/F distribution |
| *"Pie chart of average weekend alcohol by gender"* | Pie chart (Walc by sex) |
| *"Box plots of free time vs final grade"* | Box plots across freetime levels |
| *"Scatter plot of daily alcohol vs final grade"* | Correlation scatter plot |
| *"Compare average grades for students with/without internet"* | Bar comparison chart |

---

## Architecture

```
Natural Language Query
        ↓
[create_pandas_dataframe_agent]
        ↓
[IBM Granite LLM] — reasons about the data and writes Python code
        ↓
[pandas + matplotlib/seaborn] — executes the generated code
        ↓
Chart / Answer displayed inline
```

The agent's `return_intermediate_steps=True` setting lets you inspect the exact Python code the LLM generated — great for learning and debugging.

---

## Dataset Overview

**UCI Student Alcohol Consumption** — Mathematics course data from two Portuguese schools.

| Column | Description |
|--------|-------------|
| `G1`, `G2`, `G3` | Period 1, 2, and final grades (0–20) |
| `Dalc` / `Walc` | Weekday / weekend alcohol consumption (1–5) |
| `Medu` / `Fedu` | Mother's / father's education level (0–4) |
| `studytime` | Weekly study time (1–4 scale) |
| `absences` | Number of school absences (0–93) |
| `internet` | Internet access at home (yes/no) |
| `freetime` | Free time after school (1–5 scale) |
| `sex` | Gender (M/F) |
| `age` | Student age (15–22) |

Full dataset: [Kaggle — UCI Student Alcohol Consumption](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption)

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-0.1.16-green?style=flat-square)
![IBM Granite](https://img.shields.io/badge/IBM-Granite%20LLM-black?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-yellow?style=flat-square)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8.4-orange?style=flat-square)

- **LangChain** — `create_pandas_dataframe_agent` for NL-to-data operations
- **IBM Granite** (`granite-4-h-small`) — Core LLM via IBM WatsonX
- **pandas** — Data loading, filtering, and manipulation
- **matplotlib / seaborn** — Chart and graph generation
- **ibm-watsonx-ai** — IBM WatsonX model integration

---

## Project Structure

```
Data-Visualization-Agent/
│
├── data_viz_agent.ipynb      # Full implementation notebook
├── requirements.txt          # All dependencies
├── .env.example              # API key template (for local use)
└── README.md
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Saqib00712/IBM_GenerativeAI_Engineering-With-LLMS.git
cd IBM_GenerativeAI_Engineering-With-LLMS
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API keys (only needed if running locally outside IBM Skills Network)
```bash
cp .env.example .env
```
Edit `.env` and add your credentials:
```
WATSONX_API_KEY=your_watsonx_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

### 4. Run the notebook
```bash
jupyter notebook data_viz_agent.ipynb
```

> **Note:** If running inside IBM Skills Network JupyterLab, no API keys are needed — the environment is pre-configured with `project_id="skills-network"`.

---

## Key Concepts Covered

- **`create_pandas_dataframe_agent`** — LangChain's built-in agent that connects an LLM to a pandas DataFrame for natural language data queries
- **`return_intermediate_steps=True`** — exposing the LLM-generated Python code at each step for transparency and learning
- **Natural language to visualization** — converting plain English requests into matplotlib/seaborn plots automatically
- **IBM WatsonX integration** — loading and configuring IBM Granite LLM via `WatsonxLLM` wrapper
- **Conversational data analysis** — multi-turn querying of structured CSV data without writing SQL or pandas manually

---

## Example Visualizations Generated

```python
# Bar chart — gender distribution
agent.invoke("Generate a bar chart to plot the gender count.")

# Pie chart — alcohol consumption by gender
agent.invoke("Generate a pie chart to display average value of Walc for each Gender.")

# Box plots — free time vs final grade
agent.invoke("Create box plots to analyze the relationship between freetime and G3.")

# Scatter plots — alcohol vs academic performance
agent.invoke("Generate scatter plots to examine the correlation between Dalc and G3, and Walc and G3.")

# Inspect the LLM-generated code
print(response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n'))
```

---

## Insights from the Data

- Students with **higher parental education** tend to achieve better final grades (G3)
- Students with **internet access at home** show higher average grades
- **Higher weekend alcohol consumption** (Walc) correlates with lower G3 scores
- **More absences** negatively impact final grade performance

---

## Related Certifications

Built as part of the IBM **Fundamentals of Building AI Agents** and **Generative AI Engineering with Transformers & LLMs** Specializations on Coursera.

[![IBM Badge](https://img.shields.io/badge/IBM-AI%20Agents%20Specialization-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)

---

## Author

**Muhammad Saqib**
- GitHub: [@Saqib00712](https://github.com/Saqib00712)
- LinkedIn: [muhammad-saqib](https://www.linkedin.com/in/muhammad-saqib-68b9b3374/)
- Email: saqibkhosa649@gmail.com
- Credly: [15x IBM Certified](https://www.credly.com/users/muhammad-saqib.361f9b8c)
