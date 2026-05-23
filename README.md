# Insurance Risk Analytics

## Project Overview
This project analyzes insurance risk data for AlphaCare Insurance Solutions (ACIS).

The project includes:
- Exploratory Data Analysis
- Hypothesis Testing
- Predictive Modeling
- Risk-Based Pricing

## Project Structure

## Setup Instructions

### Clone Repository
```bash
git clone <repo-url>


Save.

---

# STEP 10 — Setup GitHub Actions CI/CD

Create file:
```text
.github/workflows/ci.yml

name: CI

on:
  push:
    branches:
      - main
      - task-1

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Dependencies
        run: |
          pip install -r requirements.txt

      - name: Run Tests
        run: |
          pytest

      - name: Run Lint
        run: |
          flake8 .