# VishleshAIn: Data Analytics Personal Assistant

VishleshAIn is a comprehensive data analytics platform that helps users preprocess, analyze, model, and visualize their data. It generates detailed reports and provides insights through an AI-powered chatbot.

## Features

- **Data Upload**: Support for various formats (CSV, JSON, XML, PDF, DOCX, etc.)
- **Data Preprocessing**: Automated handling of missing values, duplicates, encoding, and outliers
- **Exploratory Data Analysis**: Interactive visualizations and statistical analysis
- **Machine Learning Modeling**: Build and evaluate models with various algorithms
- **Comprehensive Reporting**: Generate detailed professional reports
- **AI Chatbot**: Get insights and answers about your data analysis

## Project Structure

The project is organized into two main directories:

- **frontend/**: Next.js React application
- **backend/**: FastAPI Python application

## Getting Started

### Prerequisites

- Node.js (v16+)
- Python (v3.9+)
- Docker and Docker Compose (optional)

### Running with Docker

1. Clone the repository:
   \`\`\`
   git clone https://github.com/yourusername/vishleshain.git
   cd vishleshain
   \`\`\`

2. Start the application using Docker Compose:
   \`\`\`
   docker-compose up
   \`\`\`

3. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000/docs

### Running Locally

#### Backend

1. Navigate to the backend directory:
   \`\`\`
   cd backend
   \`\`\`

2. Create a virtual environment:
   \`\`\`
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   \`\`\`

3. Install dependencies:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`

4. Start the backend server:
   \`\`\`
   uvicorn api.main:app --reload
   \`\`\`

#### Frontend

1. Navigate to the frontend directory:
   \`\`\`
   cd frontend
   \`\`\`

2. Install dependencies:
   \`\`\`
   npm install
   \`\`\`

3. Start the development server:
   \`\`\`
   npm run dev
   \`\`\`

4. Access the application at http://localhost:3000

## Usage

1. **Upload Data**: Start by uploading your data file or providing a URL
2. **Preprocess Data**: Select preprocessing options to clean and prepare your data
3. **Analyze Data**: Explore your data through visualizations and statistics
4. **Build Models**: Train and evaluate machine learning models
5. **Generate Reports**: Create comprehensive reports with visualizations and insights
6. **Download Results**: Download reports, visualizations, and processed data

## License

This project is licensed under the MIT License - see the LICENSE file for details.
