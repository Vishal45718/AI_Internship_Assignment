# Personal Analytics

A full-stack web application for tracking and analyzing personal developer metrics, including sleep patterns, mood scores, coding productivity, and activity sessions. The app provides AI-powered insights to help developers optimize their habits and performance.

---

## Features

- **Daily Logging**
  - Record daily metrics:
    - Sleep hours
    - Mood scores (1–5)
    - Total coding minutes
    - Productivity scores

- **Activity Tracking**
  - Log detailed activity sessions with:
    - Categories (Coding, Distraction, Generic)
    - App names
    - Durations

- **AI Insights**
  - Automated analysis providing:
    - Sleep debt warnings
    - Distraction pattern detection
    - Productivity trend insights

- **Data Visualization**
  - Interactive charts showing:
    - Productivity trends
    - Sleep patterns
    - Mood variations

- **SQLite Database**
  - Local storage using SQLAlchemy ORM

---

## Tech Stack

### Backend
- Python 3.x
- FastAPI
- SQLAlchemy
- SQLite
- Pydantic
- Pandas
- Scikit-learn

### Frontend
- React 19
- TypeScript
- Vite
- Tailwind CSS
- Recharts
- Axios
- Lucide React

---

## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

---

## Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Create a `.env` file in `backend/` based on `.env.example` and configure OAuth credentials plus session settings.

### OAuth Setup

The app supports Google and GitHub OAuth for authentication. Configure your OAuth providers:

#### Google OAuth
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google+ API
4. Create OAuth 2.0 credentials
5. Set authorized redirect URIs:
   - `http://localhost:8000/api/auth/google/callback` (development)
   - `https://yourdomain.com/api/auth/google/callback` (production)

#### GitHub OAuth
1. Go to [GitHub Settings > Developer settings > OAuth Apps](https://github.com/settings/developers)
2. Create a new OAuth App
3. Set Authorization callback URL:
   - `http://localhost:8000/api/auth/github/callback` (development)
   - `https://yourdomain.com/api/auth/github/callback` (production)

Add the credentials to your `.env` file:
```
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
```

### Production Deployment

For production deployment:
1. Update `FRONTEND_URL` and `CORS_ORIGINS` in your `.env` file
2. Ensure `SECRET_KEY` is set to a secure, random string for JWT signing
3. Update OAuth redirect URIs in provider consoles to use your production domain

---

## Frontend Setup

```bash
cd frontend
npm install
```

---

## Usage

### Run Backend

```bash
cd backend
uvicorn main:app --reload
```

API: http://localhost:8000

---

### Run Frontend

```bash
cd frontend
npm run dev
```

App: http://localhost:5173

---

### Seed Sample Data

The easiest way to generate sample data for the MVP is to click the **"Generate Demo Data"** button directly on the empty Dashboard after logging in. 

Alternatively, you can seed the database manually (for testing without a user session):
```bash
cd backend
python seed.py
```

---

## Project Structure

```
Personal_Analytics/
├── backend/
│   ├── analytics.py
│   ├── database.py
│   ├── main.py
│   ├── models.py
│   ├── requirements.txt
│   └── seed.py
└── frontend/
    ├── public/
    ├── src/
    │   ├── components/
    │   │   ├── Dashboard.tsx
    │   │   └── Login.tsx
    │   ├── context/
    │   │   └── AuthContext.tsx
    │   ├── App.tsx
    │   ├── main.tsx
    │   └── index.css
    ├── package.json
    ├── vite.config.ts
    └── tsconfig.json
```

---

## API Endpoints

### Logs

- `POST /api/logs`
  - Create a new daily log entry

- `GET /api/logs`
  - Retrieve all logs

---

### Authentication (OAuth)

- `GET /api/auth/{provider}/login`
  - Initiate login flow for Google or GitHub
- `GET /api/auth/{provider}/callback`
  - Callback endpoint for OAuth provider to return token
- `GET /api/users/me`
  - Get the currently authenticated user's profile

---

### Insights

- `GET /api/insights`
  - Get AI-generated insights
- `GET /api/analytics/summary`
  - Get comprehensive productivity and trend analytics

---

### Data Initialization

- `POST /api/seed`
  - Generate demo logs and insights for the authenticated user

## Screenshots / Demo

![Dashboard](screenshots/personal_analytics_1.jpg)
![Analytics](screenshots/personal_analytics_2.jpg)

---

## Contributing

```bash
git checkout -b feature/amazing-feature
git commit -m "Add some amazing feature"
git push origin feature/amazing-feature
```

Then open a Pull Request.

---

## License

This project is licensed under the MIT License.

Note: Some parts of this project were developed with the assistance of AI tools.
