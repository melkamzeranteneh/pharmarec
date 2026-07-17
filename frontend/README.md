# PharmaRec Frontend

React + TypeScript + Vite frontend for the PharmaRec Drug Recommendation System.

## Project Structure

```
frontend/
├── public/
│   └── index.html              # HTML entry point
├── src/
│   ├── components/
│   │   ├── DrugSearch.tsx      # Search input with autocomplete
│   │   ├── MethodSelector.tsx  # Dropdown for recommendation method
│   │   ├── RecommendationTable.tsx # Results table
│   │   └── MetricsComparisonTable.tsx # Metrics comparison
│   ├── pages/
│   │   └── Home.tsx            # Main home page
│   ├── App.tsx                 # Root component
│   ├── main.tsx                # Application entry point
│   └── index.css              # Global styles (Tailwind + custom)
├── package.json
├── tsconfig.json
├── tsconfig.node.json
└── vite.config.ts
```

## Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Run Development Server

```bash
npm run dev
```

The frontend will be available at: http://localhost:3000

**Note:** The development server proxies API requests to http://localhost:8000 (backend).

## Running in Production

```bash
# Build
npm run build

# Preview
npm run preview
```

## Configuration

### Proxy Setup

In `vite.config.ts`, API requests to `/api` are proxied to `http://localhost:8000`:

```typescript
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

### Backend Connection

To change the backend URL, modify `API_BASE` in:
- `src/App.tsx`
- `src/pages/Home.tsx`

## Features

### Components

1. **DrugSearch**: Search input with autocomplete for drug names
2. **MethodSelector**: Dropdown to choose recommendation method
3. **RecommendationTable**: Displays recommendations in a table
4. **MetricsComparisonTable**: Shows comparison metrics for all methods

### Pages

1. **Home**: Main page with all components

### UI Features

- **Responsive Design**: Works on mobile and desktop
- **Clean Cards**: White cards with shadows
- **Tailwind CSS**: Utility-first CSS framework
- **Hover Effects**: Interactive feedback
- **Loading States**: Spinners during async operations
- **Error Handling**: User-friendly error messages

## API Endpoints Used

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/drugs` | Get list of drugs |
| GET | `/api/drugs/names` | Get unique drug names |
| POST | `/api/recommend` | Get recommendations |
| GET | `/api/metrics` | Get evaluation metrics |
| GET | `/api/metrics/comparison` | Get comparison table |
| GET | `/api/health` | Check backend health |

## Running the Complete Application

### Terminal 1: Backend

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

Backend will be available at: http://localhost:8000

### Terminal 2: Frontend

```bash
cd frontend
npm run dev
```

Frontend will be available at: http://localhost:3000

Open http://localhost:3000 in your browser.

## Dependencies

### Core
- React 18
- TypeScript 5
- Vite 5

### HTTP Client
- Axios

### Styling
- Tailwind CSS (via CDN)

## Notes

- The application uses functional components with React hooks
- TypeScript provides type safety throughout the application
- Tailwind CSS is used for styling via CDN (no build step required)
- Axios is used for HTTP requests to the backend API
- The proxy setup allows seamless API calls during development
